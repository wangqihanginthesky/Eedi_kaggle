import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import json
import os
from transformers import AutoTokenizer
import time
from tqdm.auto import tqdm
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import random

import eedi.training.dataset as eedi_dataset
import eedi.training.model as eedi_model
from eedi.training.dto.config import ExperimentConfig
from eedi.training.optimizer import OptimizerFacade
from eedi.training.scheduler import SchedulerFacade
from eedi.training.metric import KaggleMetric
from eedi.training.semantic_search import SemanticSearcher
from eedi.training.trainer import EediTrainer
import gc
import warnings
from peft import PeftModelForFeatureExtraction
import time

warnings.simplefilter('ignore', FutureWarning)

CORRECT_MISCONCEPTION_ID = 2587

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')


def ddp_infer(exp_config, filepath, misconception_filepath, mapping_embedding_path, pretrain_model_path, model_path, output_dir, debug):
    os.makedirs(output_dir, exist_ok=True)
    seed_everything(exp_config.seed)
    num_gpus = torch.cuda.device_count()
    use_lora = exp_config.use_lora

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    is_mistral_7b = 'Mistral' in exp_config.llm_config.backbone

    if rank == 0:
        print(f'overwriting llm config... backbone: {exp_config.llm_config.backbone} -> {pretrain_model_path}')
    exp_config.llm_config.backbone = pretrain_model_path
    if rank == 0:
        print(f'overwriting params... test_batch_size: {exp_config.test_batch_size} -> 8')
    if is_mistral_7b:
        exp_config.test_batch_size = 1
    else:
        exp_config.test_batch_size = 8
    if rank == 0:
        print(f'overwriting params... num_workers: {exp_config.num_workers} -> 4')
    exp_config.num_workers = 4

    use_amp = exp_config.precision == "16-mixed"

    aug_col = 'a000-llama3-mega-misconception-aug-seed201_misunderstanding'
    df_mapping = pd.read_csv(misconception_filepath)
    df_mapping['MisconceptionName'] = df_mapping['MisconceptionName'] + ' ' + df_mapping[aug_col]
    df_test = pd.read_csv(filepath)

    if rank == 0:
        print(f"Test size: {len(df_test)}")

    tokenizer = AutoTokenizer.from_pretrained(exp_config.llm_config.backbone, trust_remote_code=True, use_fast=False)

    dataset_cls = getattr(eedi_dataset, exp_config.dataset_config.name)
    dataset_test = dataset_cls(df_test, exp_config.dataset_config, tokenizer, phase='test')

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_test, num_replicas=world_size, rank=rank, shuffle=False
    )    
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=exp_config.test_batch_size, sampler=sampler, num_workers=exp_config.num_workers)

    mapping_embedding = np.load(mapping_embedding_path)
    embeddings_mapping = mapping_embedding['embeddings_mapping']
    ids = mapping_embedding['ids']

    model_cls = getattr(eedi_model, exp_config.llm_config.name)


    if not use_lora:
        model = model_cls(exp_config.llm_config, total_steps=None, phase='test', device_map='cpu').half()
        model.to(rank)
        model.load_state_dict(torch.load(model_path, map_location=f'cuda:{rank}')['model_state_dict'])
    else:
        model = model_cls(exp_config.llm_config, total_steps=None, phase='test', device_map=f"cpu", torch_dtype=torch.float16)
        backbone = PeftModelForFeatureExtraction.from_pretrained(model.backbone, model_path)
        model.backbone = backbone
        model.to(rank)
    print(f'loaded model from {model_path}')

    model = DistributedDataParallel(model, device_ids=[rank])

    results = {
        'question_id_answers': None,
        'ids': None,
        'embeddings': None,
        'embeddings_mapping': None,
        'cosine_similarity': None,
    }
    embedding, embedding_mapping, ids, question_id_answers = EediTrainer.test_func_ddp(model, test_loader, None, rank, embeddings_mapping, ids)
    cosine_similarity = SemanticSearcher._cosine_similarity(embedding, embedding_mapping)
    results['question_id_answers'] = question_id_answers
    results['ids'] = ids
    results['embeddings'] = embedding
    results['embeddings_mapping'] = embedding_mapping
    results['cosine_similarity'] = cosine_similarity
    print('saving results to:', f"{output_dir}/results_{exp_config.exp_name}_rank{rank}.pkl")
    pd.to_pickle(results, f"{output_dir}/results_{exp_config.exp_name}_rank{rank}.pkl")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()
    dist.destroy_process_group()



def dp_infer(exp_config, filepath, misconception_filepath, mapping_embedding_path, pretrain_model_path, model_path, output_dir, debug):
    os.makedirs(output_dir, exist_ok=True)
    seed_everything(exp_config.seed)
    rank = 0
    num_gpus = torch.cuda.device_count()
    use_lora = exp_config.use_lora
    is_mistral_7b = 'Mistral' in exp_config.llm_config.backbone

    print(f'overwriting llm config... backbone: {exp_config.llm_config.backbone} -> {pretrain_model_path}')
    exp_config.llm_config.backbone = pretrain_model_path

    print(f'overwriting params... test_batch_size: {exp_config.test_batch_size} -> 8')
    if is_mistral_7b:
        exp_config.test_batch_size = 1
    else:
        exp_config.test_batch_size = 8
 
    print(f'overwriting params... num_workers: {exp_config.num_workers} -> 4')
    exp_config.num_workers = 4

    use_amp = exp_config.precision == "16-mixed"

    aug_col = 'a000-llama3-mega-misconception-aug-seed201_misunderstanding'
    df_mapping = pd.read_csv(misconception_filepath)
    df_mapping['MisconceptionName'] = df_mapping['MisconceptionName'] + df_mapping[aug_col] + df_mapping['a000-qwen-72b-misconception-aug-seed700_misunderstanding'] + df_mapping['a000-llama3-mega-misconception-aug-subject-rag-seed200_misunderstanding'] + df_mapping['a000-4o-misconception-aug'] + df_mapping['a000-llama3-72b-misconception-aug-seed200_misconception_aug']
    df_test = pd.read_csv(filepath)

    print(f"Test size: {len(df_test)}")

    tokenizer = AutoTokenizer.from_pretrained(exp_config.llm_config.backbone, trust_remote_code=True, use_fast=False)

    dataset_cls = getattr(eedi_dataset, exp_config.dataset_config.name)
    dataset_test = dataset_cls(df_test, exp_config.dataset_config, tokenizer, phase='test')


    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=exp_config.test_batch_size, shuffle=False, num_workers=exp_config.num_workers)

    mapping_embedding = np.load(mapping_embedding_path)
    embeddings_mapping = mapping_embedding['embeddings_mapping']
    ids = mapping_embedding['ids']

    model_cls = getattr(eedi_model, exp_config.llm_config.name)
    model = model_cls(exp_config.llm_config, total_steps=None, phase='test', device_map='cpu').half()
    if not use_lora:
        model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
    else:
        backbone = PeftModelForFeatureExtraction.from_pretrained(model.backbone, model_path)
        model.backbone = backbone
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(num_gpus)])
    
    print(f'loaded model from {model_path}')

    results = {
        'question_id_answers': None,
        'ids': None,
        'embeddings': None,
        'embeddings_mapping': None,
        'cosine_similarity': None,
    }
    embedding, embedding_mapping, ids, question_id_answers = EediTrainer.test_func(model, test_loader, None, embeddings_mapping, ids)
    cosine_similarity = SemanticSearcher._cosine_similarity(embedding, embedding_mapping)
    results['question_id_answers'] = question_id_answers
    results['ids'] = ids
    results['embeddings'] = embedding
    results['embeddings_mapping'] = embedding_mapping
    results['cosine_similarity'] = cosine_similarity
    print('saving results to:', f"{output_dir}/results_{exp_config.exp_name}_rank{rank}.pkl")
    pd.to_pickle(results, f"{output_dir}/results_{exp_config.exp_name}_rank{rank}.pkl")

    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--misconception-filepath', type=str, required=True)
    parser.add_argument('--mapping-embedding-path', type=str, required=True)
    parser.add_argument('--pretrain-model-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--ddp', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    exp_config = ExperimentConfig.parse_obj(config_dict)
    print(exp_config)
    debug = args.debug
    if args.ddp:
        ddp_infer(exp_config, args.filepath, args.misconception_filepath, args.mapping_embedding_path, args.pretrain_model_path, args.model_path, args.output_dir, debug)
    else:
        dp_infer(exp_config, args.filepath, args.misconception_filepath, args.mapping_embedding_path, args.pretrain_model_path, args.model_path, args.output_dir, debug)