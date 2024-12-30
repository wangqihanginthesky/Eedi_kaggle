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
warnings.simplefilter('ignore', FutureWarning)

CORRECT_MISCONCEPTION_ID = 2587
K = 25

def sample_train_balanced(df_train, train_misconception_ids, misconception_name_mapping):
    df_train = df_train.copy()
    df_train['sample_weight'] = df_train['is_synthetic'].apply(lambda x: 1.0 if x else 0.2)
    df_map = {}
    for misconception_id, df_slice in df_train.groupby('MisconceptionId'):
        df_map[misconception_id] = df_slice.reset_index(drop=True)
    dfs = []
    for misconception_id in df_map.keys():
        df_slice = df_map[misconception_id]
        df_slice = df_slice.sample(1, weights=df_slice['sample_weight']).reset_index(drop=True)
        dfs.append(df_slice)
    df_train_sample = pd.concat(dfs).reset_index(drop=True)
    return df_train_sample

def sample_train(df_train, train_misconception_ids, misconception_name_mapping):
    df_real = df_train[df_train['fold']>=0].reset_index(drop=True)
    df_fake = df_train[df_train['fold']==-1].reset_index(drop=True)
    df_nan = df_train[df_train['fold']==-2].reset_index(drop=True)
    fake_df_map = {}
    for misconception_id, df_slice in df_fake.groupby('MisconceptionId'):
        fake_df_map[misconception_id] = df_slice
    nan_df_map = {}
    for misconception_id, df_slice in df_nan.groupby('MisconceptionId'):
        nan_df_map[misconception_id] = df_slice

    missed_misconception_ids = set(misconception_name_mapping.keys()) - train_misconception_ids
    fake_misconception_ids = set(df_fake['MisconceptionId'].unique())
    dfs = []
    for missed_misconception_id in missed_misconception_ids:
        if missed_misconception_id in fake_misconception_ids:
            df_fake_sample = fake_df_map[missed_misconception_id]
        else:
            df_fake_sample = nan_df_map[missed_misconception_id]
#        if 'sample_weight' in df_fake_sample.columns:
#            df_fake_sample = df_fake_sample.sample(1,weights=df_fake_sample['sample_weight']).reset_index(drop=True)
#        else:
        df_fake_sample = df_fake_sample.sample(1).reset_index(drop=True)
        dfs.append(df_fake_sample)
    
    for misconception_id in train_misconception_ids:
        if misconception_id in fake_misconception_ids:
            df_fake_sample = fake_df_map[misconception_id]
#        if 'sample_weight' in df_fake_sample.columns:
#            df_fake_sample = df_fake_sample.sample(1,weights=df_fake_sample['sample_weight']).reset_index(drop=True)
#        else:
        df_fake_sample = df_fake_sample.sample(1).reset_index(drop=True)
        p = random.random()
        if p > 0.9:
            dfs.append(df_fake_sample)
    
    df_sample = pd.concat(dfs).reset_index(drop=True)
    df_train_sample = pd.concat([df_real, df_sample]).reset_index(drop=True)
    df_train_sample = df_train_sample.fillna('')
    return df_train_sample

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')

def half_ckpt(state_dict):
    state_dict = state_dict.copy()
    for k in state_dict.keys():
        state_dict[k] = state_dict[k].half()
    return state_dict

def main(exp_config, filepath, filepath_misconception, fold, output_dir, debug):
    seed_everything(exp_config.seed)
    log_path = os.path.join(output_dir, f'log_fold{fold}.txt')

    os.makedirs(output_dir, exist_ok=True)
    use_amp = exp_config.precision == "16-mixed"
    use_dp = False
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        use_dp = True
    device_ids = list(range(num_gpus))
    use_lora = exp_config.use_lora

    aug_col = 'a000-llama3-mega-misconception-aug-seed201_misunderstanding'
    df_mapping = pd.read_csv(filepath_misconception)
    df_mapping['MisconceptionName'] = df_mapping['MisconceptionName'] + df_mapping[aug_col] + df_mapping['a000-qwen-72b-misconception-aug-seed700_misunderstanding'] + df_mapping['a000-llama3-mega-misconception-aug-subject-rag-seed200_misunderstanding'] + df_mapping['a000-4o-misconception-aug'] + df_mapping['a000-llama3-72b-misconception-aug-seed200_misconception_aug']
    df_mapping['MisconceptionNameChinese'] = df_mapping['MisconceptionNameChinese'] + df_mapping['a000-qwen-72b-misconception-aug-seed700_misunderstanding_chinese'] + df_mapping['a000-llama3-mega-misconception-aug-seed201_misunderstanding_chinese'] + df_mapping['a000-llama3-mega-misconception-aug-subject-rag-seed200_misunderstanding_chinese'] + df_mapping['a000-4o-misconception-aug_chinese'] + df_mapping['a000-llama3-72b-misconception-aug-seed200_misunderstanding_chinese']
    misconception_name_mapping = df_mapping.set_index('MisconceptionId')['MisconceptionName'].to_dict()
    df = pd.read_csv(filepath)
    df_synthetic = df[df['is_synthetic']].reset_index(drop=True)

    df_train = df
    df_train = df_mapping.merge(df_train, on='MisconceptionId', how='left')
    df_train['fold'] = df_train['fold'].fillna(-2)
    df_train['is_synthetic'] = df_train['is_synthetic'].fillna(True)
    train_misconception_ids = set(df_train[(df_train['fold']>=0)]['MisconceptionId'].unique())

    print(f"Fold: {fold}, Train size: {len(df_train)}")

    tokenizer = AutoTokenizer.from_pretrained(exp_config.llm_config.backbone, trust_remote_code=True, use_fast=False)
    dataset_cls = getattr(eedi_dataset, exp_config.dataset_config.name)
    df_train_sample = sample_train(df_train, train_misconception_ids, misconception_name_mapping)
    train_step_size = len(df_train_sample) // exp_config.train_batch_size if len(df_train_sample) % exp_config.train_batch_size == 0 else len(df_train_sample) // exp_config.train_batch_size + 1
    print(f"Train size: {len(df_train_sample)}, Train step size: {train_step_size}")

    model_cls = getattr(eedi_model, exp_config.llm_config.name)
    model = model_cls(exp_config.llm_config, total_steps=train_step_size*exp_config.max_epochs, use_lora=use_lora, device_map='auto')
    if use_dp:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        model = model.cuda()
    optimizer = OptimizerFacade.create_optimizer(exp_config.optimizer_config, model)
    scheduler = SchedulerFacade.create_scheduler(exp_config.lr_scheduler_config, optimizer, train_step_size)

    scaler=None
    if use_amp:
        scaler = amp.GradScaler()

    start_epoch = 1
    metric_best_1 = 0.0
    metric_best_2 = 0.0
    metric_best = 0.0
    embedding_mapping, ids = None, None

    for epoch in range(start_epoch, exp_config.max_epochs+1):
        print(time.ctime(), 'Epoch:', epoch)
#        if embedding_mapping is None and ids is None:
#            embedding_mapping, ids = EediTrainer.get_misconception_embeddings(model, test_loader_mapping)
#        if exp_config.dataset_config.params.get('use_negative_samples', False):
#            synthetic_negative = exp_config.dataset_config.params['synthetic_negative']
#            negative_top_k = exp_config.dataset_config.params['negative_top_k']
#            negative_samples = exp_config.dataset_config.params['negative_samples']
#            predict_ids_mapping = SemanticSearcher.cosine_similarity_search(embedding_mapping, embedding_mapping, ids)
#            negative_params = {
#                'input_ids_mapping': input_ids_mapping,
#                'attention_mask_mapping': attention_mask_mapping,
#                'negative_top_k': negative_top_k,
#                'negative_samples': negative_samples,
#                'misconception_name_mapping': misconception_name_mapping,
#                'predict_ids_mapping': np.array(predict_ids_mapping),
#                'synthetic_negative': synthetic_negative
#            }
#        else:
#            negative_params = None
        negative_params = None
        df_train_sample = sample_train(df_train, train_misconception_ids, misconception_name_mapping)
        dataset_train = dataset_cls(df_train_sample, exp_config.dataset_config, tokenizer, phase='train',df_synthetic=df_synthetic)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=exp_config.train_batch_size, shuffle=True, num_workers=exp_config.num_workers, worker_init_fn=lambda x: np.random.seed())

        train_loss,  train_question_seq_loss, train_misconception_seq_loss, train_question_cls_loss, train_misconception_cls_loss = EediTrainer.train_func(model, train_loader, optimizer, scheduler, exp_config.lr_scheduler_config, scaler, debug, negative_params, exp_config.accumulate_grad_batchs)
        content = f'epoch: {epoch}, fold: {fold}, train_loss: {train_loss}, train_question_seq_loss: {train_question_seq_loss}, train_misconception_seq_loss: {train_misconception_seq_loss}, train_question_cls_loss: {train_question_cls_loss}, train_misconception_cls_loss: {train_misconception_cls_loss}'
        print(content)
        with open(log_path, 'a') as appender:
            appender.write(content + '\n')
#        embedding_mapping, ids = EediTrainer.get_misconception_embeddings(model, test_loader_mapping)
#        if epoch % exp_config.check_val_every_n_epoch == 0:
#            valid_loss_1, valid_question_seq_loss_1, valid_misconception_seq_loss_1, valid_question_cls_loss_1, valid_misconception_cls_loss_1 = EediTrainer.eval_func(model, valid_loader_1)
#            valid_loss_2, valid_question_seq_loss_2, valid_misconception_seq_loss_2, valid_question_cls_loss_2, valid_misconception_cls_loss_2 = EediTrainer.eval_func(model, valid_loader_2)
#            embedding_1, embedding_mapping_1, ids_1, _ = EediTrainer.test_func(model, test_loader_1, test_loader_mapping, embedding_mapping, ids)
#            predict_ids_1 = SemanticSearcher.cosine_similarity_search(embedding_1, embedding_mapping_1, ids_1)
#            metric_1 = KaggleMetric.mapk(ground_truth_ids_1,predict_ids_1,k=K)
#            recall_1 = KaggleMetric.recall_at_k(ground_truth_ids_1,predict_ids_1,k=K)
#            embedding_2, embedding_mapping_2, ids_2, _ = EediTrainer.test_func(model, test_loader_2, test_loader_mapping, embedding_mapping, ids)
#            predict_ids_2 = SemanticSearcher.cosine_similarity_search(embedding_2, embedding_mapping_2, ids_2)
#            metric_2 = KaggleMetric.mapk(ground_truth_ids_2,predict_ids_2,k=K)
#            recall_2 = KaggleMetric.recall_at_k(ground_truth_ids_2,predict_ids_2,k=K)
#            metric = (metric_1 + metric_2) / 2
#            content = f'epoch: {epoch}, fold: {fold}, valid_loss_exist: {valid_loss_1}, valid_question_seq_loss_exist: {valid_question_seq_loss_1}, valid_misconception_seq_loss_exist: {valid_misconception_seq_loss_1}, valid_question_cls_loss_exist: {valid_question_cls_loss_1}, valid_misconception_cls_loss_exist: {valid_misconception_cls_loss_1}'
#            print(content)
#            with open(log_path, 'a') as appender:
#                appender.write(content + '\n')
#            content = f'epoch: {epoch}, fold: {fold}, valid_loss_not_exist: {valid_loss_2}, valid_question_seq_loss_not_exist: {valid_question_seq_loss_2}, valid_misconception_seq_loss_not_exist: {valid_misconception_seq_loss_2}, valid_question_cls_loss_not_exist: {valid_question_cls_loss_2}, valid_misconception_cls_loss_not_exist: {valid_misconception_cls_loss_2}'
#            print(content)
#            with open(log_path, 'a') as appender:
#                appender.write(content + '\n')
#            content = f'epoch: {epoch}, fold: {fold}, kaggle_metric_exist: {metric_1}, kaggle_metric_not_exist: {metric_2}, kaggle_metric: {metric}, recall_exist: {recall_1}, recall_not_exist: {recall_2}'    
#            print(content)
#            with open(log_path, 'a') as appender:
#                appender.write(content + '\n\n')
#            if metric_best_1 < metric_1:
#                print(f'metric_exist_best ({metric_best_1} --> {metric_1}). Saving model ...')
#                metric_best_1 = metric_1
#                if not debug:
#                    torch.save(
#                        {
#                            'epoch': epoch,
#                            'model_state_dict': half_ckpt(model.state_dict()),
#                            'score_best_exist': metric_1,
#                            'score_best_not_exist': metric_2,
#                            'score_best': metric,
#                        },
#                        os.path.join(output_dir, f'{exp_config.exp_name}_best_exist_fold{fold}.pth')
#                    )
#            if metric_best_2 < metric_2:
#                print(f'metric_not_exist_best ({metric_best_2} --> {metric_2}). Saving model ...')
#                metric_best_2 = metric_2
#                if not debug:
#                    torch.save(
#                        {
#                            'epoch': epoch,
#                            'model_state_dict': half_ckpt(model.state_dict()),
#                            'score_best_exist': metric_1,
#                            'score_best_not_exist': metric_2,
#                            'score_best': metric,
#                        },
#                        os.path.join(output_dir, f'{exp_config.exp_name}_best_not_exist_fold{fold}.pth')
#                    )
#            if metric_best < metric:
#                print(f'metric_best ({metric_best} --> {metric}). Saving model ...')
#                metric_best = metric
#                if not debug:
#                    torch.save(
#                        {
#                            'epoch': epoch,
#                            'model_state_dict': half_ckpt(model.state_dict()),
#                            'score_best_exist': metric_1,
#                            'score_best_not_exist': metric_2,
#                            'score_best': metric,
#                        },
#                        os.path.join(output_dir, f'{exp_config.exp_name}_best_fold{fold}.pth')
#                    )
        if debug:
            break
        # Save Last
        if not debug:
            if not use_lora:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': half_ckpt(model.state_dict()),
    #                    'score_best_exist': metric_1,
    #                    'score_best_not_exist': metric_2,
    #                    'score_best': metric,
                    },
                    os.path.join(output_dir, f'{exp_config.exp_name}_last_fold{fold}.pth')
                )
            else:
                if epoch >= exp_config.max_epochs-10:
                    model.backbone.save_pretrained(os.path.join(output_dir, f'{exp_config.exp_name}_epoch_{epoch}_fold{fold}'))

    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--filepath-misconception', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    exp_config = ExperimentConfig.parse_obj(config_dict)
    print(exp_config)
    debug = args.debug
    main(exp_config, args.filepath,args.filepath_misconception, args.fold, args.output_dir, debug)