import faiss
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import argparse

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import os

def main(filepath, index_db_path, subject_master_path, model_path, output_dir):
    k=1
    index = faiss.read_index(index_db_path)
    df_subject_master = pd.read_csv(subject_master_path)
    subject_id_name = df_subject_master.set_index('SubjectId')['Name'].to_dict()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    df_query = pd.read_csv(filepath)

    def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    queries = []
    for subject_name, subject_id in zip(df_query['SubjectName'].values, df_query['SubjectId'].values):
        query = f'{subject_id}. {subject_name}'
        queries.append(query)

    input_texts = queries

    # Tokenize the input texts
    batch_size = 16
    max_length = 384
    all_embeddings = []
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:i + batch_size]
        batch_dict = tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(model.device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).detach().cpu()
        all_embeddings.append(embeddings)

    embeddings = torch.cat(all_embeddings, dim=0)
    embeddings = F.normalize(embeddings, p=2, dim=1).numpy()

    distances, indices = index.search(embeddings, k)

    third_idx = indices.flatten()
    third_name = []
    for i in third_idx:
        third_name.append(subject_id_name[i])

    second_idx = []
    second_name = []
    for i in third_idx:
        row = df_subject_master[df_subject_master['SubjectId'] == i]
        parent_id = row['ParentId'].values[0]
        row_parent = df_subject_master[df_subject_master['SubjectId'] == parent_id]
        second_idx.append(row_parent['SubjectId'].values[0])
        second_name.append(row_parent['Name'].values[0])

    first_idx = []
    first_name = []
    for i in second_idx:
        row = df_subject_master[df_subject_master['SubjectId'] == i]
        parent_id = row['ParentId'].values[0]
        row_parent = df_subject_master[df_subject_master['SubjectId'] == parent_id]
        first_idx.append(row_parent['SubjectId'].values[0])
        first_name.append(row_parent['Name'].values[0])

    df_query['FirstSubjectId'] = first_idx
    df_query['FirstSubjectName'] = first_name
    df_query['SecondSubjectId'] = second_idx
    df_query['SecondSubjectName'] = second_name
    df_query['ThirdSubjectId'] = third_idx
    df_query['ThirdSubjectName'] = third_name
    print('saving file to', os.path.join(output_dir,os.path.basename(filepath)))
    df_query.to_csv(os.path.join(output_dir,os.path.basename(filepath)), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--index-db-path', type=str, required=True)
    parser.add_argument('--subject-master-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()
    print(args)
    main(args.filepath, args.index_db_path, args.subject_master_path, args.model_path, args.output_dir)