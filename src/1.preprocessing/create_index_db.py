import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_id = 'Alibaba-NLP/gte-base-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
model.to(device)

df_subject = pd.read_csv('subject_metadata.csv')

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# No need to add instruction for retrieval documents
documents = []
df_subject_level3 = df_subject[df_subject['Level']==3].reset_index(drop=True)
for subject_name, subject_id in zip(df_subject_level3['Name'].values, df_subject_level3['SubjectId'].values):
    document = f'{subject_id}. {subject_name}'
    documents.append(document)
subject_ids = df_subject_level3['SubjectId'].values.tolist()
input_texts = documents

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

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index = faiss.IndexIDMap(index)
index.add_with_ids(embeddings, subject_ids)

filename = 'faiss_subject.index'
faiss.write_index(index, filename)