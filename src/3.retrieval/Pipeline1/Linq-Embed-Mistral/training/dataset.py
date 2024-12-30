from eedi.training.prompt.prompt_loader import PromptLoader
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from tqdm.auto import tqdm
import re

tqdm.pandas()

class EediClsDatasetV2(Dataset):
    def __init__(self, df, dataset_config, tokenizer, phase='train', df_synthetic=None):
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.dataset_config = dataset_config
        self.dataset_params = dataset_config.params
        self.max_tokens = self.dataset_params['max_tokens']
        self.num_classes = dataset_config.num_classes
        self.prompt_loader = PromptLoader(dataset_config.prompt_name)
        self.phase = phase

        self.misconception_synthetic_map = {}
        if df_synthetic is not None:
            for misconception_id, df_slice in df_synthetic.groupby('MisconceptionId'):
                self.misconception_synthetic_map[misconception_id] = df_slice.reset_index(drop=True)

        if self.phase in ('train','valid'):
            print('tokenizing misconception...')
            self.df[['input_ids_misconception','attention_mask_misconception','misconception_prompt']] = self.df.apply(self.tokenize_misconception, axis=1, result_type='expand')
            print('tokenizing answer...')
            self.df[['input_ids_answer','attention_mask_answer','answer_prompt']] = self.df.apply(self.tokenize_answer, axis=1, result_type='expand')

    def tokenize_question(self, row, random=False):
        first_subject = row['FirstSubjectName']
        second_subject = row['SecondSubjectName']
        third_subject = row['ThirdSubjectName']
        construct = row['ConstructName']
        question = row['QuestionText']
        answer = row['AnswerText']
        correct_answer = row['CorrectAnswerText']
        misunderstanding = row['p000-qwen25-32b-instruct-cot_misunderstanding']
        is_not_exist = False
        if isinstance(question, str):
            if not random:
                question_prompt_template = self.prompt_loader.question_prompt
            else:
                question_prompt_template = self.prompt_loader.get_random_question_aug_prompt()

            question_prompt = question_prompt_template.format(
                misunderstanding=misunderstanding,
                first_subject=first_subject,
                second_subject=second_subject,
                third_subject=third_subject,
                construct=construct,
                question=question,
                answer=answer,
                correct_answer=correct_answer
            )
        else:
            llm_infer = row[self.prime_synthetic_col]
            misconception = row['MisconceptionName']
            question_prompt = llm_infer
            is_not_exist = True

        input_ids_question = self.tokenizer(question_prompt, max_length=self.max_tokens, padding="max_length", truncation=True, return_tensors="pt")
        attention_mask_question = np.where(input_ids_question['input_ids'] != self.tokenizer.pad_token_id, 1, 0)
        attention_mask_question = torch.tensor(attention_mask_question).long()

        return (
            input_ids_question['input_ids'].squeeze(),
            attention_mask_question.squeeze(),
            question_prompt,
            is_not_exist,
        )
    def tokenize_other_question(self, row):
        first_subject = row['FirstSubjectName']
        second_subject = row['SecondSubjectName']
        third_subject = row['ThirdSubjectName']
        construct = row['ConstructName']
        question = row['QuestionText']
        answer = row['AnswerText']
        correct_answer = row['CorrectAnswerText']
        misunderstanding = row['misunderstanding']
        is_not_exist = False
        question_prompt = self.prompt_loader.question_prompt.format(
            misunderstanding=misunderstanding,
            first_subject=first_subject,
            second_subject=second_subject,
            third_subject=third_subject,
            construct=construct,
            question=question,
            answer=answer,
            correct_answer=correct_answer
        )

        input_ids_question = self.tokenizer(question_prompt, max_length=self.max_tokens, padding="max_length", truncation=True, return_tensors="pt")
        attention_mask_question = np.where(input_ids_question['input_ids'] != self.tokenizer.pad_token_id, 1, 0)
        attention_mask_question = torch.tensor(attention_mask_question).long()

        return (
            input_ids_question['input_ids'].squeeze(),
            attention_mask_question.squeeze(),
            question_prompt,
            is_not_exist,
        )

    def tokenize_synthetic(self, row):
        datas = []
        for col in self.synthetic_cols:
            text = row[col]
            input_ids = self.tokenizer(text, max_length=self.max_tokens, padding="max_length", truncation=True, return_tensors="pt")
            attention_mask = np.where(input_ids['input_ids'] != self.tokenizer.pad_token_id, 1, 0)
            attention_mask = torch.tensor(attention_mask).long()
            datas.append(input_ids['input_ids'].squeeze())
            datas.append(attention_mask.squeeze())
        return datas


    def tokenize_misconception(self, row):
        misconception = row['MisconceptionName']

        misconception_prompt = self.prompt_loader.misconception_prompt.format(
            Misconception=misconception
        )
        input_ids_misconception = self.tokenizer(misconception_prompt, max_length=self.max_tokens, padding="max_length", truncation=True, return_tensors="pt")
        attention_mask_misconception = np.where(input_ids_misconception['input_ids'] != self.tokenizer.pad_token_id, 1, 0)
        attention_mask_misconception = torch.tensor(attention_mask_misconception).long()

        return (
            input_ids_misconception['input_ids'].squeeze(),
            attention_mask_misconception.squeeze(), 
            misconception_prompt,
        )

    def tokenize_answer(self, row):
        misconception = row['MisconceptionName']
        answer_prompt = self.prompt_loader.answer_prompt.format(Misconception=misconception)
        input_ids_answer = self.tokenizer(answer_prompt, max_length=self.max_tokens, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = input_ids_answer['input_ids'].squeeze()
        input_ids[input_ids == self.tokenizer.pad_token_id] = -100
        attention_mask_answer = np.where(input_ids_answer['input_ids'] != self.tokenizer.pad_token_id, 1, 0)
        attention_mask_answer = torch.tensor(attention_mask_answer).long()

        return (
            input_ids_answer['input_ids'].squeeze(),
            attention_mask_answer.squeeze(),
            answer_prompt,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        misconception_id = row['MisconceptionId']
        is_synthetic = False

        if self.phase == 'train':
            is_synthetic = row['is_synthetic']
            is_not_exist = not isinstance(row["QuestionText"], str)

            if is_synthetic:
                p_other = self.dataset_params['p_other_generate']
                p = torch.rand(1).item()

                input_ids, attention_mask, question_prompt, _ = self.tokenize_question(row)
            else:
                p_synthetic = self.dataset_params['p_synthetic_exist']
                p = torch.rand(1).item()
                p2 = torch.rand(1).item()
                if (p2>p_synthetic) or (misconception_id not in self.misconception_synthetic_map):
                    input_ids, attention_mask, question_prompt, _ = self.tokenize_question(row)
                else:
                    target_synthetic_df = self.misconception_synthetic_map[misconception_id]
#                        if 'sample_weight' in target_synthetic_df.columns:
#                            row_sample = target_synthetic_df.sample(1, weights=target_synthetic_df['sample_weight']).iloc[0]
#                        else:
                    row_sample = self.misconception_synthetic_map[misconception_id].sample(1).iloc[0]
                    input_ids, attention_mask, question_prompt, _ = self.tokenize_question(row_sample)


        else:
            input_ids, attention_mask, question_prompt, _ = self.tokenize_question(row)

        if self.phase in ('train','valid'):
            input_ids_misconception = row['input_ids_misconception']
            input_ids_answer = row['input_ids_answer']
            attention_mask_misconception = row['attention_mask_misconception']
            attention_mask_answer = row['attention_mask_answer']
            labels = torch.nn.functional.one_hot(torch.tensor(misconception_id).long(), num_classes=self.num_classes).float()
            labels_misconception = torch.nn.functional.one_hot(torch.tensor(misconception_id).long(), num_classes=self.num_classes).float()
            question_id_answer = row['QuestionId_Answer']
            if not isinstance(question_id_answer, str):
                question_id_answer = 'dummy'
            return {
                'question_id_answer': question_id_answer,
                'misconception_id': misconception_id,
                'is_synthetic': is_synthetic,
                'input_ids_question': input_ids,
                'input_ids_missconception': input_ids_misconception,
                'input_ids_answer': input_ids_answer,
                'attention_mask_question': attention_mask,
                'attention_mask_misconception': attention_mask_misconception,
                'attention_mask_answer': attention_mask_answer,
                'labels': labels,
                'labels_misconception': labels_misconception,
                'question_prompt': question_prompt,
                'misconception_prompt': row['misconception_prompt'],
                'answer_prompt': row['answer_prompt'],
            }
        elif self.phase == 'test':
            return {
                'question_id_answer': row['QuestionId_Answer'],
                'input_ids_question': input_ids,
                'attention_mask_question': attention_mask,
            }
        else:
            raise ValueError(f'phase {self.phase} not found')