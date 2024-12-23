import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import argparse

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import os

def main(filepath, output_dir):
    df = pd.read_csv(filepath)
    rows = []
    has_misconception = 'MisconceptionAId' in df.columns
    for i in range(len(df)):
        row = df.iloc[i]
        qid = row['QuestionId']
        cid = row['ConstructId']
        cname = row['ConstructName']
        sid = row['SubjectId']
        sname = row['SubjectName']
        question = row['QuestionText']
        correct_ans = row['CorrectAnswer']
        correct_ans_text = row[f'Answer{correct_ans}Text']
        first_sid = row['FirstSubjectId']
        first_sname = row['FirstSubjectName']
        second_sid = row['SecondSubjectId']
        second_sname = row['SecondSubjectName']
        third_sid = row['ThirdSubjectId']
        third_sname = row['ThirdSubjectName']

        for ans in ['A','B','C','D']:
            question_id_answer = f'{qid}_{ans}'
            ans_text = row[f'Answer{ans}Text']
            if ans == correct_ans:
                correct = 1
            else:
                correct = 0
            if has_misconception:
                misconception_id = row[f'Misconception{ans}Id']
            else:
                misconception_id = -1
            rows.append([question_id_answer ,qid, cid, cname, sid, sname, question, ans, ans_text, correct, correct_ans, correct_ans_text, first_sid, first_sname, second_sid, second_sname, third_sid, third_sname, misconception_id])

    df = pd.DataFrame(rows, columns=['QuestionId_Answer', 'QuestionId', 'ConstructId', 'ConstructName', 'SubjectId', 'SubjectName', 'QuestionText', 'Answer', 'AnswerText', 'Correct', 'CorrectAnswer', 'CorrectAnswerText', 'FirstSubjectId', 'FirstSubjectName', 'SecondSubjectId', 'SecondSubjectName', 'ThirdSubjectId', 'ThirdSubjectName', 'MisconceptionId'])
    df = df[(np.logical_not(np.isnan(df['MisconceptionId'].values))) & ((df['Correct']==0).values)].reset_index(drop=True)
    print('saving file to', os.path.join(output_dir,os.path.basename(filepath)))
    df.to_csv(os.path.join(output_dir,os.path.basename(filepath)), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()
    print(args)
    main(args.filepath, args.output_dir)