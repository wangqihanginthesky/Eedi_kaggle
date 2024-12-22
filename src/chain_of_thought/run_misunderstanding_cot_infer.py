
import pandas as pd
import torch
import numpy as np
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm
import argparse
import os

def main(filepath, exp_name, model_path, output_dir):
    df = pd.read_csv(filepath)
    num_gpu = torch.cuda.device_count()
    llm = LLM(
        model_path,
        tensor_parallel_size=num_gpu,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=4096,
        disable_log_stats=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0.7,  # randomness of the sampling
        seed=777,  # Seed for reprodicibility
        skip_special_tokens=True,  # Whether to skip special tokens in the output.
        max_tokens=512,  # Maximum number of tokens to generate per output sequence.
    )
    system_prompt_template = "You are an excellent math teacher about to teach students of year group 1 to 14. The detail of your lesson is as follows. Subject:{first_subject}, Topic: {second_subject}, Subtopic {third_subject}. Your students have made a mistake in the following question. Please explain the mistake step by step briefly and describe the misunderstanding behind the wrong answer at conceptual level. No need to provide the correct way to achieve the answer."
    user_prompt_template = "Question: {question_text}\nCorrect Answer: {correct_text}\nWrong Answer of your students: {answer_text}\n\nExplanation: \nMisunderstanding: "

    all_texts = []
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        first_subject = row['FirstSubjectName']
        second_subject = row['SecondSubjectName']
        third_subject = row['ThirdSubjectName']
        question_text = row['QuestionText']
        correct_text = row['CorrectAnswerText']
        answer_text = row['AnswerText']
        user_prompt = user_prompt_template.format(question_text=question_text, correct_text=correct_text, answer_text=answer_text)
        system_prompt = system_prompt_template.format(first_subject=first_subject, second_subject=second_subject, third_subject=third_subject)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_texts.append(text)

    output = llm.generate(all_texts, sampling_params=sampling_params)
    output_texts = [x.outputs[0].text for x in output]
    df[f'{exp_name}_misunderstanding'] = output_texts
    print('saving file to', os.path.join(output_dir,os.path.basename(filepath)))
    df.to_csv(os.path.join(output_dir,os.path.basename(filepath)), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()
    print(args)
    main(args.filepath, args.exp_name, args.model_path, args.output_dir)