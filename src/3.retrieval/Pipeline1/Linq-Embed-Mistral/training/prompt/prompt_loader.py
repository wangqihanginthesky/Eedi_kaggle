import os
import glob
import random

class PromptLoader(object):
    def __init__(self, prompt_name):
        if prompt_name=='qwen':
            print('PromptLoader: qwen')
            self.prompt_dir = os.path.join(os.path.dirname(__file__), prompt_name)
        elif prompt_name=='gte':
            print('PromptLoader: gte')
            self.prompt_dir = os.path.join(os.path.dirname(__file__), prompt_name)
        elif prompt_name=='gte_qwen2':
            print('PromptLoader: gte_qwen2')
            self.prompt_dir = os.path.join(os.path.dirname(__file__), prompt_name)
        elif prompt_name=='minicpm':
            print('PromptLoader: minicpm')
            self.prompt_dir = os.path.join(os.path.dirname(__file__), prompt_name)
        elif prompt_name == 'stella':
            print('PromptLoader: stella')
            self.prompt_dir = os.path.join(os.path.dirname(__file__), prompt_name)
        elif prompt_name == 'bce':
            print('PromptLoader: bce')
            self.prompt_dir = os.path.join(os.path.dirname(__file__), prompt_name)
        elif prompt_name == 'mistral':
            print('PromptLoader: mistral')
            self.prompt_dir = os.path.join(os.path.dirname(__file__), prompt_name)
        else:
            raise ValueError(f'Prompt {prompt_name} not found')

        with open(os.path.join(self.prompt_dir,'question_prompt.txt'), 'r') as f:
            self._question_prompt = f.read()
        
        with open(os.path.join(self.prompt_dir,'misconception_prompt.txt'), 'r') as f:
            self._misconception_prompt = f.read()

        with open(os.path.join(self.prompt_dir,'answer_prompt.txt'), 'r') as f:
            self._answer_prompt = f.read() 

        with open(os.path.join(self.prompt_dir,'construct_prompt.txt'), 'r') as f:
            self._construct_prompt = f.read()

        with open(os.path.join(self.prompt_dir,'subject_prompt.txt'), 'r') as f:
            self._subject_prompt = f.read()

        with open(os.path.join(self.prompt_dir,'rerank_prompt.txt'), 'r') as f:
            self._rerank_prompt = f.read()

        question_aug_prompts = sorted(glob.glob(os.path.join(self.prompt_dir,'question_prompt_aug*.txt')))
        print('question_aug_prompts:', question_aug_prompts)
        self._question_aug_prompts = []
        for question_aug_prompt in question_aug_prompts:
            with open(question_aug_prompt, 'r') as f:
                self._question_aug_prompts.append(f.read())

    @property
    def question_prompt(self):
        return self._question_prompt
    
    @property
    def misconception_prompt(self):
        return self._misconception_prompt

    @property
    def answer_prompt(self):
        return self._answer_prompt
    
    @property
    def construct_prompt(self):
        return self._construct_prompt
    
    @property
    def subject_prompt(self):
        return self._subject_prompt

    @property
    def rerank_prompt(self):
        return self._rerank_prompt

    @property
    def question_aug_prompts(self):
        return self._question_aug_prompts

    def get_random_question_aug_prompt(self):
        return random.choice(self._question_aug_prompts)

    