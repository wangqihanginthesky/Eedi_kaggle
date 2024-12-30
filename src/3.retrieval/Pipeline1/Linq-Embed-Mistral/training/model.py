from torch import nn
import torch
import math
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoModel, LlamaTokenizer, BitsAndBytesConfig
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from llm2vec import LLM2Vec

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, params, total_steps, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.params = list(params.values())
        step_params = []
        if total_steps is not None:
            for param in self.params:
                ratio = param['ratio']
                steps = int(total_steps * ratio)
                step_params.extend([param] * steps)
            if len(step_params) < total_steps + 1:
                step_params.extend([self.params[-1]] * (total_steps + 1 - len(step_params)))
        else:
            step_params = self.params
        self.step_params = step_params
        self.current_param_idx = -1
        self.step()

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight2 = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight2)

        self.easy_margin = easy_margin
        self.cos_0 = math.cos(0)
        self.sin_0 = math.sin(0)
        self.th_0 = math.cos(math.pi - 0)
        self.mm_0 = math.sin(math.pi - 0) * 0

    def set_params(self, s, m):
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def step(self):
        self.current_param_idx += 1
        self.set_params(s=self.step_params[self.current_param_idx]['s'], m=self.step_params[self.current_param_idx]['m'])

    def forward(self, input, label, metrics_learning=True):
        # cos(theta) & phi(theta)
        if self.training:
            if metrics_learning:
                cosine = F.linear(F.normalize(input), F.normalize(self.weight))
                sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
                phi = cosine * self.cos_m - sine * self.sin_m
                if self.easy_margin:
                    phi = torch.where(cosine > 0, phi, cosine)
                else:
                    phi = torch.where(cosine > self.th, phi, cosine - self.mm)
                # convert label to one-hot
                #one_hot = torch.zeros(cosine.size(), device='cuda')
                #one_hot.scatter_(1, label.view(-1, 1).long(), 1)
                # torch.where(out_i = {x_i if condition_i else y_i)
                one_hot = label
                output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
                output = self.s * output
            else:
                cosine = F.linear(F.normalize(input), F.normalize(self.weight2))
                output = cosine
        else:
            if metrics_learning:
                cosine = F.linear(F.normalize(input), F.normalize(self.weight))
                output = cosine
                output = self.s * output
            else:
                cosine = F.linear(F.normalize(input), F.normalize(self.weight2))
                output = cosine
        return output

class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, params, total_steps):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.params = list(params.values())
        step_params = []
        if total_steps is not None:
            for param in self.params:
                ratio = param['ratio']
                steps = int(total_steps * ratio)
                step_params.extend([param] * steps)
            if len(step_params) < total_steps + 1:
                step_params.extend([self.params[-1]] * (total_steps + 1 - len(step_params)))
        else:
            step_params = self.params
        self.step_params = step_params
        self.current_param_idx = -1
        self.step()

        self.kernel = nn.Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

        self.cos_0 = math.cos(0)
        self.sin_0 = math.sin(0)
        self.th_0 = math.cos(math.pi - 0)
        self.mm_0 = math.sin(math.pi - 0) * 0
        
    def set_params(self, s, m):
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def step(self):
        self.current_param_idx += 1
        self.set_params(s=self.step_params[self.current_param_idx]['s'], m=self.step_params[self.current_param_idx]['m'])
    
    def l2_norm(self,input, axis = 1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)

        return output

    def forward(self, embbedings, label, metrics_learning=True):
        label = torch.argmax(label, dim=1)
        if self.training:
            embbedings = self.l2_norm(embbedings, axis = 1)
            kernel_norm = self.l2_norm(self.kernel, axis = 0)
            cos_theta = torch.mm(embbedings, kernel_norm)
            cos_theta = cos_theta.clamp(-1, 1).float()  # for numerical stability
            target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
            mask = cos_theta > cos_theta_m

            final_target_logit = torch.where(target_logit > self.th, cos_theta_m, target_logit - self.mm)

            hard_example = cos_theta[mask]
            with torch.no_grad():
                self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
            cos_theta[mask] = hard_example * (self.t + hard_example)
            cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
            output = cos_theta * self.s
        else:
            if metrics_learning:
                embbedings = self.l2_norm(embbedings, axis = 1)
                kernel_norm = self.l2_norm(self.kernel, axis = 0)
                cos_theta = torch.mm(embbedings, kernel_norm)
                cos_theta = cos_theta.clamp(-1, 1).float()  # for numerical stability
                output = self.s * cos_theta
            else:
                embbedings = self.l2_norm(embbedings, axis = 1)
                kernel_norm = self.l2_norm(self.kernel, axis = 0)
                cos_theta = torch.mm(embbedings, kernel_norm)
                cos_theta = cos_theta.clamp(-1, 1).float()  # for numerical stability
                output = cos_theta
        return output


class EediSeq2SeqClsV1(nn.Module):
    def __init__(self, llm_config, phase='train'):
        super(EediSeq2SeqClsV1, self).__init__()
        self.llm_config = llm_config
        self.num_classes = self.llm_config.num_classes
        self.phase = phase
        self.backbone = AutoModelForCausalLM.from_pretrained(self.llm_config.backbone)
        self.backbone.config.pad_token_id = self.backbone.config.eos_token_id

        self.use_metrics_learning = self.llm_config.use_metrics_learning

        self.metrics_learning_module = None        
        if self.use_metrics_learning:
            module_name = self.llm_config.metrics_learning_module.module_name
            params = self.llm_config.metrics_learning_module.params
            if module_name == 'ArcFace':
                self.metrics_learning_module = ArcFace(self.backbone.config.hidden_size, self.num_classes, s=params['s'], m=params['m'])
            else:
                raise ValueError(f'Metrics learning module {module_name} not found')
        else:
            self.metrics_learning_module = nn.Linear(self.backbone.config.hidden_size, self.num_classes)
        
        self.cls_criterion = nn.CrossEntropyLoss()

    def switch_to_test_phase(self):
        self.phase = 'test'
        return

    def switch_to_train_phase(self):
        self.phase = 'train'
        return

    def forward_train(self, input_ids, attention_mask, answer_input_ids, labels):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=answer_input_ids, output_hidden_states=True)
        # TODO: embeddingがあってるかがわからん
        embedding = outputs.hidden_states[-1][:,-1,:]
        if self.use_metrics_learning:
            logits = self.metrics_learning_module(embedding, labels)
        else:
            logits = self.metrics_learning_module(embedding)
        
        cls_loss = self.cls_criterion(logits, labels)
        seq_loss = outputs.loss
        return {
            'cls_loss': cls_loss,
            'seq_loss': seq_loss,
            'outputs': outputs,
            'logits': logits,
            'embeddings': embedding,
        }

    def forward_test(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        embedding = outputs.hidden_states[-1][:,-1,:]
        if self.use_metrics_learning:
            logits = self.metrics_learning_module(embedding, label=None)
        else:
            logits = self.metrics_learning_module(embedding)
        
        return {
            'outputs': outputs,
            'logits': logits,
            'embeddings': embedding
        }

    def forward(self, input_ids, attention_mask, answer_input_ids=None, labels=None):
        if self.phase == 'train':
            return self.forward_train(input_ids, attention_mask,answer_input_ids, labels)
        elif self.phase == 'test':
            return self.forward_test(input_ids, attention_mask)
        else:
            raise ValueError(f'Phase {self.phase} not found')


class EediClsV1(nn.Module):
    def __init__(self, llm_config, phase='train'):
        super(EediClsV1, self).__init__()
        self.llm_config = llm_config
        self.num_classes = self.llm_config.num_classes
        self.phase = phase
        self.backbone = AutoModelForSequenceClassification.from_pretrained(self.llm_config.backbone, num_labels=self.num_classes, torch_dtype="auto", device_map="auto")
        self.use_metrics_learning = self.llm_config.use_metrics_learning

        self.metrics_learning_module = None        
        if self.use_metrics_learning:
            module_name = self.llm_config.metrics_learning_module.module_name
            params = self.llm_config.metrics_learning_module.params
            if module_name == 'ArcFace':
                self.metrics_learning_module = ArcFace(self.backbone.config.hidden_size, self.num_classes, s=params['s'], m=params['m'])
            else:
                raise ValueError(f'Metrics learning module {module_name} not found')
        else:
            self.metrics_learning_module = nn.Linear(self.backbone.config.hidden_size, self.num_classes)
        self.metrics_learning_module = self.metrics_learning_module.bfloat16()
        self.cls_criterion = nn.CrossEntropyLoss()

    def forward_train(self, input_ids, attention_mask, answer_input_ids, labels):
        embedding = self.backbone(input_ids=input_ids, attention_mask=attention_mask)['logits']
        if self.use_metrics_learning:
            logits = self.metrics_learning_module(embedding, labels)
        else:
            logits = self.metrics_learning_module(embedding)
        
        cls_loss = self.cls_criterion(logits, labels)

        return {
            'cls_loss': cls_loss,
            'seq_loss': None,
            'outputs': None,
            'logits': logits,
            'embeddings': embedding,
        }

    def forward_test(self, input_ids, attention_mask):
        embedding = self.backbone(input_ids=input_ids, attention_mask=attention_mask)['logits']
        if self.use_metrics_learning:
            logits = self.metrics_learning_module(embedding, labels)
        else:
            logits = self.metrics_learning_module(embedding)
        
        return {
            'outputs': None,
            'logits': logits,
            'embeddings': embedding
        }

    def forward(self, input_ids, attention_mask, answer_input_ids=None, labels=None):
        if self.phase == 'train':
            return self.forward_train(input_ids, attention_mask,answer_input_ids, labels)
        elif self.phase == 'test':
            return self.forward_test(input_ids, attention_mask)
        else:
            raise ValueError(f'Phase {self.phase} not found')


class EediSeq2SeqV1(nn.Module):
    def __init__(self, llm_config, phase='train'):
        super(EediSeq2SeqV1, self).__init__()
        self.llm_config = llm_config
        self.num_classes = self.llm_config.num_classes
        self.phase = phase
        self.backbone = AutoModelForCausalLM.from_pretrained(self.llm_config.backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_config.backbone,use_fast=False, padding_size='right')

    def switch_to_test_phase(self):
        self.phase = 'test'
        return

    def switch_to_train_phase(self):
        self.phase = 'train'
        return

    def forward_train(self, input_ids, attention_mask, answer_input_ids, labels):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=answer_input_ids, output_hidden_states=True)
        # TODO: embeddingがあってるかがわからん
        embedding = outputs.hidden_states[-1][:,-1,:]
        seq_loss = outputs.loss
        return {
            'cls_loss': None,
            'seq_loss': seq_loss,
            'outputs': outputs,
            'logits': None,
            'embeddings': embedding,
        }

    def forward_test(self, input_ids, attention_mask):
        generated_ids = self.backbone.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=384)
        generated_ids = [
            output_ids[len(model_input_ids):] for model_input_ids, output_ids in zip(input_ids, generated_ids)
        ]
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return {
            'text': generated_text
        }

    def forward(self, input_ids, attention_mask, answer_input_ids=None, labels=None):
        if self.phase == 'train':
            return self.forward_train(input_ids, attention_mask,answer_input_ids, labels)
        elif self.phase == 'test':
            return self.forward_test(input_ids, attention_mask)
        else:
            raise ValueError(f'Phase {self.phase} not found')

class EediClsV2(nn.Module):
    def __init__(self, llm_config, total_steps=None, phase='train',use_lora=False, **kwargs):
        super(EediClsV2, self).__init__()
        self.llm_config = llm_config
        self.num_classes = self.llm_config.num_classes
        self.phase = phase
        self.backbone = AutoModel.from_pretrained(self.llm_config.backbone, trust_remote_code=True, **kwargs)
        if use_lora:
            peft_config = LoraConfig(
                    lora_alpha=128,
                    lora_dropout=0.05,
                    r=64,
                    bias="none",
                    task_type="FEATURE_EXTRACTION",
                    target_modules= ["k_proj", "q_proj", 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()

#        self.backbone.config.pad_token_id = self.backbone.config.eos_token_id
#        self.backbone.score = nn.Identity()
        self.use_metrics_learning = self.llm_config.use_metrics_learning

        self.metrics_learning_module = None        
        if self.use_metrics_learning:
            module_name = self.llm_config.metrics_learning_module.module_name
            params = self.llm_config.metrics_learning_module.params
            if module_name == 'ArcFace':
                self.metrics_learning_module = ArcFace(self.backbone.config.hidden_size, self.num_classes, params = params, total_steps=total_steps)
            elif module_name == 'CurricularFace':
                self.metrics_learning_module = CurricularFace(self.backbone.config.hidden_size, self.num_classes, params = params, total_steps=total_steps)
            else:
                raise ValueError(f'Metrics learning module {module_name} not found')
        else:
            self.metrics_learning_module = nn.Linear(self.backbone.config.hidden_size, self.num_classes)
        self.metrics_learning_module = self.metrics_learning_module
        self.cls_criterion = nn.CrossEntropyLoss()
        self.cls_criterion2 = nn.BCEWithLogitsLoss()

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = (attention_mask.sum(dim=1) - 1).long()
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def foward_embedding(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        embedding = self.last_token_pool(outputs.last_hidden_state, attention_mask)
        return {'embeddings': embedding}

    def forward_train(self, input_ids, attention_mask, answer_input_ids, labels, metrics_learning=True):
        embedding = self.foward_embedding(input_ids, attention_mask)['embeddings']
        if self.use_metrics_learning:
            logits = self.metrics_learning_module(embedding, labels, metrics_learning)
        else:
            logits = self.metrics_learning_module(embedding)
        cls_loss = self.cls_criterion(logits, labels)
        cls_loss2 = self.cls_criterion2(logits, labels)
        seq_loss = torch.tensor(-1.0)

        return {
            'cls_loss': cls_loss,
            'cls_loss2': cls_loss2,
            'seq_loss': seq_loss,
            'outputs': None,
            'logits': logits,
            'embeddings': embedding,
        }

    def forward_test(self, input_ids, attention_mask):
        embedding = self.foward_embedding(input_ids, attention_mask)['embeddings']
#        if self.use_metrics_learning:
#            logits = self.metrics_learning_module(embedding, labels)
#        else:
#            logits = self.metrics_learning_module(embedding)
        
        return {
            'outputs': None,
            'logits': None,
            'embeddings': embedding
        }

    def forward(self, input_ids, attention_mask, answer_input_ids=None, labels=None, metrics_learning=True, phase='test'):
        if phase == 'train':
            return self.forward_train(input_ids, attention_mask,answer_input_ids, labels, metrics_learning)
        elif phase == 'test':
            return self.forward_test(input_ids, attention_mask)
        else:
            raise ValueError(f'Phase {phase} not found')


class EediClsMiniCPMV2(EediClsV2):
    def last_token_pool(self, last_hidden_states, attention_mask):
        s = torch.sum(last_hidden_states * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        reps = s / d
        return reps

class EediClsStellaV2(EediClsV2):
    def __init__(self, llm_config, total_steps=None, phase='train',**kwargs):
        super(EediClsStellaV2, self).__init__(llm_config, total_steps, phase, **kwargs)

        vector_dim = llm_config.params['dim']

        self.use_metrics_learning = self.llm_config.use_metrics_learning

        self.metrics_learning_module = None        
        if self.use_metrics_learning:
            module_name = self.llm_config.metrics_learning_module.module_name
            params = self.llm_config.metrics_learning_module.params
            if module_name == 'ArcFace':
                self.metrics_learning_module = ArcFace(vector_dim, self.num_classes, params = params, total_steps=total_steps)
            else:
                raise ValueError(f'Metrics learning module {module_name} not found')
        else:
            self.metrics_learning_module = nn.Linear(vector_dim, self.num_classes)
        self.metrics_learning_module = self.metrics_learning_module

        vector_linear_directory = f"2_Dense_{vector_dim}"
        self.vector_linear = torch.nn.Linear(in_features=self.backbone.config.hidden_size, out_features=vector_dim)
        vector_linear_dict = {
            k.replace("linear.", ""): v for k, v in
            torch.load(os.path.join(llm_config.backbone, f"{vector_linear_directory}/pytorch_model.bin")).items()
        }
        self.vector_linear.load_state_dict(vector_linear_dict)

    def last_token_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        vectors = self.vector_linear(vectors)
        return vectors
        
    def foward_embedding(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        embedding = self.last_token_pool(outputs[0], attention_mask)
        return {'embeddings': embedding}

class EediRerankerMiniCPMV1(nn.Module):
    def __init__(self, llm_config, phase='train',**kwargs):
        super(EediRerankerMiniCPMV1, self).__init__()
        self.llm_config = llm_config
        self.phase = phase
        self.backbone = AutoModelForSequenceClassification.from_pretrained(self.llm_config.backbone, trust_remote_code=True, **kwargs)
#        self.backbone.config.pad_token_id = self.backbone.config.eos_token_id
        self.cls_criterion = nn.BCEWithLogitsLoss()

    def switch_to_test_phase(self):
        self.phase = 'test'
        return

    def switch_to_train_phase(self):
        self.phase = 'train'
        return

    def forward_train(self, input_ids, attention_mask, answer_input_ids, labels, metrics_learning=True):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        score = outputs.logits
        cls_loss = self.cls_criterion(score, labels)
        return {
            'cls_loss': cls_loss,
            'outputs': None,
            'logits': score,
        }

    def forward_test(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        score = outputs.logits
        return {
            'outputs': None,
            'logits': score,
        }

    def forward(self, input_ids, attention_mask, answer_input_ids=None, labels=None, metrics_learning=True):
        if self.phase == 'train':
            return self.forward_train(input_ids, attention_mask,answer_input_ids, labels, metrics_learning)
        elif self.phase == 'test':
            return self.forward_test(input_ids, attention_mask)
        else:
            raise ValueError(f'Phase {self.phase} not found')


class EediClsV3(nn.Module):
    def __init__(self, llm_config, total_steps=None, phase='train',use_lora=False, **kwargs):
        super(EediClsV3, self).__init__()
        self.llm_config = llm_config
        self.num_classes = self.llm_config.num_classes
        self.phase = phase
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        llm2vec_model = LLM2Vec.from_pretrained(self.llm_config.backbone, trust_remote_code=True, quantization_config=bnb_config, **kwargs)
        self.backbone = llm2vec_model.model
        self.backbone =  prepare_model_for_kbit_training(self.backbone)
        if use_lora:
            peft_config = LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                bias="none",
                lora_dropout=0.05,
                task_type="FEATURE_EXTRACTION",
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()

#        self.backbone.config.pad_token_id = self.backbone.config.eos_token_id
#        self.backbone.score = nn.Identity()
        self.use_metrics_learning = self.llm_config.use_metrics_learning

        self.metrics_learning_module = None        
        if self.use_metrics_learning:
            module_name = self.llm_config.metrics_learning_module.module_name
            params = self.llm_config.metrics_learning_module.params
            if module_name == 'ArcFace':
                self.metrics_learning_module = ArcFace(self.backbone.config.hidden_size, self.num_classes, params = params, total_steps=total_steps)
            elif module_name == 'CurricularFace':
                self.metrics_learning_module = CurricularFace(self.backbone.config.hidden_size, self.num_classes, params = params, total_steps=total_steps)
            else:
                raise ValueError(f'Metrics learning module {module_name} not found')
        else:
            self.metrics_learning_module = nn.Linear(self.backbone.config.hidden_size, self.num_classes)
        self.metrics_learning_module = self.metrics_learning_module
        self.cls_criterion = nn.CrossEntropyLoss()
        self.cls_criterion2 = nn.BCEWithLogitsLoss()

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = (attention_mask.sum(dim=1) - 1).long()
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def foward_embedding(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        embedding = self.last_token_pool(outputs.last_hidden_state, attention_mask)
        return {'embeddings': embedding}

    def forward_train(self, input_ids, attention_mask, answer_input_ids, labels, metrics_learning=True):
        embedding = self.foward_embedding(input_ids, attention_mask)['embeddings']
        if self.use_metrics_learning:
            logits = self.metrics_learning_module(embedding, labels, metrics_learning)
        else:
            logits = self.metrics_learning_module(embedding)
        cls_loss = self.cls_criterion(logits, labels)
        cls_loss2 = self.cls_criterion2(logits, labels)
        seq_loss = torch.tensor(-1.0)

        return {
            'cls_loss': cls_loss,
            'cls_loss2': cls_loss2,
            'seq_loss': seq_loss,
            'outputs': None,
            'logits': logits,
            'embeddings': embedding,
        }

    def forward_test(self, input_ids, attention_mask):
        embedding = self.foward_embedding(input_ids, attention_mask)['embeddings']
#        if self.use_metrics_learning:
#            logits = self.metrics_learning_module(embedding, labels)
#        else:
#            logits = self.metrics_learning_module(embedding)
        
        return {
            'outputs': None,
            'logits': None,
            'embeddings': embedding
        }

    def forward(self, input_ids, attention_mask, answer_input_ids=None, labels=None, metrics_learning=True, phase='test'):
        if phase == 'train':
            return self.forward_train(input_ids, attention_mask,answer_input_ids, labels, metrics_learning)
        elif phase == 'test':
            return self.forward_test(input_ids, attention_mask)
        else:
            raise ValueError(f'Phase {phase} not found')
