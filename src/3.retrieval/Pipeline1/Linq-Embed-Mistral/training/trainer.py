import torch
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
import torch.cuda.amp as amp
from eedi.training.loss import MultipleNegativeRankingLoss


class EediTrainer(object):    
    @staticmethod
    def train_func(model, loader_train, optimizer, scheduler, lr_scheduler_config, scaler=None, debug=False, negative_params=None, accumulate_grad_batchs=1):
        if lr_scheduler_config.interval == "epoch":
            scheduler.step()
        model.train()
        train_loss = []
        train_question_cls_loss = []
        train_question_seq_loss = []
        train_misconception_cls_loss = []
        train_misconception_seq_loss = []
        bar = tqdm(loader_train)

        use_amp = scaler is not None
        for i,batch in enumerate(bar):
            if debug and i >= 5:
                break

            misconcetion_ids = batch['misconception_id']
            is_synthetics = batch['is_synthetic']
            input_ids_question = batch['input_ids_question'].cuda()
            attention_mask_question = batch['attention_mask_question'].cuda()
            input_ids_misconception = batch['input_ids_missconception'].cuda()
            attention_mask_misconception = batch['attention_mask_misconception'].cuda()
            labels = batch['labels'].cuda()
            labels_misconception = batch['labels_misconception'].cuda()

            with amp.autocast(enabled=use_amp):
                result_question = model(input_ids=input_ids_question, attention_mask=attention_mask_question, answer_input_ids=None, labels=labels, phase='train')
                result_misconception = model(input_ids=input_ids_misconception, attention_mask=attention_mask_misconception, answer_input_ids=None, labels=labels_misconception, phase='train')
                loss_negative_samples = 0
                if negative_params is not None:
                    negative_criterion = MultipleNegativeRankingLoss(margin=1.0)
                    anchor_embedding = result_question['embeddings']
                    positive_embedding = result_misconception['embeddings']

                    negative_samples = negative_params['negative_samples']
                    predict_ids_mapping = negative_params['predict_ids_mapping']
                    misconception_name_mapping = negative_params['misconception_name_mapping']
                    negative_top_k = negative_params['negative_top_k']
                    input_ids_mapping = negative_params['input_ids_mapping']
                    attention_mask_mapping = negative_params['attention_mask_mapping']
                    predict_ids_mapping = predict_ids_mapping[:, :negative_top_k]
                    synthetic_negative = negative_params['synthetic_negative']

                    for misconception_id, is_synthetic in zip(misconcetion_ids, is_synthetics):
                        if (not synthetic_negative) and (is_synthetic):
                            continue
                        predict_ids_candidates = predict_ids_mapping[misconception_id]
                        predict_ids_negative = np.random.choice(predict_ids_candidates, negative_samples, replace=False)
                        negative_samples_input_ids = [input_ids_mapping[predict_id] for predict_id in predict_ids_negative]
                        negative_samples_attention_mask = [attention_mask_mapping[predict_id] for predict_id in predict_ids_negative]
                        negative_samples_input_ids = torch.stack(negative_samples_input_ids).cuda()
                        negative_samples_attention_mask = torch.stack(negative_samples_attention_mask).cuda()
                        result_negative = model.foward_embedding(input_ids=negative_samples_input_ids, attention_mask=negative_samples_attention_mask)
                        negative_embedding = result_negative['embeddings']
                        loss_negative_sample = negative_criterion(anchor_embedding, positive_embedding, negative_embedding)
                        loss_negative_samples += loss_negative_sample/len(misconcetion_ids)
                    
                #loss = 2*(result_question['cls_loss'] + result_question['seq_loss'])/3 + (result_misconception['cls_loss'] + result_misconception['seq_loss'])/3
                loss = (result_question['cls_loss'] + result_misconception['cls_loss'])/2 + loss_negative_samples
                train_loss.append(loss.item())
                train_question_cls_loss.append(result_question['cls_loss'].item())
                train_misconception_cls_loss.append(result_misconception['cls_loss'].item())
                train_question_seq_loss.append(result_question['seq_loss'].item())
                train_misconception_seq_loss.append(result_misconception['seq_loss'].item())
                if use_amp:
                    scaler.scale(loss).backward()
                    if (i+1) % accumulate_grad_batchs == 0:
                        scaler.unscale_(optimizer)
                        #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else: 
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    if (i+1) % accumulate_grad_batchs == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    #loss.backward()
                    #optimizer.step()
#                if model.use_metrics_learning:
#                    model.metrics_learning_module.step()
                if lr_scheduler_config.interval == "step":
                    scheduler.step()
                bar.set_description(f'smth:{np.mean(train_loss[-30:]):.4f}')

        return np.mean(train_loss), np.mean(train_question_seq_loss), np.mean(train_misconception_seq_loss), np.mean(train_question_cls_loss), np.mean(train_misconception_cls_loss)

    @staticmethod
    def train_rerank_func(model, loader_train, optimizer, scheduler, lr_scheduler_config, scaler=None, debug=False):
        if lr_scheduler_config.interval == "epoch":
            scheduler.step()
        model.train()
        train_loss = []
        bar = tqdm(loader_train)

        use_amp = scaler is not None
        for i,batch in enumerate(bar):
            if debug and i >= 5:
                break
            optimizer.zero_grad()
            misconcetion_ids = batch['misconception_id']
            input_ids_question = batch['input_ids_question'].cuda()
            attention_mask_question = batch['attention_mask_question'].cuda()
            labels = batch['labels'].cuda()

            with amp.autocast(enabled=use_amp):
                result_question = model(input_ids=input_ids_question, attention_mask=attention_mask_question, answer_input_ids=None, labels=labels, phase='train')
                #loss = 2*(result_question['cls_loss'] + result_question['seq_loss'])/3 + (result_misconception['cls_loss'] + result_misconception['seq_loss'])/3
                loss = result_question['cls_loss']
                train_loss.append(loss.item())
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    scaler.step(optimizer)
                    scaler.update()
                else: 
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    #loss.backward()
                    #optimizer.step()
                if lr_scheduler_config.interval == "step":
                    scheduler.step()
                bar.set_description(f'smth:{np.mean(train_loss[-30:]):.4f}')

        return np.mean(train_loss)

    @staticmethod
    def train_func_subject(model, loader_train, optimizer, scheduler, lr_scheduler_config, scaler=None, debug=False):
        if lr_scheduler_config.interval == "epoch":
            scheduler.step()
        model.train()
        train_loss = []
        train_question_cls_loss = []
        train_question_seq_loss = []
        train_misconception_cls_loss = []
        train_misconception_seq_loss = []
        bar = tqdm(loader_train)

        use_amp = scaler is not None
        for i,batch in enumerate(bar):
            if debug and i >= 5:
                break
            optimizer.zero_grad()
            input_ids_question = batch['input_ids_question'].cuda()
            attention_mask_question = batch['attention_mask_question'].cuda()
            input_ids_misconception = batch['input_ids_missconception'].cuda()
            attention_mask_misconception = batch['attention_mask_misconception'].cuda()
            labels = batch['labels'].cuda()
            labels_misconception = batch['labels_misconception'].cuda()

            with amp.autocast(enabled=use_amp):
                result_question = model(input_ids=input_ids_question, attention_mask=attention_mask_question, answer_input_ids=None, labels=labels, metrics_learning=False, phase='train')
                result_misconception = model(input_ids=input_ids_misconception, attention_mask=attention_mask_misconception, answer_input_ids=None, labels=labels_misconception, phase='train')
                #loss = 2*(result_question['cls_loss'] + result_question['seq_loss'])/3 + (result_misconception['cls_loss'] + result_misconception['seq_loss'])/3
                loss = (result_question['cls_loss2'] + result_misconception['cls_loss'])/2
                train_loss.append(loss.item())
                train_question_cls_loss.append(result_question['cls_loss2'].item())
                train_misconception_cls_loss.append(result_misconception['cls_loss'].item())
                train_question_seq_loss.append(result_question['seq_loss'].item())
                train_misconception_seq_loss.append(result_misconception['seq_loss'].item())
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    scaler.step(optimizer)
                    scaler.update()
                else: 
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    #loss.backward()
                    #optimizer.step()
                if model.use_metrics_learning:
                    model.metrics_learning_module.step()
                if lr_scheduler_config.interval == "step":
                    scheduler.step()
                bar.set_description(f'smth:{np.mean(train_loss[-30:]):.4f}')

        return np.mean(train_loss), np.mean(train_question_seq_loss), np.mean(train_misconception_seq_loss), np.mean(train_question_cls_loss), np.mean(train_misconception_cls_loss)

    @staticmethod
    def eval_func(model, loader_valid):
        bar = tqdm(loader_valid)
        model.eval()
        valid_loss = []
        valid_question_cls_loss = []
        valid_question_seq_loss = []
        valid_misconception_cls_loss = []
        valid_misconception_seq_loss = []
        for i,batch in enumerate(bar):
            input_ids_question = batch['input_ids_question'].cuda()
            attention_mask_question = batch['attention_mask_question'].cuda()
            input_ids_misconception = batch['input_ids_missconception'].cuda()
            attention_mask_misconception = batch['attention_mask_misconception'].cuda()
            labels = batch['labels'].cuda()
            labels_misconception = batch['labels_misconception'].cuda()
            
            with torch.no_grad(), amp.autocast(enabled=True):
                result_question = model(input_ids=input_ids_question, attention_mask=attention_mask_question, answer_input_ids=None, labels=labels, phase='train')
                result_misconception = model(input_ids=input_ids_misconception, attention_mask=attention_mask_misconception, answer_input_ids=None, labels=labels_misconception, phase='train')
                loss_cls = result_question['cls_loss']
                loss_missconception_cls = result_misconception['cls_loss']
                loss_seq = result_question['seq_loss']
                loss_missconception_seq = result_misconception['seq_loss']
                loss = loss_cls
                valid_loss.append(loss.item())
                valid_question_cls_loss.append(loss_cls.item())
                valid_misconception_cls_loss.append(loss_missconception_cls.item())
                valid_question_seq_loss.append(loss_seq.item())
                valid_misconception_seq_loss.append(loss_missconception_seq.item())

                loss = loss
                bar.set_description(f'smth:{np.mean(valid_loss[-30:]):.4f}')
        return np.mean(valid_loss), np.mean(valid_question_seq_loss), np.mean(valid_misconception_seq_loss), np.mean(valid_question_cls_loss), np.mean(valid_misconception_cls_loss)

    @staticmethod
    def eval_rerank_func(model, loader_valid):
        bar = tqdm(loader_valid)
        model.eval()
        valid_loss = []
        for i,batch in enumerate(bar):
            input_ids_question = batch['input_ids_question'].cuda()
            attention_mask_question = batch['attention_mask_question'].cuda()
            labels = batch['labels'].cuda()
            
            with torch.no_grad(), amp.autocast(enabled=True):
                result_question = model(input_ids=input_ids_question, attention_mask=attention_mask_question, answer_input_ids=None, labels=labels, phase='train')
                loss = result_question['cls_loss']
                valid_loss.append(loss.item())
                loss = loss
                bar.set_description(f'smth:{np.mean(valid_loss[-30:]):.4f}')
        return np.mean(valid_loss)

    @staticmethod
    def eval_func_subject(model, loader_valid):
        bar = tqdm(loader_valid)
        model.eval()
        valid_loss = []
        valid_question_cls_loss = []
        valid_question_seq_loss = []
        valid_misconception_cls_loss = []
        valid_misconception_seq_loss = []
        for i,batch in enumerate(bar):
            input_ids_question = batch['input_ids_question'].cuda()
            attention_mask_question = batch['attention_mask_question'].cuda()
            input_ids_misconception = batch['input_ids_missconception'].cuda()
            attention_mask_misconception = batch['attention_mask_misconception'].cuda()
            labels = batch['labels'].cuda()
            labels_misconception = batch['labels_misconception'].cuda()

            with torch.no_grad(), amp.autocast(enabled=True):
                result_question = model(input_ids=input_ids_question, attention_mask=attention_mask_question, answer_input_ids=None, labels=labels, metrics_learning=False, phase='train')
                result_misconception = model(input_ids=input_ids_misconception, attention_mask=attention_mask_misconception, answer_input_ids=None, labels=labels_misconception, phase='train')
                loss_cls = result_question['cls_loss2']
                loss_missconception_cls = result_misconception['cls_loss']
                loss_seq = result_question['seq_loss']
                loss_missconception_seq = result_misconception['seq_loss']
                loss = loss_cls
                valid_loss.append(loss.item())
                valid_question_cls_loss.append(loss_cls.item())
                valid_misconception_cls_loss.append(loss_missconception_cls.item())
                valid_question_seq_loss.append(loss_seq.item())
                valid_misconception_seq_loss.append(loss_missconception_seq.item())

                loss = loss
                bar.set_description(f'smth:{np.mean(valid_loss[-30:]):.4f}')
        return np.mean(valid_loss), np.mean(valid_question_seq_loss), np.mean(valid_misconception_seq_loss), np.mean(valid_question_cls_loss), np.mean(valid_misconception_cls_loss)

    @staticmethod
    def test_func(model, loader_test, loader_test_mapping, embeddings_mapping=None, ids=None):
        bar = tqdm(loader_test)
        model.eval()

        embeddings = []
        question_id_answers = []
        for i,batch in enumerate(bar):
            input_ids_question = batch['input_ids_question'].cuda()
            attention_mask_question = batch['attention_mask_question'].cuda()
            question_id_answers.append(batch['question_id_answer'])
            with torch.no_grad(), amp.autocast(enabled=True):
                result_question = model(input_ids=input_ids_question, attention_mask=attention_mask_question, answer_input_ids=None, labels=None)
                embedding = result_question['embeddings']
                embeddings.append(embedding.cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
        question_id_answers = np.concatenate(question_id_answers, axis=0)

        if embeddings_mapping is None and ids is None:
            embeddings_mapping, ids = EediTrainer.get_misconception_embeddings(model, loader_test_mapping)

        return embeddings, embeddings_mapping, ids, question_id_answers

    @staticmethod
    def test_func_ddp(model, loader_test, loader_test_mapping, rank, embeddings_mapping=None, ids=None):
        bar = tqdm(loader_test)
        model.eval()
        device = f'cuda:{rank}'
        embeddings = []
        question_id_answers = []
        for i,batch in enumerate(bar):
            input_ids_question = batch['input_ids_question'].to(device)
            attention_mask_question = batch['attention_mask_question'].to(device)
            question_id_answers.append(batch['question_id_answer'])
            with torch.no_grad(), amp.autocast(enabled=True):
                result_question = model(input_ids=input_ids_question, attention_mask=attention_mask_question, answer_input_ids=None, labels=None)
                embedding = result_question['embeddings']
                embeddings.append(embedding.cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
        question_id_answers = np.concatenate(question_id_answers, axis=0)

        if embeddings_mapping is None and ids is None:
            embeddings_mapping, ids = EediTrainer.get_misconception_embeddings(model, loader_test_mapping)

        return embeddings, embeddings_mapping, ids, question_id_answers


    @staticmethod
    def get_misconception_embeddings(model, loader_test_mapping, device=None):
        bar = tqdm(loader_test_mapping)
        ids = []
        embeddings_mapping = []
        for i,batch in enumerate(bar):
            if device is None:
                input_ids_mapping = batch['input_ids_mapping'].cuda()
                attention_mask_mapping = batch['attention_mask_mapping'].cuda()
            else:
                input_ids_mapping = batch['input_ids_mapping'].to(device)
                attention_mask_mapping = batch['attention_mask_mapping'].to(device)
            ids.append(batch['misconception_id'])
            with torch.no_grad(), amp.autocast(enabled=True):
                result_mapping = model(input_ids=input_ids_mapping, attention_mask=attention_mask_mapping, answer_input_ids=None, labels=None)
                embedding = result_mapping['embeddings']
                embeddings_mapping.append(embedding.cpu().numpy())
        embeddings_mapping = np.concatenate(embeddings_mapping, axis=0).astype(np.float32)
        ids = np.concatenate(ids, axis=0)

        return embeddings_mapping, ids


    @staticmethod
    def test_rerank_func(model, loader_test):
        bar = tqdm(loader_test)
        model.eval()

        scores = []
        for i,batch in enumerate(bar):
            input_ids_question = batch['input_ids_question'].cuda()
            attention_mask_question = batch['attention_mask_question'].cuda()
            with torch.no_grad(), amp.autocast(enabled=True):
                result_question = model(input_ids=input_ids_question, attention_mask=attention_mask_question, answer_input_ids=None, labels=None)
                score = result_question['logits']
                scores.append(score.cpu().numpy())
        scores = np.concatenate(scores, axis=0)
        bs = scores.shape[0]
        scores = scores.reshape(-1)
        assert len(scores) == bs
        return scores