from dataclasses import dataclass
import math
import random
import numpy as np
import torch
import json
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    GPT2Tokenizer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from datasets import Dataset
from tuner.utils.config import args
from tuner.utils.loader import load_model,load_data
from datasets import DatasetDict
from tuner.utils.select_retrieve import schedule_ctx 

class DataManager():
    def __init__(self, config, training_stage, tokenizer_path):
        self.config = config
        if self.config.architectures[0].lower() == "llamaforcausallm":
            self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
            self.tokenizer.unk_token = "<unk>"
            self.tokenizer.bos_token = "<s>"
            self.tokenizer.eos_token = "</s>"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.padding = True
        self.max_length = args.max_length
        self.pad_to_multiple_of = 8
        self.return_tensors = "pt"
        self.add_special_tokens = True
        self.training_stage = training_stage

    
    def batch_decode(self, model_output):
        # model_output = [batch, seq_len]
        return self.tokenizer.batch_decode(model_output, skip_special_tokens=True)

    
    def train_data_collator_adapt(self, features):
        samples_num = len(features)
        training_stage = self.training_stage
        origin_state = (self.tokenizer.padding_side, self.tokenizer.truncation_side)

        self.tokenizer.truncation_side = "left"
        ps = []
        ss = []
        sft_index = []
        for feature_index, feature in enumerate(features):
            ps.append(feature['o'][0])
            ps.append(feature['p'][0])
            ps.append(feature['f'][0])
            ps.append(feature['c'][0])
            ss.append(feature['answers'][0])
            ss.append(feature['answers'][0])
            ss.append(feature['answers'][0])
            ss.append(feature['answers'][0])
        sft_index.append([0])

        ps = self.batch_decode(
            self.tokenizer(
                ps,
                max_length = self.max_length - 128,
                truncation = True,
                add_special_tokens = self.add_special_tokens,
            )['input_ids']
        )

        ps_input_ids = self.tokenizer(
            ps,
            add_special_tokens = self.add_special_tokens,
        )['input_ids']
        ps_lens = [len(p_input_ids)-1 for p_input_ids in ps_input_ids]
        
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        
        texts = []
        for p, s in zip(ps, ss):
            texts.append(p + " " + s)
        
        batch = self.tokenizer(
            texts,
            padding=self.padding,
            max_length = self.max_length,
            truncation = True,
            add_special_tokens = self.add_special_tokens,
            return_tensors = self.return_tensors,
        )
        
        seq_len = batch["attention_mask"].shape[1]
        prefix_mask = []
        for p_len in ps_lens:
            assert seq_len > p_len
            prefix_mask.append(
                [1 if i<p_len else 0 for i in range(seq_len)]
            )
        batch["prefix_mask"] = torch.tensor(prefix_mask)
        
        batch['labels'] = batch["input_ids"].clone().detach()
        for key in batch:
            #print(batch[key].size())
            batch[key] = batch[key].view(samples_num,training_stage,-1)
        
        batch['sft_index'] = torch.tensor(sft_index) # [batch]
        # restore states
        self.tokenizer.padding_side, self.tokenizer.truncation_side = origin_state

        return batch
    
    def load_train_data_adaptad(
        self, 
        data_collator, 
        data_file_path, 
        extension='json', 
        stream = None, 
    ):
        raw_data = open(data_file_path,encoding='utf-8')
        raw_data = json.load(raw_data)
        ret = []
        prompt_w = '<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[INST] {ctx}\n{query} [/INST]'
        prompt_wo = '<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[INST] {query} [/INST]'
        sys = 'You need to complete the question and answer pairs according to the format in the example.The answer to be completed should be a short phrase entity,not sentences.Here are some examples to guide you.'
        examples = [
        '\nExample 1:\nQuestion: What is the capital of France?\nAnswer: Paris.',
        '\nExample 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
        '\nExample 3:\nQuestion: What is the largest planet in our solar system?\nAnswer: Jupiter.'
        ]
        train_data_cache = args.train_data_cache
        for i,sample in enumerate(raw_data):
            #sample["ad_ctx"].append(sample['best_ctx']['text'])
            ad_ctxs = sample["ad_ctx"]
            best_ctx = sample['best_ctx']
            #known_or_unknown = sample['flag']
            #neg_samples = sample['neg']
            #neg_pos_ctx,neg_neg_ctx = schedule_ctx(sample['neg'][0]['ctxs'])
            sys_ctx_one_p = 'The following contexts will help you complete the question and answer pair.\nContext1:{ctx_0}'.format(ctx_0=ad_ctxs[0])
            sys_ctx_one_f = 'The following contexts will help you complete the question and answer pair.\nContext1:{ctx_0}'.format(ctx_0=ad_ctxs[1])
            sys_ctx_one_c = 'The following contexts will help you complete the question and answer pair.\nContext1:{ctx_0}'.format(ctx_0=ad_ctxs[2])
            if i<len(raw_data)/2:
                sys_ctx_two_p = 'The following contexts will help you complete the question and answer pair.\nContext1:{ctx_0}\nContext2:{ctx_1}'.format(ctx_0=best_ctx['text'],ctx_1=ad_ctxs[0])
                sys_ctx_two_f = 'The following contexts will help you complete the question and answer pair.\nContext1:{ctx_0}\nContext2:{ctx_1}'.format(ctx_0=best_ctx['text'],ctx_1=ad_ctxs[1])
                sys_ctx_two_c = 'The following contexts will help you complete the question and answer pair.\nContext1:{ctx_0}\nContext2:{ctx_1}'.format(ctx_0=best_ctx['text'],ctx_1=ad_ctxs[2])
            else:
                sys_ctx_two_p = 'The following contexts will help you complete the question and answer pair.\nContext1:{ctx_0}\nContext2:{ctx_1}'.format(ctx_1=best_ctx['text'],ctx_0=ad_ctxs[0])
                sys_ctx_two_f = 'The following contexts will help you complete the question and answer pair.\nContext1:{ctx_0}\nContext2:{ctx_1}'.format(ctx_1=best_ctx['text'],ctx_0=ad_ctxs[1])
                sys_ctx_two_c = 'The following contexts will help you complete the question and answer pair.\nContext1:{ctx_0}\nContext2:{ctx_1}'.format(ctx_1=best_ctx['text'],ctx_0=ad_ctxs[2])
            sys_ctx_best = 'The following contexts will help you complete the question and answer pair.\nContext1:{ctx_0}'.format(ctx_0=best_ctx['text'])
            #sys_ctx_best_neg = 'The following contexts will help you complete the question and answer pair.\nContext1:{ctx_0}'.format(ctx_0=neg_pos_ctx[0]['text'])
            #p_sys_ctx_two_neg = 'The following contexts will help you complete the question and answer pair.\nContext1:{ctx_0}\nContext2:{ctx_1}'.format(ctx_0=neg_pos_ctx[0]['text'],ctx_1=neg_neg_ctx[0]['text'])
            #f_sys_ctx_two_neg = 'The following contexts will help you complete the question and answer pair.\nContext1:{ctx_0}\nContext2:{ctx_1}'.format(ctx_0=neg_pos_ctx[0]['text'],ctx_1=neg_samples[0]['counter_fac'][0])
            #c_sys_ctx_two_neg = 'The following contexts will help you complete the question and answer pair.\nContext1:{ctx_0}\nContext2:{ctx_1}'.format(ctx_0=neg_pos_ctx[0]['text'],ctx_1=random.sample(raw_data,1)[0]['best_ctx']['text'])
            o_sentence = prompt_wo.format(system_prompt=sys+examples[0]+examples[1]+examples[2],query='Question:'+sample['question'])
            p_one_sentence = prompt_w.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctx_one_p,query='Question:'+sample['question'])
            f_one_sentence = prompt_w.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctx_one_f,query='Question:'+sample['question'])
            c_one_sentence = prompt_w.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctx_one_c,query='Question:'+sample['question'])
            p_two_sentence = prompt_w.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctx_two_p,query='Question:'+sample['question'])
            f_two_sentence = prompt_w.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctx_two_f,query='Question:'+sample['question'])
            c_two_sentence = prompt_w.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctx_two_c,query='Question:'+sample['question'])
            best_sentence = prompt_w.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctx_best,query='Question:'+sample['question'])
            #neg_one_best_sentence = prompt_w.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctx_best_neg,query='Question:'+neg_samples[0]['question'])
            #p_neg_two_sentence = prompt_w.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=p_sys_ctx_two_neg,query='Question:'+neg_samples[0]['question'])
            #f_neg_two_sentence = prompt_w.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=f_sys_ctx_two_neg,query='Question:'+neg_samples[0]['question'])
            #c_neg_two_sentence = prompt_w.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=c_sys_ctx_two_neg,query='Question:'+neg_samples[0]['question'])
            
            
            o_cache = [best_sentence]
            p_cache = [p_two_sentence]
            f_cache = [f_two_sentence]
            c_cache = [c_two_sentence]
            weights_1 = [1.0]
            weights_2 = [1.0]
            ret.append({'o':[random.choices(o_cache,weights=weights_1,k=1)[0]],
            'p':[random.choices(p_cache,weights=weights_2,k=1)[0]],
            'f':[random.choices(f_cache,weights=weights_2,k=1)[0]],
            'c':[random.choices(c_cache,weights=weights_2,k=1)[0]],
            'answers':['\nAnswer: '+best_ctx['has_answer']]})
                #'\nAnswer: '+neg_samples[0]['answers'][0]]})

        with open(train_data_cache,'w',encoding='utf-8') as f:
            json.dump(ret ,f,ensure_ascii=False,indent=2)
        raw_datasets = load_dataset('json', data_files=train_data_cache)
        dataloader = DataLoader(
            raw_datasets['train'], 
            shuffle=True,
            collate_fn=data_collator, 
            batch_size=args.per_device_train_batch_size
        )

        return dataloader
    
    def infer_generate(self, model, prefixes):
        # prefixes = [prefix, prefix]
        origin_state = (self.tokenizer.padding_side, self.tokenizer.truncation_side)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        
        new_prefixes = []
        for p in prefixes:
            new_prefixes.append(p)
        prefixes = new_prefixes

        prefixes = self.batch_decode(
            self.tokenizer(
                new_prefixes,
                max_length = self.max_length - 128,
                truncation = True,
                add_special_tokens = self.add_special_tokens,
            )["input_ids"]
        )
        #print(prefixes)

        batch = self.tokenizer(
            prefixes,
            padding=self.padding,
            max_length = self.max_length - 128,
            truncation = True,
            add_special_tokens = self.add_special_tokens,
            return_tensors = self.return_tensors,
        ).to(model.device)
        batch_size = len(prefixes)
        truncated_prefixes = self.batch_decode(batch['input_ids'])
        
        with torch.no_grad():
            predicted_sents = model.generate(
                **batch, 
                max_new_tokens = 128,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=1,
                do_sample=False,
                top_k=50,
                top_p=0.6,
                temperature=0.3,
                repetition_penalty=1.0,
                num_return_sequences = 1,
            )
        
        instant_text = self.batch_decode(predicted_sents)
        print('===================')
        print(instant_text)
        
        # restore states
        self.tokenizer.padding_side, self.tokenizer.truncation_side = origin_state
        
        for index in range(len(instant_text)):
            #print('===================')
            #print(instant_text[index])
            assert truncated_prefixes[index].rstrip() in instant_text[index], (truncated_prefixes[index].strip(), instant_text[index])
            instant_text[index] = instant_text[index].replace(truncated_prefixes[index].rstrip(), "").strip()
            instant_text[index] = instant_text[index].strip()        
            #print('++++++++++++++++++++++++++++++++++++++++++++++++++++++==')
            #print(instant_text[index])  
        return instant_text
