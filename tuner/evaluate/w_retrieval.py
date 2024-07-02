from tuner.metrics.em_f1 import compute_metrics
from tuner.utils.answer_processor import extract_answer,generate_answer_w_one_retrieval,generate_answer_w_noisy_retrieval,generate_answer_w_two_retrieval
from tuner.utils.answer_processor import vllm_w_noise_retrieval,vllm_w_one_retrieval,vllm_w_two_retrieval,vllm_wo_retrieval
from tuner.utils.loader import load_data,load_model
from tuner.utils.select_retrieve import select_best_retrieve,select_noisy_retrieve,select_two_retrieve,select_one_retrieve
import json
from tuner.utils.config import args

def evaluate_w_one_retrieval(args,data_path,model_path,save_cache=None,multi_eval=True,result_save=None,retrieve_type='best',vllm=False):
    data = select_one_retrieve(data_path,save_cache=save_cache,type=retrieve_type)
    f1_list,em_list = [],[]
    if vllm==False:
        model,tokenizer = load_model(model_path)
        for sample in data:
            print(sample['question'])
            texts = generate_answer_w_one_retrieval(model,tokenizer,sample['question'],sample['ctx']['text'],multi_eval=multi_eval)
            print(sample['answers'])
            f1_sub_list,em_sub_list = [],[]
            for text in texts:
                print(text)
                text = extract_answer(text)
                print(text)
                if text:
                    output = compute_metrics(text,sample['answers'])
                    f1_sub_list.append(output['f1'])
                    em_sub_list.append(output['em'])
                else:
                    f1_sub_list.append(0.0)
                    em_sub_list.append(0.0)
            sample['f1'] = f1_sub_list[0]
            sample['em'] = em_sub_list[0]
            with open(result_save,'w',encoding='utf-8') as f:
                json.dump(data,f,ensure_ascii=False,indent=2)
            f1_list.append(max(f1_sub_list))
            em_list.append(max(em_sub_list))
    else:
        texts = vllm_w_one_retrieval(args,data)
        for i,sample in enumerate(data):
            #print(sample['answers'])
            f1_sub_list,em_sub_list = [],[]
            text = extract_answer(texts[i])
            #print(text)
            if text:
                output = compute_metrics(text,sample['answers'])
                f1 = output['f1']
                em = output['em']
            else:
                f1 = 0.0
                em = 0.0
            sample['f1'] = f1
            sample['em'] = em
            f1_list.append(f1)
            em_list.append(em)
        with open(result_save,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2)

    print({"f1":sum(f1_list)/len(f1_list), "em": sum(em_list)/len(em_list)})


def evaluate_w_noisy_retrieval(args,data_path,model_path,save_cache=False,noise_rate=0.0,multi_eval=True,vllm=False):
    data = select_noisy_retrieve(data_path,save_cache=save_cache,noise_rate=noise_rate)
    f1_list,em_list = [],[]
    if vllm == False:
        model,tokenizer = load_model(model_path)
        for sample in data:
            #print(sample['question'])
            texts = generate_answer_w_noisy_retrieval(model,tokenizer,sample['question'],sample['ctx'],multi_eval=multi_eval)
            #print(sample['answers'])
            #print(texts)
            f1_sub_list,em_sub_list = [],[]
            for text in texts:
                text = extract_answer(text)
                #print(text)
                if text:
                    output = compute_metrics(text,sample['answers'])
                    f1_sub_list.append(output['f1'])
                    em_sub_list.append(output['em'])
                else:
                    f1_sub_list.append(0.0)
                    em_sub_list.append(0.0)
            f1_list.append(max(f1_sub_list))
            em_list.append(max(em_sub_list))
    else:
        texts = vllm_w_noisy_retrieval(args,data)
        for i,sample in enumerate(data):
            print(sample['answers'])
            f1_sub_list,em_sub_list = [],[]
            text = extract_answer(texts[i])
            print(text)
            if text:
                output = compute_metrics(text,sample['answers'])
                f1 = output['f1']
                em = output['em']
            else:
                f1 = 0.0
                em = 0.0
            sample['f1'] = f1
            sample['em'] = em
            f1_list.append(f1)
            em_list.append(em)
        with open(result_save,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2)

    print({"f1":sum(f1_list)/len(f1_list), "em": sum(em_list)/len(em_list)})

def evaluate_w_two_retrieval(args,data_path,model_path,retrieve_type='complete_irrelevant',save_cache=False,noise_rate=0.5,multi_eval=True,result_save=None,vllm=False):
    data = select_two_retrieve(data_path,save_cache=save_cache,noise_rate=0.5,type=retrieve_type)
    f1_list,em_list = [],[]
    if vllm == False:
        model,tokenizer = load_model(model_path)
        for sample in data:
            print(sample['question'])
            texts = generate_answer_w_two_retrieval(model,tokenizer,sample['question'],sample['ctx'],multi_eval=multi_eval)
            print(sample['answers'])
            print(texts)
            f1_sub_list,em_sub_list = [],[]
            for text in texts:
                text = extract_answer(text)
                #print(text)
                if text:
                    output = compute_metrics(text,sample['answers'])
                    f1_sub_list.append(output['f1'])
                    em_sub_list.append(output['em'])
                else:
                    f1_sub_list.append(0.0)
                    em_sub_list.append(0.0)
            sample['f1'] = f1_sub_list[0]
            sample['em'] = em_sub_list[0]
            with open(result_save,'w',encoding='utf-8') as f:
                json.dump(data,f,ensure_ascii=False,indent=2)
            f1_list.append(max(f1_sub_list))
            em_list.append(max(em_sub_list))
    else:
        texts = vllm_w_two_retrieval(args,data)
        for i,sample in enumerate(data):
            #print(sample['answers'])
            f1_sub_list,em_sub_list = [],[]
            text = extract_answer(texts[i])
            #print(text)
            if text:
                output = compute_metrics(text,sample['answers'])
                f1 = output['f1']
                em = output['em']
            else:
                f1 = 0.0
                em = 0.0
            sample['f1'] = f1
            sample['em'] = em
            f1_list.append(f1)
            em_list.append(em)
        with open(result_save,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
        

    print({"f1":sum(f1_list)/len(f1_list), "em": sum(em_list)/len(em_list)})




def infer_evaluate(data):
    f1_list,em_list = [],[]
    for sample in data:
        text = extract_answer(sample['infer'])
        if text:
            output = compute_metrics(text,sample['answers'])
            f1 = output['f1']
            em = output['em']
        else:
            f1 = 0.0
            em = 0.0
        #print(f1)
        #print(em)
        f1_list.append(f1)
        em_list.append(em)
    ret_f1 = sum(f1_list)/len(f1_list)
    ret_em = sum(em_list)/len(em_list)
    return ret_f1,ret_em



