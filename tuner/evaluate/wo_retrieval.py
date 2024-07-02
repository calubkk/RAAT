from tuner.metrics.em_f1 import compute_metrics
from tuner.utils.answer_processor import extract_answer,generate_answer_wo_retrieval,vllm_wo_retrieval
from tuner.utils.loader import load_data,load_model
import json
from tuner.utils.config import args
def evaluate_wo_retrieval(args,data_path,model_path,multi_eval=True,result_save=None,vllm=False):
    data = load_data(data_path)
    f1_list,em_list = [],[]
    if vllm==False:
        model,tokenizer = load_model(model_path)
        for sample in data:
            print(sample['question'])
            texts = generate_answer_wo_retrieval(model,tokenizer,sample['question'],multi_eval=True)
            print(sample['answers'])
            f1_sub_list,em_sub_list = [],[]
            for text in texts:
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
        texts = vllm_wo_retrieval(args,data)
        for i,sample in enumerate(data):
            print(sample['answers'])
            f1_sub_list,em_sub_list = [],[]
            print(texts[i])
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



