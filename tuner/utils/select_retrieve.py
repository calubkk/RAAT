import json
from tuner.utils.loader import load_data
import random

def rerank_ctx(ctxs):
    with_answers = [ctx for ctx in ctxs if ctx['has_answer']!=False]
    without_answers = [ctx for ctx in ctxs if ctx['has_answer']==False]
    sorted_ctx = [ctx for ctx in with_answers]
    for ctx in without_answers:
        sorted_ctx.append(ctx)
    return sorted_ctx

def schedule_ctx(ctxs):
    with_answers = [ctx for ctx in ctxs if ctx['has_answer']!=False]
    without_answers = [ctx for ctx in ctxs if ctx['has_answer']==False]

    return with_answers,without_answers

def rerank_two_ctx(ctxs):
    with_answers = [ctx for ctx in ctxs if ctx['has_answer']!=False]
    without_answers = [ctx for ctx in ctxs if ctx['has_answer']==False]
    sorted_ctx=[]
    sorted_ctx.append(with_answers[0])
    sorted_ctx.append(without_answers[0])
    return sorted_ctx

def select_best_retrieve(data_path,save_cache=None):
    '''
    save_cache:是否需要缓存数据。直接放入cache地址。
    '''
    if save_cache==None:
        data = load_data(data_path)
        data_retrieved = []
        if 'nq' in data_path:
            for sample in data:
                #sample['ctxs'] = rerank_ctx(sample['ctxs'])
                data_retrieved.append({
                        'question':sample['question'],
                        'answers':sample['answers'],
                        'ctx':sample['best_ctx'][0]
                    })
                        
        elif 'trivia' in data_path:
            for sample in data:
                #sample['ctxs'] = rerank_ctx(sample['ctxs'])
                data_retrieved.append({
                        'question':sample['question'],
                        'answers':sample['answers'],
                        'ctx':sample['best_ctx'][0]
                    })
        if save_cache is not None:
            with open(save_cache,'w',encoding='utf-8') as f:
                json.dump(data_retrieved,f,ensure_ascii=False,indent=2)
        print(f'The number of data_retrieved is {len(data_retrieved)}.')
    else:
        data_retrieved = load_data(save_cache)
    return data_retrieved

def select_one_retrieve(data_path,save_cache=None,type='best'):
    '''
    save_cache:是否需要缓存数据。直接放入cache地址。
    '''
    if type == 'best':
        return select_best_retrieve(data_path,save_cache)
    elif type == 'complete_irrelevant':
        data = load_data(data_path)
        data_retrieved = []
        if 'nq' in data_path:
            for sample in data:
                temp_sample = random.sample(data,1)[0]
                data_retrieved.append({
                        'question':sample['question'],
                        'answers':sample['answers'],
                        'ctx':temp_sample['ctxs'][0]
                    })
                        
        elif 'trivia' in data_path:
            for sample in data:
                temp_sample = random.sample(data,1)[0]
                data_retrieved.append({
                        'question':sample['question'],
                        'answers':sample['answers'],
                        'ctx':temp_sample['ctxs'][0]
                    })
    elif type == 'partial_relevant':
        data = load_data(data_path)
        data_retrieved = []
        if 'nq' in data_path:
            print('=======================================')
            for sample in data:
                sample['ctxs'] = rerank_two_ctx(sample['ctxs'])
                data_retrieved.append({
                        'question':sample['question'],
                        'answers':sample['answers'],
                        'ctx':sample['ctxs'][1]
                    })
        elif 'trivia' in data_path:
            for sample in data:
                sample['ctxs'] = rerank_two_ctx(sample['ctxs'])
                data_retrieved.append({
                        'question':sample['question'],
                        'answers':sample['answers'],
                        'ctx':sample['ctxs'][1]
                    })

    elif type == 'counterfactual':
        data = load_data(data_path)
        data_retrieved = []
        if 'nq' in data_path:
            for sample in data:
                data_retrieved.append({
                        'question':sample['question'],
                        'answers':sample['answers'],
                        'ctx':{'text':random.sample(sample['counter_fac'],1)[0]}
                    })
        elif 'trivia' in data_path:
            for sample in data:
                data_retrieved.append({
                        'question':sample['question'],
                        'answers':sample['answers'],
                        'ctx':{'text':random.sample(sample['counter_fac'],1)[0]}
                    })
    if save_cache is not None:
        with open(save_cache,'w',encoding='utf-8') as f:
            json.dump(data_retrieved,f,ensure_ascii=False,indent=2)
    print(f'The number of data_retrieved is {len(data_retrieved)}.')
    return data_retrieved

def select_noisy_retrieve(data_path,save_cache=None,noise_rate=0.0,data_list=None):
    if data_list == None:
        data = load_data(data_path)
    else:
        data = data_list
    data_retrieved = []
    num_pos = int(2-2*noise_rate)
    num_neg = int(2*noise_rate)
    if 'nq' in data_path:
        for sample in data:
            candidate_ctx = []
            sample['ctxs'] = rerank_ctx(sample['ctxs'])
            for i in range(num_pos):
                candidate_ctx.append(sample['ctxs'][i])
            for i in range(num_neg):
                candidate_ctx.append(sample['ctxs'][-(i+1)])
            data_retrieved.append({
                        'question':sample['question'],
                        'answers':sample['answers'],
                        'ctx':candidate_ctx
                    })
    elif 'trivia' in data_path:
        for sample in data:
            candidate_ctx = []
            sample['ctxs'] = rerank_ctx(sample['ctxs'])
            for i in range(num_pos):
                candidate_ctx.append(sample['ctxs'][i])
            for i in range(num_neg):
                candidate_ctx.append(sample['ctxs'][-(i+1)])
            data_retrieved.append({
                        'question':sample['question'],
                        'answers':sample['answers'],
                        'ctx':candidate_ctx
                    })
    if save_cache is not None:
        with open(save_cache,'w',encoding='utf-8') as f:
            json.dump(data_retrieved,f,ensure_ascii=False,indent=2)
    print(f'The number of data_retrieved is {len(data_retrieved)}.')
    return data_retrieved


def select_two_retrieve(data_path,save_cache=None,noise_rate=0.5,data_list=None,type='complete_irrelevant'):
    if save_cache==None:
        if data_list == None:
            data = load_data(data_path)
        else:
            data = data_list
        data_retrieved = []
        num_pos = 1
        num_neg = 1
        if 'nq' in data_path:
            for sample in data:
                candidate_ctx = []
                sample['ctxs'] = rerank_two_ctx(sample['ctxs'])
                for i in range(num_pos):
                    candidate_ctx.append(sample['best_ctx'][i])
                for i in range(num_neg):
                    if type=='complete_irrelevant':
                        temp_sample = random.sample(data,1)[0]
                        #print(temp_sample)
                        candidate_ctx.append(temp_sample['ctxs'][-(i+1)])
                    elif type=='partial_relevant':
                        candidate_ctx.append(sample['ctxs'][-(i+1)])
                    elif type=='counterfactual':
                        candidate_ctx.append({'text':random.sample(sample['counter_fac'],1)[0]})
                data_retrieved.append({
                            'question':sample['question'],
                            'answers':sample['answers'],
                            'ctx':candidate_ctx
                        })
        elif 'trivia' in data_path:
            for sample in data:
                candidate_ctx = []
                sample['ctxs'] = rerank_two_ctx(sample['ctxs'])
                for i in range(num_pos):
                    candidate_ctx.append(sample['best_ctx'][i])
                for i in range(num_neg):
                    if type=='complete_irrelevant':
                        temp_sample = random.sample(data,1)[0]
                        #print(temp_sample)
                        candidate_ctx.append(temp_sample['ctxs'][-(i+1)])
                    elif type=='partial_relevant':
                        candidate_ctx.append(sample['ctxs'][-(i+1)])
                    elif type=='counterfactual':
                        candidate_ctx.append({'text':random.sample(sample['counter_fac'],1)[0]})
                data_retrieved.append({
                            'question':sample['question'],
                            'answers':sample['answers'],
                            'ctx':candidate_ctx
                        })
        if save_cache is not None:
            with open(save_cache,'w',encoding='utf-8') as f:
                json.dump(data_retrieved,f,ensure_ascii=False,indent=2)
        print(f'The number of data_retrieved is {len(data_retrieved)}.')
    else:
        data_retrieved = load_data(save_cache)
    
    return data_retrieved


