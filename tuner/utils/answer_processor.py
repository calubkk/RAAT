import re
from typing import Tuple, List, Dict
import ftfy
from tuner.utils.config import args
from vllm import LLM, SamplingParams

def extract_answer(text):
    pattern =  r"Answer:(.*)"
    # 使用 re.search 查找匹配项
    match = re.search(pattern, text)
    if match:
        extracted_text = match.group(1)  # 提取匹配的子字符串
        #print("提取的内容：", extracted_text)
        if extracted_text!='' and extracted_text[-1] == '.':
            extracted_text = extracted_text[:-1]
        return extracted_text
    else:
        #print("没有找到匹配项")
        return False

def vllm_wo_retrieval(args,data):
    if 'Llama' in args.test_model_name_or_path or 'llama' in args.test_model_name_or_path:
        prompt='<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[INST] {query} [/INST]'
    elif 'Mistral' in args.test_model_name_or_path:
        prompt='<s>[INST] {system_prompt}\n{query} [/INST]'
    elif 'chatglm' in args.test_model_name_or_path:
        prompt='{system_prompt}\n{query}'
    elif 'Qwen' in args.test_model_name_or_path:
        prompt='<|im_start|>user\n{system_prompt}\n{query}<|im_end|>\n<|im_start|>assistant\n'
    sys = 'You need to complete the question-and-answer pair following the format provided in the example. The answers should be short phrases or entities, not full sentences. Here are some examples to guide you.'
    examples = [
        '\nExample 1:\nQuestion: What is the capital of France?\nAnswer: Paris.',
        '\nExample 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
        '\nExample 3:\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
    ]

    llm = LLM(model=args.test_model_name_or_path,tensor_parallel_size=1,  tokenizer_mode='auto',
                trust_remote_code=True)
    print("加载完成")

    sampling_param = SamplingParams(max_tokens=512,top_k=50,
                top_p=0.6,
                temperature=0.3,
                repetition_penalty=1.0,
                stop_token_ids=[7])
    sens = []
    for sample in data:
        sentence = prompt.format(system_prompt=sys+examples[0]+examples[1]+examples[2],query='Question:'+sample['question'])
        sens.append(sentence)

    outputs = llm.generate(sens, sampling_params=sampling_param)
    ret = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        ret.append(generated_text)

    return ret

def generate_answer_wo_retrieval(model,tokenizer,query,multi_eval=True,wo_ict=False):
    '''
    multi_eval:是否需要多次利用不同例子进行推理。
    '''
    if wo_ict==True:
        input_ids = tokenizer(query,return_tensors="pt").input_ids.to('cuda')    
        #print(input_ids)
        generate_input = {
        "input_ids":input_ids,
        "max_new_tokens":512,
        "do_sample":True,
        "top_k":50,
        "top_p":0.6,
        "temperature":0.3,
        "repetition_penalty":1.0,
        "eos_token_id":tokenizer.eos_token_id,
        "bos_token_id":tokenizer.bos_token_id,
        "pad_token_id":tokenizer.pad_token_id
        }    
        generate_ids  = model.generate(**generate_input)
        #print(input_ids.size())
        text = tokenizer.decode(generate_ids[0][input_ids.size(1):-1])
        texts.append(text)
    else:
        prompt='<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[INST] {query} [/INST]'
        sys = 'You need to complete the question-and-answer pair following the format provided in the example. The answers should be short phrases or entities, not full sentences. Here are some examples to guide you.'
        examples = [
            '\nExample 1:\nQuestion: What is the capital of France?\nAnswer: Paris.',
            '\nExample 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
            '\nExample 3:\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
        ]
        texts = []
        if multi_eval:
            #for example in examples:
            sentence = prompt.format(system_prompt=sys+examples[0]+examples[1]+examples[2],query='Question:'+query)
            input_ids = tokenizer(sentence,return_tensors="pt").input_ids.to('cuda')    
            #print(input_ids)
            generate_input = {
            "input_ids":input_ids,
            "max_new_tokens":512,
            "do_sample":True,
            "top_k":50,
            "top_p":0.6,
            "temperature":0.3,
            "repetition_penalty":1.0,
            "eos_token_id":tokenizer.eos_token_id,
            "bos_token_id":tokenizer.bos_token_id,
            "pad_token_id":tokenizer.pad_token_id
            }    
            generate_ids  = model.generate(**generate_input)
            #print(input_ids.size())
            text = tokenizer.decode(generate_ids[0][input_ids.size(1):-1])
            texts.append(text)
            #print(text)
        else:
            sentence = prompt.format(system_prompt=sys+examples[0],query='Question:'+query)
            input_ids = tokenizer(sentence,return_tensors="pt").input_ids.to('cuda')    
            #print(input_ids)
            generate_input = {
            "input_ids":input_ids,
            "max_new_tokens":512,
            "do_sample":True,
            "top_k":50,
            "top_p":0.6,
            "temperature":0.3,
            "repetition_penalty":1.0,
            "eos_token_id":tokenizer.eos_token_id,
            "bos_token_id":tokenizer.bos_token_id,
            "pad_token_id":tokenizer.pad_token_id
            }    
            generate_ids  = model.generate(**generate_input)
            #print(input_ids.size())
            text = tokenizer.decode(generate_ids[0][input_ids.size(1):-1])
            texts.append(text)
            #print(text)

    return texts

def vllm_w_one_retrieval(args,data):
    print('===============')
    print(len(data))
    if 'Llama' in args.test_model_name_or_path or 'llama' in args.test_model_name_or_path:
        prompt='<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[INST] {ctx}\n{query} [/INST]'
    elif 'Mistral' in args.test_model_name_or_path:
        prompt='<s>[INST] {system_prompt}\n{ctx}\n{query} [/INST]'
    elif 'chatglm' in args.test_model_name_or_path:
        prompt='[Round 1]\n\n问：{system_prompt}\n{ctx}\n{query} \n\n答：'
    elif 'Qwen' in args.test_model_name_or_path:
        prompt='{system_prompt}\n{ctx}\n{query}'
    sys = 'You need to complete the question-and-answer pair following the format provided in the example. The answers should be short phrases or entities, not full sentences. Here are some examples to guide you.'
    examples = [
        '\nExample 1:\nQuestion: What is the capital of France?\nAnswer: Paris.',
        '\nExample 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
        '\nExample 3:\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
    ]
    sys_ctx = 'The following context will help you complete the question-and-answer pair.\nContext:'

    llm = LLM(model=args.test_model_name_or_path,tensor_parallel_size=1,  tokenizer_mode='auto',
                trust_remote_code=True)
    print("加载完成")

    sampling_param = SamplingParams(max_tokens=512,top_k=50,
                top_p=0.6,
                temperature=0.3,
                repetition_penalty=1.0,
                stop_token_ids=[7])
    sens = []
    for sample in data:
        sentence = prompt.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctx+sample['ctx']['text'],query='Question:'+sample['question'])
        sens.append(sentence)

    outputs = llm.generate(sens, sampling_params=sampling_param)
    ret = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        ret.append(generated_text)

    return ret



def generate_answer_w_one_retrieval(model,tokenizer,query,ctx,multi_eval=True):
    prompt='<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[INST] {ctx}\n{query} [/INST]'
    sys = 'You need to complete the question-and-answer pair following the format provided in the example. The answers should be short phrases or entities, not full sentences. Here are some examples to guide you.'
    examples = [
        '\nExample 1:\nQuestion: What is the capital of France?\nAnswer: Paris.',
        '\nExample 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
        '\nExample 3:\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
    ]
    texts = []
    sys_ctx = 'The following context will help you complete the question-and-answer pair.\nContext:'
    if multi_eval:
        sentence = prompt.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctx+ctx,query='Question:'+query)
        input_ids = tokenizer(sentence,return_tensors="pt").input_ids.to('cuda')    
        #print(input_ids)
        generate_input = {
        "input_ids":input_ids,
        "max_new_tokens":512,
        "do_sample":True,
        "top_k":50,
        "top_p":0.6,
        "temperature":0.3,
        "repetition_penalty":1.0,
        "eos_token_id":tokenizer.eos_token_id,
        "bos_token_id":tokenizer.bos_token_id,
        "pad_token_id":tokenizer.pad_token_id
        }    
        generate_ids  = model.generate(**generate_input)
        #print(input_ids.size())
        text = tokenizer.decode(generate_ids[0][input_ids.size(1):-1])
        texts.append(text)
            #print(text)
    else:
        sentence = prompt.format(system_prompt=sys+examples[0],ctx=sys_ctx+ctx,query='Question:'+query)
        input_ids = tokenizer(sentence,return_tensors="pt").input_ids.to('cuda')    
        #print(input_ids)
        generate_input = {
        "input_ids":input_ids,
        "max_new_tokens":512,
        "do_sample":True,
        "top_k":50,
        "top_p":0.6,
        "temperature":0.3,
        "repetition_penalty":1.0,
        "eos_token_id":tokenizer.eos_token_id,
        "bos_token_id":tokenizer.bos_token_id,
        "pad_token_id":tokenizer.pad_token_id
        }    
        generate_ids  = model.generate(**generate_input)
        #print(input_ids.size())
        text = tokenizer.decode(generate_ids[0][input_ids.size(1):-1])
        texts.append(text)
        #print(text)

    return texts


def vllm_w_noise_retrieval(args,data):
    if 'Llama' in args.test_model_name_or_path or 'llama' in args.test_model_name_or_path:
        prompt='<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[INST] {ctx}\n{query} [/INST]'
    elif 'Mistral' in args.test_model_name_or_path:
        prompt='<s>[INST] {system_prompt}\n{ctx}\n{query} [/INST]'
    elif 'chatglm' in args.test_model_name_or_path:
        prompt='[Round 1]\n\n问：{system_prompt}\n{ctx}\n{query} \n\n答：'
    elif 'Qwen' in args.test_model_name_or_path:
        prompt='{system_prompt}\n{ctx}\n{query}'
    sys = 'You need to complete the question-and-answer pair following the format provided in the example. The answers should be short phrases or entities, not full sentences. Here are some examples to guide you.'
    examples = [
        '\nExample 1:\nQuestion: What is the capital of France?\nAnswer: Paris.',
        '\nExample 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
        '\nExample 3:\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
    ]
    
    llm = LLM(model=args.test_model_name_or_path,tensor_parallel_size=1,  tokenizer_mode='auto',
                trust_remote_code=True)
    print("加载完成")

    sampling_param = SamplingParams(max_tokens=2048,top_k=50,
                top_p=0.6,
                temperature=0.3,
                repetition_penalty=1.0,
                stop_token_ids=[7])
    sens = []
    for sample in data:
        ctx_list = sample['ctx']
        sys_ctx = 'The following contexts will help you complete the question-and-answer pair.\nContext1:{ctx_0}\nContext2:{ctx_1}\nContext3:{ctx_2}\nContext4:{ctx_3}'.format(ctx_0=ctx_list[0]['text'],ctx_1=ctx_list[1]['text'],ctx_2=ctx_list[2]['text'],ctx_3=ctx_list[3]['text'])
        sentence = prompt.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctx,query='Question:'+sample['question'])
        sens.append(sentence)

    outputs = llm.generate(sens, sampling_params=sampling_param)
    ret = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        ret.append(generated_text)

    return ret

def generate_answer_w_noisy_retrieval(model,tokenizer,query,ctx_list,multi_eval=True):
    prompt='<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[INST] {ctx}\n{query} [/INST]'
    sys = 'You need to complete the question-and-answer pair following the format provided in the example. The answers should be short phrases or entities, not full sentences. Here are some examples to guide you.'
    examples = [
        '\nExample 1:\nQuestion: What is the capital of France?\nAnswer: Paris.',
        '\nExample 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
        '\nExample 3:\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
    ]
    texts = []
    sys_ctx = 'The following contexts will help you complete the question-and-answer pair.\nContext1:{ctx_0}\nContext2:{ctx_1}\nContext3:{ctx_2}\nContext4:{ctx_3}'.format(ctx_0=ctx_list[0]['text'],ctx_1=ctx_list[1]['text'],ctx_2=ctx_list[2]['text'],ctx_3=ctx_list[3]['text'])
    if multi_eval:
        sentence = prompt.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctx,query='Question:'+query)
        input_ids = tokenizer(sentence,return_tensors="pt").input_ids.to('cuda')    
        #print(input_ids)
        generate_input = {
        "input_ids":input_ids,
        "max_new_tokens":512,
        "do_sample":True,
        "top_k":50,
        "top_p":0.6,
        "temperature":0.3,
        "repetition_penalty":1.0,
        "eos_token_id":tokenizer.eos_token_id,
        "bos_token_id":tokenizer.bos_token_id,
        "pad_token_id":tokenizer.pad_token_id
        }    
        generate_ids  = model.generate(**generate_input)
        #print(generate_ids)
        text = tokenizer.decode(generate_ids[0][input_ids.size(1):-1])
        texts.append(text)
        #print(text)
    else:
        sentence = prompt.format(system_prompt=sys+examples[0],ctx=sys_ctx+ctx,query='Question:'+query)
        input_ids = tokenizer(sentence,return_tensors="pt").input_ids.to('cuda')    
        #print(input_ids)
        generate_input = {
        "input_ids":input_ids,
        "max_new_tokens":512,
        "do_sample":True,
        "top_k":50,
        "top_p":0.6,
        "temperature":0.3,
        "repetition_penalty":1.0,
        "eos_token_id":tokenizer.eos_token_id,
        "bos_token_id":tokenizer.bos_token_id,
        "pad_token_id":tokenizer.pad_token_id
        }    
        generate_ids  = model.generate(**generate_input)
        #print(generate_ids)
        text = tokenizer.decode(generate_ids[0][input_ids.size(1):-1])
        texts.append(text)
        #print(text)

    return texts

def vllm_w_two_retrieval(args,data):
    if 'Llama' in args.test_model_name_or_path or 'llama' in args.test_model_name_or_path:
        prompt='<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[INST] {ctx}\n{query} [/INST]'
    elif 'Mistral' in args.test_model_name_or_path:
        prompt='<s>[INST] {system_prompt}\n{ctx}\n{query} [/INST]'
    elif 'chatglm' in args.test_model_name_or_path:
        prompt='[Round 1]\n\n问：{system_prompt}\n{ctx}\n{query} \n\n答：'
    elif 'Qwen' in args.test_model_name_or_path:
        prompt='{system_prompt}\n{ctx}\n{query}'
    sys = 'You need to complete the question-and-answer pair following the format provided in the example. The answers should be short phrases or entities, not full sentences. Here are some examples to guide you.'
    examples = [
        '\nExample 1:\nQuestion: What is the capital of France?\nAnswer: Paris.',
        '\nExample 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
        '\nExample 3:\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
    ]
    
    llm = LLM(model=args.test_model_name_or_path,tensor_parallel_size=1,  tokenizer_mode='auto',
                trust_remote_code=True)
    print("加载完成")

    sampling_param = SamplingParams(max_tokens=2048,top_k=50,
                top_p=0.6,
                temperature=0.3,
                repetition_penalty=1.0,
                stop_token_ids=[7])
    sens = []
    half_length = int(len(data)/2)
    for i,sample in enumerate(data):
        ctx_list = sample['ctx']
        if i<half_length:
            sys_ctx = 'The following contexts will help you complete the question-and-answer pair.\nContext1:{ctx_0}\nContext2:{ctx_1}'.format(ctx_0=ctx_list[0]['text'],ctx_1=ctx_list[1]['text'])
        else:
            sys_ctx = 'The following contexts will help you complete the question-and-answer pair.\nContext1:{ctx_0}\nContext2:{ctx_1}'.format(ctx_1=ctx_list[0]['text'],ctx_0=ctx_list[1]['text'])
        sentence = prompt.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctx,query='Question:'+sample['question'])
        sens.append(sentence)

    outputs = llm.generate(sens, sampling_params=sampling_param)
    ret = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        ret.append(generated_text)

    return ret


def generate_answer_w_two_retrieval(model,tokenizer,query,ctx_list,multi_eval=True):
    prompt='<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[INST] {ctx}\n{query} [/INST]'
    sys = 'You need to complete the question-and-answer pair following the format provided in the example. The answers should be short phrases or entities, not full sentences. Here are some examples to guide you.'
    examples = [
        '\nExample 1:\nQuestion: What is the capital of France?\nAnswer: Paris.',
        '\nExample 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
        '\nExample 3:\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
    ]
    texts = []
    sys_ctx = 'The following contexts will help you complete the question-and-answer pair.\nContext1:{ctx_0}\nContext2:{ctx_1}'.format(ctx_0=ctx_list[0]['text'],ctx_1=ctx_list[1]['text'])
    sentence = prompt.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctx,query='Question:'+query)
    input_ids = tokenizer(sentence,return_tensors="pt").input_ids.to('cuda')    
    #print(input_ids)
    generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.6,
    "temperature":0.3,
    "repetition_penalty":1.0,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
    }    
    generate_ids  = model.generate(**generate_input)
    #print(generate_ids)
    text = tokenizer.decode(generate_ids[0][input_ids.size(1):-1])
    texts.append(text)
        #print(text)
        #print(text)

    return texts
