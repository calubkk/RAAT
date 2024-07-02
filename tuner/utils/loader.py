import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tuner.utils.config import args
def load_data(data_path):
    data = open(data_path,encoding='utf-8')
    data = json.load(data)
    return data

def load_model(model_path):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path,device_map="auto",padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,device_map="auto",use_safetensors=True)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    #model = model.cuda()
    model = model.eval()
    return model,tokenizer