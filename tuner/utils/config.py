import random
import numpy as np
import torch
import argparse
from transformers import SchedulerType

def parse_args():
    parser = argparse.ArgumentParser(description="...")

    #print('==========================================')
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--test_model_name_or_path",
        type=str,
        help='model for testing'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help='model for training e.g. llama2-7b'
    )
    #print('===================test setting=======================')
    parser.add_argument(
        "--multi_eval",
        action="store_true",
    )
    parser.add_argument(
        "--selected_retrieve_cache",
        type=str, 
        default=None,
        help='whether the retrieved data needs to be cached',
    )
    parser.add_argument(
        "--result_save",
        type=str, 
        default=None,
    )
    parser.add_argument(
        "--retrieve_type",
        type=str, 
        default='best',
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--reg_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--adv_rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
    )
    parser.add_argument(
        "--test_data_path",
        type=str, 
        default=None,
        help='test data',
    )
    parser.add_argument(
        "--wo_retrieval",
        action="store_true",
    )
    parser.add_argument(
        "--w_one_retrieval",
        action="store_true",
    )
    parser.add_argument(
        "--w_two_retrieval",
        action="store_true",
    )
    parser.add_argument(
        "--complete_irrelevant",
        action="store_true",
    )



    #print('===================training setting=======================')
    parser.add_argument(
        "--do_train",
        action="store_true",
    )
    parser.add_argument(
        "--do_validation",
        action="store_true",
    )
    parser.add_argument(
        "--train_data_cache",
        type=str, 
        default='/home/fangfeiteng/robust/tuner/data/temp.json',
        help='store training data cache',
    )
    parser.add_argument(
        "--choice_cache",
        type=str, 
        default='/home/fangfeiteng/ACL_RAAT/tuner/data/record.json',
        help='record each selection',
    )
    parser.add_argument(
        "--sft_weight",
        type=float,
        default=2,
    )
    parser.add_argument(
        "--train_file_path", type=str, default=None,
    )
    parser.add_argument(
        "--validation_file_path", type=str, default=None,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--checkpointing_step",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--training_stage_num",
        type=int,
        default=4,
    )
    
    
    args = parser.parse_args()

    return args




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

args = parse_args()
setup_seed(args.seed)