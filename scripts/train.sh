
export NCCL_P2P_LEVEL="NVL"
export OMP_NUM_THREADS=16
root_dir=..
mkdir -p $root_dir/logs/raat
mkdir -p $root_dir/checkpoints/raat
train_path=$root_dir/tuner/data/retrieval_robustness_benchmark/train.json
val_path=$root_dir/tuner/data/retrieval_robustness_benchmark/dev.json
model_name_or_path=  #where u put ur LLMs
train_data_cache=$root_dir/tuner/data/temp.json
choice_cache=$root_dir/tuner/data/record.json
accelerate config
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch  ../train.py \
    --train_file_path $train_path \
    --validation_file_path  $val_path \
    --output_dir $root_dir/checkpoints/raat \
    --log_path  $root_dir/logs/raat \
    --model_name_or_path $model_name_or_path\
    --train_data_cache $train_data_cache\
    --choice_cache $choice_cache\
    --seed 42 \
    --sft_weight 2 \
    --reg_weight 0.2 \
    --num_train_epochs 2 \
    --training_stage_num 4 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --do_train  \
    --do_validation  
   # --do_validation > $root_dir/logs/$id/$ranking_len/train_detail.log 2>&1



