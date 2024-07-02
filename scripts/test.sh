model= #where u store ur checkpoints
model_type=raat-acl
dataset=total
root_dir=..
test_data=$root_path/tuner/data/retrieval_robustness_benchmark/test.json   #benchmark
mkdir -p $root_path/test_cache/$dataset  
mkdir -p $root_path/test_cache/$dataset/$model_type   #calculation results
mkdir -p $root_path/benchmark_cache #test file cache
#/home/fangfeiteng/LLMs/temp-llama
#/home/fangfeiteng/robust/tuner/data/trivia_test/trivia-hybrid-test-counter-factual.json
#/home/zhanglei/zoo/llama-2-70b-chat
#/home/fangfeiteng/LLaMA-Factory-main/outputs/adv-robust-llama-025

#Golden retrieval
CUDA_VISIBLE_DEVICES=0 python3 ../baseline.py \
    --w_one_retrieval \
    --retrieve_type 'best' \
    --test_model_name_or_path $model \
    --result_save $root_path/test_cache/$dataset/$model_type/o.json \
    --selected_retrieve_cache $root_path/benchmark_cache/o.json \
    --test_data_path $test_data

#Golden retrieval + irrelevant retrieval noise
CUDA_VISIBLE_DEVICES=0  python3 ../baseline.py \
    --w_two_retrieval \
    --retrieve_type 'complete_irrelevant' \
    --test_model_name_or_path $model \
    --result_save $root_path/test_cache/$dataset/$model_type/c.json \
    --selected_retrieve_cache $root_path/benchmark_cache/c.json \
    --test_data_path $test_data

    
#Golden retrieval + relevant retrieval noise
CUDA_VISIBLE_DEVICES=0  python3 ../baseline.py \
    --w_two_retrieval \
    --retrieve_type 'partial_relevant' \
    --test_model_name_or_path $model \
    --result_save $root_path/test_cache/$dataset/$model_type/p.json \
    --selected_retrieve_cache $root_path/benchmark_cache/p.json \
    --test_data_path $test_data

#Golden retrieval + counterfactual retrieval noise
CUDA_VISIBLE_DEVICES=0  python3 ../baseline.py \
    --w_two_retrieval \
    --retrieve_type 'counterfactual' \
    --test_model_name_or_path $model \
    --result_save $root_path/test_cache/$dataset/$model_type/f.json \
    --selected_retrieve_cache $root_path/benchmark_cache/f.json \
    --test_data_path $test_data

