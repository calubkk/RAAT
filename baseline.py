from tuner.evaluate.wo_retrieval import evaluate_wo_retrieval
from tuner.evaluate.w_retrieval import evaluate_w_one_retrieval,evaluate_w_noisy_retrieval,evaluate_w_two_retrieval
from tuner.utils.config import args

if args.wo_retrieval:
    evaluate_wo_retrieval(args=args,data_path=args.test_data_path,model_path=args.test_model_name_or_path,multi_eval=args.multi_eval,result_save=args.result_save,vllm=True)
elif args.w_one_retrieval:
    evaluate_w_one_retrieval(args=args,data_path=args.test_data_path,model_path=args.test_model_name_or_path,save_cache=args.selected_retrieve_cache,multi_eval=args.multi_eval,result_save=args.result_save,retrieve_type=args.retrieve_type,vllm=True)
elif args.w_two_retrieval:
    evaluate_w_two_retrieval(args=args,data_path=args.test_data_path,model_path=args.test_model_name_or_path,save_cache=args.selected_retrieve_cache,noise_rate=0.5,multi_eval=args.multi_eval,retrieve_type=args.retrieve_type,result_save=args.result_save,vllm=True)

if args.noise_ratio is not None:
    evaluate_w_noisy_retrieval(args=args,data_path=args.test_data_path,model_path=args.test_model_name_or_path,save_cache=args.selected_retrieve_cache,noise_rate=args.noise_ratio,multi_eval=args.multi_eval,vllm=True)


