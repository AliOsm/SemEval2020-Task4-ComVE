python run_multiple_choice.py \
--model_type $1 \
--task_name comsen \
--model_name_or_path models_dir/$2 \
--do_eval \
--data_dir data_dir \
--max_seq_length 40 \
--output_dir models_dir/$2 \
--per_gpu_eval_batch_size=16
