python run_multiple_choice.py \
--model_type $1 \
--task_name comsen \
--model_name_or_path $2 \
--do_train \
--do_eval \
--data_dir data_dir \
--learning_rate 1e-5 \
--num_train_epochs $3 \
--max_seq_length 40 \
--output_dir models_dir/$2 \
--per_gpu_eval_batch_size=16 \
--per_gpu_train_batch_size=16 \
--gradient_accumulation_steps 2 \
--overwrite_output \
--save_steps 9000000000