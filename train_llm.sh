BASE_MODEL=""
OUT_PANTH=""
BASE_PROMPTS="qspell250k"
set -x

world_size=$(($ARNOLD_WORKER_NUM * $ARNOLD_WORKER_GPU))

accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
    --num_machines $ARNOLD_WORKER_NUM \
    --machine_rank $METIS_TASK_INDEX \
    --num_processes $world_size \
    --main_process_ip $METIS_WORKER_0_HOST \
    --main_process_port $METIS_WORKER_0_PORT \
    train_llm.py \
    --output_dir ${OUT_PANTH}/${BASE_PROMPTS} \
    --bf16 \
    --dataset_name "none" \
    --max_seq_length 128 \
    --per_device_train_batch_size 128 \
    --gradient_checkpointing \
    --learning_rate 3e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --report_to wandb \
    --optim "adamw_torch" \
    --run_name none \
    --logging_steps 1 \
    --log_level info 