source /pfs/yangyuanming/set_proxy.sh
export PYTHONPATH=/pfs/yangyuanming/code2/flow_grpo
export WANDB_OFFLINE=true
export WANDB_DIR=../


# SD3.5 1 GPU Online
export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file scripts/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port 29503 scripts/train_sd3_online.py --config config/grpo.py:pickscore_sd3_1gpu_online
# add eval and save ckpt
python scripts/train_sd3_v1_pure_single_online.py --config config/grpo.py:pickscore_sd3_1gpu_online
# accelerate launch --config_file scripts/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port 29503 scripts/train_sd3_online_run_new.py --config config/grpo.py:pickscore_sd3_1gpu_online

# SD3.5 4 GPU Online (commented out)
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29501 scripts/train_sd3_online.py --config config/grpo.py:pickscore_sd3_4gpu


# accelerate launch --config_file scripts/accelerate_configs/multi_gpu_online.yaml --num_processes=8 --main_process_port 29503 scripts/train_sd3_online_run.py --config config/grpo.py:pickscore_sd3_8gpu_online
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu_online.yaml --num_processes=8 --main_process_port 29503 scripts/train_sd3_online_run_new.py --config config/grpo.py:pickscore_sd3_8gpu_online
