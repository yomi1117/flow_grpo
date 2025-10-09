source /pfs/yangyuanming/set_proxy.sh
export PYTHONPATH=/pfs/yangyuanming/code2/flow_grpo
export WANDB_OFFLINE=true
export WANDB_DIR=../

# 1 GPU
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu

# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3_debug.py --config config/grpo.py:pickscore_sd3_1gpu
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:pickscore_sd3_1gpu

# 4 GPU
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_4gpu


# 8 GPU
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:pickscore_sd3_8gpu