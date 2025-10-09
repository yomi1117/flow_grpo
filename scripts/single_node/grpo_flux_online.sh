# 科学上网
source /pfs/yangyuanming/set_proxy.sh
export WANDB_OFFLINE=true
export WANDB_DIR=../

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1


# 8 GPU
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux.py --config config/grpo.py:pickscore_flux_8gpu
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero3.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux_o3.py --config config/grpo.py:pickscore_flux_8gpu


## 1 GPU to online training 
# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=1 --main_process_port 29502 scripts/train_flux_online3.py --config config/grpo.py:clipscore_flux_1gpu

## 1 GPU to online training, use bigger batch size to update reward model
export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=1 --main_process_port 29502 scripts/train_flux_online3_1.py --config config/grpo.py:clipscore_flux_1gpu_online

# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero1.yaml --num_processes=1 --main_process_port 29502 scripts/debug_train_flux.py --config config/grpo.py:pickscore_flux_1gpu_online
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=1 --main_process_port 29503 scripts/train_flux_online3_1.py --config config/grpo.py:pickscore_flux_1gpu_online



# 8GPU to online training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8 --main_process_port 29502 scripts/train_flux_online3_1.py --config config/grpo.py:clipscore_flux_8gpu_online
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8 --main_process_port 29502 scripts/train_flux_online3_1_run.py --config config/grpo.py:pickscore_flux_8gpu_online




