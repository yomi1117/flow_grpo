# 科学上网
source /pfs/yangyuanming/set_proxy.sh
export WANDB_OFFLINE=true
export WANDB_DIR=../

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

export PYTHONPATH=/pfs/yangyuanming/code2/flow_grpo:$PYTHONPATH

# 8 GPU
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux.py --config config/grpo.py:pickscore_flux_8gpu
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux.py --config config/grpo.py:geneval_flux_8gpu
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero3.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux_o3.py --config config/grpo.py:pickscore_flux_8gpu
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux_online3.py --config config/grpo.py:clipscore_flux_1gpu

# torchrun --nproc_per_node 1 --master_port 29501 scripts/train_flux.py --config config/grpo.py:pickscore_flux_8gpu --config_file scripts/accelerate_configs/deepspeed_zero2.yaml

torchrun --nproc_per_node=1 --master_port=29501 \
  -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
  --num_processes=8 --main_process_port=29501 \
  scripts/train_flux.py --config config/grpo.py:pickscore_flux_8gpu
