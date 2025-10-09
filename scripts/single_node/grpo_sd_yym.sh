source /pfs/yangyuanming/set_proxy.sh
export PYTHONPATH=/pfs/yangyuanming/code2/flow_grpo
export WANDB_OFFLINE=true
export WANDB_DIR=../

# torchrun --nproc_per_node=8 scripts/train_sd3_v1_fsdp2_online_gpt.py \
#   --config config/grpo.py:pickscore_sd3_8gpu_online_new

torchrun --nproc_per_node=1 scripts/train_sd3_v2_fsdp2_online_gpt.py \
  --config config/grpo.py:pickscore_sd3_1gpu_online