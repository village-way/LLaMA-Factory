export NNODES=1
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NPROC_PER_NODE=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train /workspace/mnt/cmss-wangdepeng/LLaMA-Factory/examples/train_full/qwen1505_full_cpt_ds3.yaml
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train /workspace/mnt/cmss-wangdepeng/LLaMA-Factory/examples/train_full/tinyllama_1.0_1.5T_cpt.yaml
