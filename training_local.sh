NNODES=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500
NPROC_PER_NODE=8
ASCEND_LAUNCH_BLOCKING=1
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train /workspace/mnt/cmss-wangdepeng/LLaMA-Factory/examples/train_full/qwen1505_full_cpt_ds3.yaml
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train /workspace/mnt/cmss-wangdepeng/LLaMA-Factory/examples/train_full/tinyllama_1.0_1.5T_cpt.yaml
