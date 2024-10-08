NNODES=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500
NPROC_PER_NODE=8
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train /workspace/mnt/cmss-wangdepeng/LLaMA-Factory/examples/train_full/qwen1505_full_cpt_ds3.yaml
