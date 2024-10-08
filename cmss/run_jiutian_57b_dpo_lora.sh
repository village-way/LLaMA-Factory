#!/bin/bash 

# kill zombie training process 
ps -elf | grep -i factory | head -n -1 | awk '{print $4}' | while read pid; do kill -9 "$pid"; done

# clean ascend log
ASCEND_LOG=/root/ascend/log
rm -rf ${ASCEND_LOG}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_SOCKET_IFNAME=eth0

export MODELSCOPE_CACHE=/workspace/mnt/cmss-hewu/cache
export HF_DATASETS_CACHE=/workspace/mnt/cmss-hewu/cache

export FORCE_TORCHRUN=1

llamafactory-cli train cmss/jiutian_lora_dpo.yaml