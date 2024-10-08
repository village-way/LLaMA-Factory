#!/bin/bash 

. /opt/conda/etc/profile.d/conda.sh && conda activate llamafactory

source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# kill zombie training process 
ps -elf | grep -i factory | head -n -1 | awk '{print $4}' | while read pid; do kill -9 "$pid"; done

# clean ascend log
ASCEND_LOG=/root/ascend/log
rm -rf ${ASCEND_LOG}
LOG_DIR=/workspace/mnt/cmss-wangdepeng/cpt/logs

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_SOCKET_IFNAME=eth0

export MODELSCOPE_CACHE=/workspace/mnt/cmss-wangdepeng/cache
export HF_DATASETS_CACHE=/workspace/mnt/cmss-wangdepeng/cache

export FORCE_TORCHRUN=1
export NNODES=$PET_NNODES 
export RANK=$RANK
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

echo "NNODES=${NNODES} RANK=${RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

cd /workspace/mnt/cmss-wangdepeng/LLaMA-Factory
export ASCEND_PROCESS_LOG_PATH=${LOG_DIR}/plogs/rank-${RANK}
/opt/conda/envs/llamafactory/bin/llamafactory-cli train /workspace/mnt/cmss-wangdepeng/LLaMA-Factory/examples/train_full/qwen1505_full_cpt_ds3.yaml