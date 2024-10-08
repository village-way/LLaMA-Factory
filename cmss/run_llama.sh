#!/bin/bash

cp -r /workspace/mnt/cmss-hewu/model_zoo/chatcm-57bv1-base-hf/config.json \
    /workspace/mnt/cmss-hewu/model_zoo/JIUTIAN-57B/models/config.json

python tools/infer.py > llama.log
