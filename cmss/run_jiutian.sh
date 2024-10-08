#!/bin/bash

cp -r /workspace/mnt/cmss-hewu/model_zoo/JIUTIAN-57B/models/config.bak \
    /workspace/mnt/cmss-hewu/model_zoo/JIUTIAN-57B/models/config.json

python tools/infer.py > jiutian.log
