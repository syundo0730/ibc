#!/bin/bash

python3 ibc/ibc/train_eval.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/arm_test_state/mlp_ebm.gin \
  --task=ARM \
  --skip_eval=True \
  --tag=ibc_dfo \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/arm_test_state/arm_state*.tfrecord'" \
  --video
