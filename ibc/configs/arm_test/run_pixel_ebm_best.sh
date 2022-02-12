#!/bin/bash

python3 ibc/ibc/train_eval.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/arm_test/pixel_ebm_best.gin \
  --task=ARM \
  --skip_eval=True \
  --tag=pixel_ibc_dfo_best \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/arm_test/arm_test_*.tfrecord'" \
  --video
