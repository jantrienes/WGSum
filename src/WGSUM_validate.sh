#!/bin/sh
DATA_PATH=../data/mimic/bert/radiology
MODEL_PATH=../models/mimic-replication

python train.py \
-task abs \
-mode validate \
-batch_size 3000 \
-test_batch_size 500 \
-bert_data_path $DATA_PATH \
-log_file $MODEL_PATH/validate.log \
-model_path $MODEL_PATH \
-sep_optim true \
-use_interval true \
-visible_gpus 0 \
-max_pos 512 \
-max_length 50 \
-alpha 0.95 \
-min_length 6 \
-result_path $MODEL_PATH/summaries \
-test_all
