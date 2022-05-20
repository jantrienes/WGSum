#!/bin/sh
DATA_PATH=../bert_openI/radiology/radiology
MODEL_PATH=../models/openi
CHECKPOINT=$MODEL_PATH/openi.pt
GPUS=0

python ../src/train.py \
-task abs \
-mode test \
-batch_size 3000 \
-test_batch_size 500 \
-bert_data_path $DATA_PATH \
-log_file $MODEL_PATH/test.log \
-model_path $MODEL_PATH \
-sep_optim true \
-use_interval true \
-visible_gpus $GPUS \
-max_pos 512 \
-max_length 50 \
-alpha 0.95 \
-min_length 6 \
-result_path $MODEL_PATH/summaries \
-test_from $CHECKPOINT \
