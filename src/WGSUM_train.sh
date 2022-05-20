#!/bin/sh
DATA_PATH=../data/mimic/bert/radiology
MODEL_PATH=../models/mimic-replication
GPUS=1,2,3,4,5
mkdir -p $MODEL_PATH

# Training
python train.py \
-mode train -accum_count 5 \
-batch_size 300 \
-bert_data_path $DATA_PATH \
-dec_dropout 0.1 \
-log_file $MODEL_PATH/training.log \
-lr 0.05 \
-model_path $MODEL_PATH \
-save_checkpoint_steps 200 \
-seed 777 \
-sep_optim false \
-train_steps 20000 \
-use_bert_emb true \
-use_interval true \
-warmup_steps 8000  \
-visible_gpus $GPUS \
-max_pos 512 \
-report_every 50 \
-enc_hidden_size 512  \
-enc_layers 6 \
-enc_ff_size 2048 \
-enc_dropout 0.1 \
-dec_layers 6 \
-dec_hidden_size 512 \
-dec_ff_size 2048 \
-encoder baseline \
-task abs
