#!/bin/sh
IN_PATH=../data/mimic/
OUT_PATH=../data/mimic/bert/
mkdir -p $OUT_PATH

python ../graph_construction/graph_construction.py ../data/mimic/train.jsonl
python ../graph_construction/graph_construction.py ../data/mimic/valid.jsonl
python ../graph_construction/graph_construction.py ../data/mimic/test.jsonl

CUDA_VISIBLE_DEVICES=1 python preprocess.py \
-mode format_to_bert \
-raw_path $IN_PATH \
-save_path $OUT_PATH  \
-lower \
-n_cpus 1 \
-log_file $OUT_PATH/preprocess.log \
-type edge_words \
-min_src_nsents 1 \
-min_src_ntokens_per_sent 3 \
-min_tgt_ntokens 1
