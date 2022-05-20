in_path=../data/mimic/
out_path=../data/mimic/bert/
mkdir -p $out_path

CUDA_VISIBLE_DEVICES=1 python preprocess.py \
-mode format_to_bert \
-raw_path $in_path \
-save_path $out_path  \
-lower \
-n_cpus 1 \
-log_file $out_path/preprocess.log \
-type edge_words
