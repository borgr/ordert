#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c4
#SBATCH --time=7-0
#SBATCH --gres=gpu:1,vmem:8g
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=/cs/snapless/oabend/borgr/ordert/slurm/en_bert%j.out

lang=en
data_dir=/cs/labs/daphna/guy.hacohen/borgr/ordert/data
train_path=${data_dir}/train.$lang
eval_path=${data_dir}/replica_val.$lang
blimp_dir=/cs/snapless/oabend/borgr/ordert/blimp
eval_dir=/cs/snapless/oabend/borgr/ordert/transformers/output
model=bertSmall
config="/cs/snapless/oabend/borgr/ordert/configs/${model}.json"
script_path=/cs/snapless/oabend/borgr/ordert/transformers/borgr_code/run_language_modeling_with_tokenizers.py

module load tensorflow/2.0.0
source /cs/snapless/oabend/borgr/envs/tg/bin/activate
# run_language_modeling.py original version
# run_language_modeling_with_tokenizers.py -- it's the version with support for fast tokenizers, see above
#python -m torch.utils.bottleneck $script_path \
#python -m cProfile -s tottime $script_path \
python $script_path \
    --eval_blimp \
    --blimp_dir $blimp_dir\
    --overwrite_output_dir\
    --train_data_file $train_path \
    --eval_data_file $eval_path \
    --evaluate_during_training \
    --tokenizer_batch_size 10000 \
    --output_dir ./output/$model \
    --model_type bert \
    --mlm \
    --do_train \
    --do_eval \
    --line_by_line \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --save_total_limit 20 \
    --tokenizer_name /cs/snapless/oabend/borgr/ordert/transformers/tokenizers/en_tokenizer_bpe_32k \
    --save_steps 5000 \
    --per_gpu_train_batch_size 6 \
    --warmup_steps=10000 \
    --logging_steps=10 \
    --gradient_accumulation_steps=4 \
    --seed 666 --block_size=512

#    --config_name ./MyRoBERTaConfig \
#    --tokenizer_name ./MyRoBERTaConfig \