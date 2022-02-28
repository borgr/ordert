#!/bin/bash
#SBATCH --mem=64g
#SBATCH -c16
#SBATCH --time=7-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=/cs/snapless/oabend/borgr/ordert/slurm/ngram%j.out
n=$1
dir=/cs/labs/daphna/guy.hacohen/borgr/ordert
/cs/snapless/oabend/borgr/ordert/kenlm/build/bin/lmplz -o "${n}" -S 50G -T ${dir}/tmp_ngram_lm${n} --prune 10  --skip_symbols --text ${dir}/data/train.en --arpa "${dir}/egw${n}.arpa"
echo "done ${n} gram"