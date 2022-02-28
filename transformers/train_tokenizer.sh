#!/bin/bash
#SBATCH --mem=80g
#SBATCH -c4
#SBATCH --time=2-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=/cs/snapless/oabend/borgr/ordert/slurm/en_tok%j.out

script_path=/cs/snapless/oabend/borgr/ordert/transformers/borgr_code/train_bert_tokenizer.py
script_path=/cs/snapless/oabend/borgr/ordert/transformers/borgr_code/train_gpt2_tokenizer.py
module load tensorflow/2.0.0
source /cs/snapless/oabend/borgr/envs/tg/bin/activate
python $script_path
echo "done"