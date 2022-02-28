from pathlib import Path
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from glob import glob
import os

print("Training tokenizer...")
lang = "en"
data_dir = "/cs/labs/daphna/guy.hacohen/borgr/ordert/data"
paths = list(
    glob(f"{data_dir}/*.{lang}")
)

# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer(lowercase=False, handle_chinese_chars=False)






# Customize training
tokenizer.train(files=paths, vocab_size=32000, min_frequency=3, special_tokens=[
"[UNK]",
"[SEP]",
"[CLS]",
"[PAD]",
"[MASK]",
])

# Save files to disk
OUT_DIR = f"/cs/snapless/oabend/borgr/ordert/transformers/tokenizers/{lang}_tokenizer_bpe_32k"
os.makedirs(OUT_DIR, exist_ok=True)
tokenizer.save(OUT_DIR, lang)
