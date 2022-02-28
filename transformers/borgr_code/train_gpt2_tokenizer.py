from pathlib import Path
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer, Tokenizer, models, pre_tokenizers, processors, decoders
from transformers import GPT2TokenizerFast
from glob import glob
import os
#
# # Initialize a tokenizer
# tokenizer = Tokenizer(models.BPE())
#
# # Customize pre-tokenization and decoding
# tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
# tokenizer.decoder = decoders.ByteLevel()
# tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)


print("Training tokenizer...")
lang = "en"
data_dir = "/cs/labs/daphna/guy.hacohen/borgr/ordert/data"
paths = list(glob(f"{data_dir}/*.{lang}"))

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=False, add_prefix_space=True)

# Customize training
tokenizer.train(files=paths, vocab_size=32000, min_frequency=3, special_tokens=[
    "[UNK]",
    "[SEP]",
    "[CLS]",
    "[PAD]",
    "[MASK]",
])

# Save files to disk
OUT_DIR = f"/cs/snapless/oabend/borgr/ordert/transformers/tokenizers/{lang}_tokenizer_gpt2_32k"
os.makedirs(OUT_DIR, exist_ok=True)
tokenizer.save(OUT_DIR, lang)
