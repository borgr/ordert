# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import argparse
import glob
import json
import logging
import os
import sys
import pickle
import random
import re
import shutil
import subprocess
import uuid
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup, GPT2TokenizerFast,
    TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizerFast
)

sys.path.append(os.path.dirname(__file__) + os.sep + "..")
print("adding current dir as a workdir", os.path.dirname(__file__))
from borgr_code.embed_ff import BowGptLMHeadModel
from borgr_code.embed_fc import BowFCLMHeadModel

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bowgpt": (GPT2Config, BowGptLMHeadModel, GPT2TokenizerFast),
    "bowfc": (GPT2Config, BowFCLMHeadModel, GPT2TokenizerFast),
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
    "transformerXL": (TransfoXLConfig, TransfoXLLMHeadModel, PreTrainedTokenizerFast)
}


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        # block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)
        # block_size
        #
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LibeByLineGeneratorDataset(IterableDataset):
    def __init__(self, t: PreTrainedTokenizer, args, file_path: str, block_size=512, shuffle=False):
        self.file_path = file_path
        self.shuffled_path = None
        self.shuffle = shuffle
        self.length = None
        self.args = args
        self.block_size = block_size
        assert os.path.isfile(file_path)
        logger.info("Creating features from dataset file at %s", file_path)

        self.tokenizer = load_tokenizer(args, block_size)

    def __len__(self):
        if self.length == None:
            self.length = self.file_len()
        return self.length

    def shuffle_file(self):
        if self.shuffle:
            self.shuffled_path = self.file_path + str(uuid.uuid4())[4] + ".shuff"
            command = f"shuf {self.file_path} -o {self.shuffled_path}"
            logger.info(f"command {command}")
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            process.wait()
        else:
            self.shuffled_path = self.file_path

    def __iter__(self):
        self.shuffle_file()
        self.examples = []
        lines = []
        skipped = 0
        with open(self.shuffled_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if self.good_line(line):
                    lines.append(line)
                else:
                    skipped += 1
                if len(lines) > self.args.tokenizer_batch_size:
                    logger.info(f"Tokenized {i} out of which {skipped} were skipped")
                    encoded = self.encode_batch(lines, self.args)
                    for encoding in encoded:
                        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
                            yield torch.tensor(encoding, dtype=torch.long)
                        else:
                            yield torch.tensor(encoding.ids, dtype=torch.long)
                    lines = []
        encoded = self.encode_batch(lines, self.args)
        logging.info(f"ending tokenization smaller batch {len(encoded)}")
        if self.shuffle:
            os.remove(self.shuffled_path)
        for encoding in encoded:
            if isinstance(self.tokenizer, PreTrainedTokenizerFast):
                yield torch.tensor(encoding, dtype=torch.long)
            else:
                yield torch.tensor(encoding.ids, dtype=torch.long)

    def encode_batch(self, lines, args):
        if "gpt" in args.tokenizer_name.lower():
            batch_examples = self.tokenizer.encode_batch(lines)
            examples = [example for example in batch_examples if len(example.tokens) < self.block_size]
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            examples = self.tokenizer.batch_encode_plus(lines)["input_ids"]
            examples = [example for example in examples if len(example) < self.block_size]
        else:
            examples = self.tokenizer.encode_batch(lines)
        return examples

    def good_line(self, line):
        return line.strip()

    def file_len(self):
        i = 0
        with open(self.file_path) as f:
            for line in f:
                if self.good_line(line):
                    i += 1
        return i


loaded_tokenizers = {}


def load_tokenizer(args, block_size):
    global loaded_tokenizers
    if (args.tokenizer_name, block_size) in loaded_tokenizers:
        return loaded_tokenizers[(args.tokenizer_name, block_size)]
    bert_tokenizer = os.path.join(args.tokenizer_name, "vocab.txt") if args.tokenizer_name else ""
    if not args.tokenizer_name:
        logger.info("Loading default tokenizer")
        tokenizer = MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    elif "gpt" in bert_tokenizer.lower():
        from tokenizers import ByteLevelBPETokenizer
        logger.info("Loading GPT2 tokenizer")
        tokenizer = ByteLevelBPETokenizer(os.path.join(args.tokenizer_name, "vocab.json"),
                                          os.path.join(args.tokenizer_name, "merges.txt"), lowercase=False,
                                          add_prefix_space=True)
        tokenizer.pad_token = None
        tokenizer._pad_token = None
        # tokenizer.enable_truncation(block_size)  # copied from bert, might not fit gpt2
    elif os.path.exists(bert_tokenizer):
        logger.info("Loading BERT tokenizer")
        from tokenizers import BertWordPieceTokenizer
        tokenizer = BertWordPieceTokenizer(os.path.join(args.tokenizer_name, "vocab.txt"),
                                           handle_chinese_chars=False, lowercase=False)
        tokenizer.enable_truncation(block_size)
    elif "XL" in bert_tokenizer:
        logger.info("Loading pretrained XL tokenizer")
        tokenizer = TransfoXLTokenizerFast().from_pretrained('transfo-xl-wt103')
        tokenizer.pad_token = None
    else:
        from tokenizers import ByteLevelBPETokenizer
        from tokenizers.processors import BertProcessing
        logger.info("Loading RoBERTa tokenizer")

        tokenizer = ByteLevelBPETokenizer(
            os.path.join(args.tokenizer_name, "vocab.json"),
            os.path.join(args.tokenizer_name, "merges.txt")
        )
        tokenizer.pad_token = None
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=block_size)
    loaded_tokenizers[(args.tokenizer_name, block_size)] = tokenizer
    return tokenizer


class LineByLineBlimpDataset(Dataset):
    def __init__(self, t: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # logger.info("Creating features from blimp file at %s", file_path)

        self.tokenizer = load_tokenizer(args, block_size)
        # logger.info("Running tokenization")

        lines = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                lines.append(row["sentence_good"])
                lines.append(row["sentence_bad"])
        assert len(lines) == 2000
        if "gpt" in args.tokenizer_name.lower():
            self.examples = self.tokenizer.encode_batch(lines)
            self.examples = [example for example in self.examples if len(example.tokens) < block_size]
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            self.examples = self.tokenizer.batch_encode_plus(lines)["input_ids"]
            self.examples = [example for example in self.examples if len(example) < block_size]
        else:
            self.examples = self.tokenizer.encode_batch(lines)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            return torch.tensor(self.examples[i], dtype=torch.long)
        return torch.tensor(self.examples[i].ids, dtype=torch.long)


class LineByLineTextDataset(Dataset):
    def __init__(self, t: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        logger.info("Creating features from dataset file at %s", file_path)

        # -------------------------- CHANGES START
        bert_tokenizer = os.path.join(args.tokenizer_name, "vocab.txt") if args.tokenizer_name else ""
        if not os.path.isfile(bert_tokenizer) and os.path.isfile(os.path.join(args.tokenizer_name, "vocab.json")):
            with open(os.path.join(args.tokenizer_name, "vocab.json")) as fl:
                json_dict = json.load(fl)
            with open(bert_tokenizer) as fl:
                for i in range(len(json_dict)):
                    fl.write(json_dict[i] + "\n")
            logger.info("created")

        self.tokenizer = load_tokenizer(args, block_size)
        logger.info("Running tokenization")
        self.examples = []
        lines = []
        with open(file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if (len(line) > 0 and not line.isspace()):
                    lines.append(line)
                if len(lines) > args.tokenizer_batch_size:
                    logger.info(f"Tokenized {i}")
                    if "gpt" in bert_tokenizer.lower():
                        batch_examples = self.tokenizer.encode_batch(lines)
                        self.examples += [example for example in batch_examples if
                                          len(example.tokens) < block_size]
                    elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
                        batch_examples = self.tokenizer.batch_encode_plus(lines)["input_ids"]
                        self.examples += [example for example in batch_examples if
                                          len(example.tokens) < self.block_size]
                    else:
                        self.examples += self.tokenizer.encode_batch(lines)
                    lines = []
        if "gpt" in bert_tokenizer.lower():
            batch_examples = self.tokenizer.encode_batch(lines)
            self.examples += [example for example in batch_examples if len(example.tokens) < block_size]
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            batch_examples = self.tokenizer.batch_encode_plus(lines)["input_ids"]
            self.examples += [example for example in batch_examples if len(example.tokens) < self.block_size]
        else:
            self.examples += self.tokenizer.encode_batch(lines)

        logger.info("Tokenization done.")

        # -------------------------- CHANGES END

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            return torch.tensor(self.examples[i], dtype=torch.long)
        return torch.tensor(self.examples[i].ids, dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        if args.data_fits_memory:
            return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
        else:
            return LibeByLineGeneratorDataset(tokenizer, args, file_path=file_path, block_size=args.block_size,
                                              shuffle=not evaluate)
    else:
        return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


def set_seed(args):
    logging.info(f"Seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        logger.info(f"checkpoints: {checkpoints_sorted}")
        shutil.rmtree(checkpoint)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    # logger.info(f"Padding token: {tokenizer.pad_token}")
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer.get_vocab()), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def generate(args, model, tokenizer, input_sents):
    """
    generates, consider using run_generation.py
    :param args:
    :param model:
    :param tokenizer:
    :param input_sents:
    :param sent_len:
    :return:
    """
    gen_tokenizer = load_tokenizer(args, args.block_size)
    out_sents = []
    # sents = [".", "The", "The boy did", "Little did he know"]
    for in_sent in input_sents:
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            input = gen_tokenizer.batch_encode_plus([in_sent])["input_ids"]
            input = torch.tensor([inp for inp in input], dtype=torch.long)
        else:
            input = gen_tokenizer.encode_batch([in_sent])
            input = torch.tensor([inp.ids for inp in input], dtype=torch.long)
        input = input.to(args.device)

        out_sent = model.generate(input, max_length=args.length + len(input[0]), temperature=args.temperature,
                                  top_k=args.k,
                                  top_p=args.p,
                                  repetition_penalty=args.repetition_penalty,
                                  do_sample=True,
                                  num_return_sequences=args.num_return_sequences)
        out_sent = out_sent.tolist()
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            out_sent = gen_tokenizer.decode(out_sent[0])
        else:
            out_sent = gen_tokenizer.decode_batch(out_sent)
        out_sents.append(out_sent)
        # logger.info(f"Random sentence starting with {in_sent}: {out_sent}")
    return out_sents


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    non_wrapped_model = model
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    if args.eval_blimp or args.blimp_dir:
        if not (args.eval_blimp and args.blimp_dir):
            raise ValueError(
                f"When evaluating blimp the location should be specified evaluating? {args.eval_blimp} path: {args.blimp_dir}")
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    def collate(examples: List[torch.Tensor]):
        # logger.info(f"Padding token: {tokenizer.pad_token}")
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    if args.data_fits_memory:
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.train_batch_size, collate_fn=collate
        )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    if args.prompts_path:
        os.makedirs(os.path.join(args.output_dir, "gens"),exist_ok=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer.get_vocab()))
    # assert len(tokenizer.get_vocab()) == tokenizer.vocab_size, (len(tokenizer.get_vocab()), (tokenizer.vocab_size, len(tokenizer.additional_special_tokens)))

    model.zero_grad()
    # train_iterator = trange(
    #     epochs_trained, int(args.num_train_epochs), miniters=1, desc="Epoch",
    #     disable=args.local_rank not in [-1, 0])
    train_iterator = range(epochs_trained, int(args.num_train_epochs))
    set_seed(args)  # Added here for reproducibility
    for epoch in train_iterator:
        logger.info(f"Epoch: {epoch}")
        # epoch_iterator = tqdm(train_dataloader, miniters=100, desc="Iteration",
        #                       disable=args.local_rank not in [-1, 0])
        epoch_iterator = train_dataloader
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            # logger.info(f"mem: {torch.cuda.memory_allocated()}")
            # logger.info(f"inputs.shape {inputs.shape}")
            model.train()
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                torch.cuda.empty_cache()
                global_step += 1
                if global_step % (max(args.logging_steps, 100) // 10) == 0:
                    logger.info(f"Done step {global_step}")
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        gen_tokenizer = load_tokenizer(args, args.block_size)
                        starting_sents = [".", "The", "The boy did", "Little did he know"]
                        for starting_sent in starting_sents:
                            if isinstance(tokenizer, PreTrainedTokenizerFast):
                                input = gen_tokenizer.batch_encode_plus([starting_sent])["input_ids"]
                                input = torch.tensor([inp for inp in input], dtype=torch.long)
                            else:
                                input = gen_tokenizer.encode_batch([starting_sent])
                                input = torch.tensor([inp.ids for inp in input], dtype=torch.long)
                            input = input.to(args.device)

                            sent = non_wrapped_model.generate(input, max_length=20)
                            sent = sent.tolist()
                            if isinstance(tokenizer, PreTrainedTokenizerFast):
                                sent = gen_tokenizer.decode(sent[0])
                            else:
                                sent = gen_tokenizer.decode_batch(sent)
                            logger.info(f"Random sentence starting with {starting_sent}: {sent}")

                        if args.eval_blimp or args.blimp_dir:
                            evaluate_blimp(args, model, tokenizer, global_step, results["perplexity"])
                        if args.prompts_path:
                            # generate a sentence for each prompt
                            with open(args.prompts_path) as fl:
                                generation_results = generate(args, non_wrapped_model, tokenizer, fl)
                            with open(
                                    os.path.join(args.output_dir, "gens", f"steps{global_step}_perplexity{results['perplexity']}"),
                                    "w") as fl:
                                logger.info(f"generation_results first line: {generation_results[0]}")
                                for line in generation_results:
                                    fl.write(str(line) + "\n")

                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    # tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate_blimp_sub_dataset(args, model, tokenizer, path, outdir_name):
    blimp_dataset = LineByLineBlimpDataset(tokenizer, args, path, block_size=args.block_size)
    # evaluate with current model
    per_gpu = 1  # args.per_gpu_eval_batch_size can't be used as otherwise loss is aggregated on the gpu batch
    args.eval_batch_size = per_gpu * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        # logger.info(f"Padding token: {tokenizer.pad_token}")
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(blimp_dataset)
    blimp_dataloader = DataLoader(
        blimp_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # # Eval!
    # logger.info("***** Running evaluation {} *****".format(prefix))
    # logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Batch size = %d", args.eval_batch_size)
    # eval_loss = 0.0
    # nb_eval_steps = 0
    model.eval()
    # logger.info(f"len(blimp_dataloader) {len(blimp_dataloader)} len(blimp_dataset) {len(blimp_dataset)}")
    lm_losses = []
    # for batch in tqdm(blimp_dataloader, miniters=10, desc="Evaluating"):
    for i, batch in enumerate(blimp_dataloader):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        # logger.info(f"{i} #inputs {len(inputs)}")
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            # logger.info(f"inputs shape {inputs.shape} ")
            # logger.info(f"loss {outputs[0]}, loss num {len(outputs[0])}, {len(lm_loss.flatten().tolist())} " + f"flatten {lm_loss.flatten().tolist()}")
            # logger.info(f"outputs.shape {[output.shape for output in outputs]}")
            # if len(lm_loss.to_list()) != len(lm_loss.flatten().to_list()):
            #     logger.info("Flattening to per sentence score")
            #     res = lm_loss.flatten().to_list()
            if "XL" in args.model_type:
                lm_loss = torch.exp(torch.mean(torch.log(lm_loss), 1))
            lm_losses += lm_loss.tolist()

        # # DELETE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        # logging.info(f"{i} inputs {len(inputs)}")
        # inputs = inputs.to(args.device)
        # labels = labels.to(args.device)
        #
        # with torch.no_grad():
        #     tmp_outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
        #     tmp_lm_loss = tmp_outputs[0]
        #     logger.info(
        #         f"tmp:loss {tmp_outputs[0]}, loss num {len(tmp_outputs[0])} flaten {lm_loss.flatten().tolist()}")
        #     # logger.info(f"tmp_outputs.shape {[output.shape for output in tmp_outputs]}")
        #     assert tmp_lm_loss.tolist() == lm_loss.tolist()
        # inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch[:1], batch[:1])
        # logging.info(f"{i} inputs {len(inputs)}")
        # inputs = inputs.to(args.device)
        # labels = labels.to(args.device)
        #
        # with torch.no_grad():
        #     tmp_outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
        #     tmp_lm_loss = tmp_outputs[0]
        #     logger.info(
        #         f"tmp:loss {tmp_outputs[0]}, loss num {len(tmp_outputs[0])} flaten {lm_loss.flatten().tolist()}")
        #     # logger.info(f"tmp_outputs.shape {[output.shape for output in tmp_outputs]}")
        #     assert tmp_lm_loss.tolist() == lm_loss[:1].tolist()
        # #### delete until here!!!!!!!!!!!!!!!!!!!
        #
        # #     eval_loss += lm_loss.mean().item()
        # nb_eval_steps += 1

    assert len(lm_losses) == 2000, f"Unexpected len {len(lm_losses)} of lm_losses {lm_losses}"
    # eval_loss = eval_loss / nb_eval_steps
    # perplexity = torch.exp(torch.tensor(eval_loss))
    #
    # result = {"perplexity": perplexity}
    ############################
    # write to args.eval_blimp_dest + subpath + filename good_prob, bad_prob
    filename = os.path.splitext(os.path.basename(path))[0]
    outdir_name = os.path.join(args.output_dir, "blimp", outdir_name)
    os.makedirs(outdir_name, exist_ok=True)
    outpath = os.path.join(outdir_name, filename + ".txt")
    correct = 0
    wrong = 0
    with open(outpath, "w") as fl:
        for i, lm_loss in enumerate(iterate_losses(lm_losses)):
            fl.write(f"{lm_loss}\n")
            if i % 2 == 1:  # lm_loss is like perplexity (need e^ [loss * token num]), lower is better
                bad_loss = lm_loss
                if bad_loss > good_loss:
                    correct += 1
                else:
                    wrong += 1
            else:
                good_loss = lm_loss
    # logger.info(f"i {i} #lm_loss {len(lm_losses)}")
    logger.info(
        f" {os.path.basename(outpath)} correct: {correct}/{wrong + correct}={correct / (wrong + correct)} ")


def iterate_losses(losses):
    """iterate losses whether they come from multiple gpus or from one"""
    for loss in losses:
        try:
            if len(loss) == 1:
                yield loss
            else:
                for subloss in loss:
                    yield subloss
        except TypeError as e:
            yield loss
            # logging.info(f"loss {loss}")


def evaluate_blimp(args, model, tokenizer, steps, perplexity):
    outdir_name = f"steps{steps}_perplexity{perplexity}"
    for root, dirs, filenames in os.walk(args.blimp_dir):
        for filename in filenames:
            if filename.endswith("jsonl"):
                evaluate_blimp_sub_dataset(args, model, tokenizer, os.path.join(root, filename), outdir_name)


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    def collate(examples: List[torch.Tensor]):
        # logger.info(f"Padding token: {tokenizer.pad_token}")
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    if args.data_fits_memory:
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
        )
    else:
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.eval_batch_size, collate_fn=collate
        )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    # eval_dataloader = tqdm(eval_dataloader, miniters=10, desc="Evaluating")
    for batch in eval_dataloader:
        if len(batch) != args.eval_batch_size:
            logger.info(f"skipping smaller (last) batch to eval:{len(batch)}")
            continue
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            # logger.info(f"lm_loss {lm_loss.shape}")
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--data_fits_memory",
        action="store_true",
        help="Whether to keep data in memory.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        help="Window size when relevant.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--eval_blimp",
        action="store_true",
        help="If set, saves the blimp probabilities every eval step",
    )
    parser.add_argument(
        "--blimp_dir",
        default=None,
        type=str,
        help="Specifies the path to directory containing blimp style files (jsonl with sentence_good sentence_bad and possibly other characteristics) ",
    )
    parser.add_argument(
        "--prompts_path",
        default=None,
        type=str,
        help="Specifies the path to file containing prompts (sentences separated by lines)",
    )
    parser.add_argument(
        "--tokenizer_batch_size",
        default=-1,
        type=int,
        help="The maximum number of lines per batch ",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
             "The training dataset will be truncated in block of this size for training."
             "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    #### generation arguments
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="The number of samples to generate per input prompt.")
    args = parser.parse_args()

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]
            args.overwrite_output_dir = True
            logger.info(f"Should continue from {args.model_name_or_path}")
    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    return args


def main():
    args = parse_arguments()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    if args.seed is None:
        args.seed = random.randint(0, 100000)
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    tokenizer = load_tokenizer(args, args.block_size)
    # if args.tokenizer_name:
    #     tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    # elif args.model_name_or_path:
    #     tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    # else:
    #     raise ValueError(
    #         "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
    #         "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
    #     )
    # logger.info(f"tokenizer length {len(tokenizer)}, {dir(tokenizer)}")
    # if args.block_size <= 0:
    #     args.block_size = tokenizer.max_len
    #     # Our input block size will be the max possible for the model
    # else:
    #     args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
        if args.window_size is not None:
            model.set_window(args.window_size)
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        logger.info("loading")
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        logger.info("device")
        model.to(args.device)
        logger.info("loaded")

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
