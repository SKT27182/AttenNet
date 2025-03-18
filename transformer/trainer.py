import random
import time
from datasets import load_dataset
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, random_split
import einops
from tqdm import tqdm


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, trg=None, pad=2):  # 2 = <pad>
        self.src = src
        self.src_mask = einops.rearrange((src != pad), "... -> ... 1")
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)  # (batch_size, 1, seq_len)
        tgt_mask = tgt_mask & Batch.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

    @staticmethod
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
            torch.uint8
        )
        return subsequent_mask == 0


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


class Trainer:

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device=None,
        scheduler=None,
        source_tokenizer=None,
        target_tokenizer=None,
    ):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

    def run_epoch(
        self,
        data_iter,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
    ):
        """Train a single epoch"""
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        n_accum = 0
        for i, batch in enumerate(tqdm(data_iter)):
            out = self.model.forward(
                batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
            )
            loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
            # loss_node = loss_node / accum_iter
            if mode == "train" or mode == "train+log":
                loss_node.backward()
                train_state.step += 1
                train_state.samples += batch.src.shape[0]
                train_state.tokens += batch.ntokens
                if i % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    n_accum += 1
                    train_state.accum_step += 1
                scheduler.step()

            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i % 40 == 1 and (mode == "train" or mode == "train+log"):
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                print(
                    (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                    )
                    % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
                )
                start = time.time()
                tokens = 0
            del loss
            del loss_node
        return total_loss / total_tokens, train_state


class Inference:
    """Object for holding a model during inference and some utility methods used when training the model."""

    def __init__(self, src, tgt=None, pad=0):  # 0 = <blank>
        self.src = src
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = Batch.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        encoder_output = model.encode(src, src_mask)
        ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            out = model.decode(
                ys,
                encoder_output,
                src_mask,
                Batch.subsequent_mask(ys.size(1)).type_as(src.data),
            )
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat(
                [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
        return ys
