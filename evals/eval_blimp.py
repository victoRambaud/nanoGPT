import math
import os
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

import datasets
from tqdm.auto import tqdm
from typing import Dict, Any, List

from models.gpt import GPT

from generate_tiktoken import load_local_tiktoken


class BlimpEvaluator:
    def __init__(
        self,
        model: GPT,
        enc_path: str = "/lustre/fswork/projects/rech/fku/uir17ua/dev/nanoGPT/gpt2_tiktoken_full.json",
        device: str = "cuda",
        blim_save_path: str = "/lustre/fsmisc/dataset/HuggingFace/blimp/",
        max_seq_len: int = 1024
    ):
        
        self.device = torch.device(device)
        self.model = model
        
        self.enc = load_local_tiktoken(enc_path)
        self.model = self.model.to(device)
        self.model.eval()
        self.max_seq_len = max_seq_len

        self.blim_save_path = blim_save_path
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    @torch.no_grad()
    def sentence_log_prob(
        self,
        text: str,
        device: torch.device,
        max_seq_len: int | None = 1024,
    ) -> float:
        """
        Compute log p(text) under an autoregressive LM by summing token log-probs.

        Assumes model predicts token t from prefix up to t-1 (standard GPT).
        """
        ids = self.encode(text)

        # Optionally clip to max_seq_len to avoid OOM
        if max_seq_len is not None and len(ids) > max_seq_len:
            ids = ids[:max_seq_len]

        if len(ids) < 2:
            return float("-inf")

        # Predict tokens 1..T-1 from 0..T-2
        # context: ids[:-1], targets: ids[1:]
        context = torch.tensor(ids[:-1], dtype=torch.long, device=device)#.unsqueeze(0)
        targets = torch.tensor(ids[1:], dtype=torch.long, device=device)#.unsqueeze(0)

        logits, _ = self.model(context)  # (1, T-1, V)

        # print(logits.shape, context.shape, targets.shape)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return nll.item()

    @torch.no_grad()
    def evaluate_blimp_subset(
        self,
        blimp_subset: str,
        device: str | torch.device = "cuda",
        max_seq_len: int | None = 1024,
    ) -> Dict[str, Any]:

        path = os.path.join(self.blim_save_path, blimp_subset)
        ds = datasets.load_from_disk(path)["train"]

        correct = 0
        total = 0

        for example in tqdm(ds, desc=f"BLiMP/{blimp_subset}"):
            good = example["sentence_good"]
            bad = example["sentence_bad"]

            log_p_good = self.sentence_log_prob(good, device, max_seq_len)
            log_p_bad = self.sentence_log_prob(bad, device, max_seq_len)

            # Some safety: if both scores are -inf (too short / failure), skip
            if not math.isfinite(log_p_good) and not math.isfinite(log_p_bad):
                continue

            if log_p_good > log_p_bad:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else float("nan")

        return {
            f"{blimp_subset}_accuracy": accuracy,
            f"{blimp_subset}_correct": correct,
            f"{blimp_subset}_total": total,
        }

    def evaluate_blimp_all(
        self,
        device: str | torch.device = "cuda",
        max_seq_len: int | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate on all 67 BLiMP subsets and return a dict of results.
        """
        # Full list of BLiMP phenomena as configs in nyu-mll/blimp

        results: Dict[str, Dict[str, Any]] = {}
        total_correct = 0
        total_items = 0

        for subset in os.listdir(self.blim_save_path)[:5]:
            res = self.evaluate_blimp_subset(
                blimp_subset=subset,
                device=device,
                max_seq_len=max_seq_len,
            )
            for k, v in res.items():
                results[k] = v
            total_correct += res[f"{subset}_correct"]
            total_items += res[f"{subset}_total"]

        results["overall_accuracy"] = (
            total_correct / total_items if total_items > 0 else float("nan")
        )
        results["overall_correct"] = total_correct
        results["overall_total"] = total_items

        print("Overall BLiMP accuracy:", results["overall_accuracy"])
        return results
