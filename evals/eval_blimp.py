
import math
import os
import json
import pickle
import hashlib

import torch
import torch.nn.functional as F
import datasets
from tqdm.auto import tqdm
from typing import Dict, Any, Optional
from tokenizers import Tokenizer

from models.gpt import GPT


class BlimpEvaluator:
    def __init__(
        self,
        model: GPT,
        enc_path: str = "data/babyLM/tokenizer.json",
        device: str = "cuda",
        blim_save_path: str = "/lustre/fsmisc/dataset/HuggingFace/blimp/",
        max_seq_len: int = 1024,
        n_datasets: Optional[int] = None,
        tokenized_cache_dir: str = "data/blimp_tokenized",
    ):
        self.device = torch.device(device)
        self.model = model.to(device)
        self.enc = Tokenizer.from_file(enc_path)
        self.max_seq_len = max_seq_len
        self.blim_save_path = blim_save_path
        self.n_datasets = n_datasets
        self.tokenized_cache_dir = tokenized_cache_dir

        with open(os.path.join(os.path.dirname(enc_path), "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        self.eos_id = meta["eot_token_id"]

        if self.enc.get_vocab_size(with_added_tokens=True) != meta["vocab_size"]:
            raise ValueError("BLiMP tokenizer differs from the training tokenizer")

        with open(enc_path, "rb") as f:
            self.tokenizer_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, add_special_tokens=False).ids

    def load_tokenized_subset(self, blimp_subset: str, split: str = "train"):
        path = os.path.join(self.blim_save_path, blimp_subset)
        ds = datasets.load_from_disk(path)

        if isinstance(ds, datasets.DatasetDict):
            ds = ds[split]
        elif split != "train":
            raise ValueError(f"{blimp_subset} has no {split!r} split")

        cache_dir = os.path.join(self.tokenized_cache_dir, blimp_subset)
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"{split}_{self.tokenizer_hash}_{ds._fingerprint}.json"
        )

        if os.path.isfile(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)

        pairs = [
            [self.encode(row["sentence_good"]), self.encode(row["sentence_bad"])]
            for row in tqdm(ds, desc=f"Tokenizing BLiMP/{blimp_subset}/{split}")
        ]

        tmp_path = cache_path + f".{os.getpid()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f)
        os.replace(tmp_path, cache_path)
        return pairs

    @torch.no_grad()
    def sentence_log_prob(
        self,
        text: str,
        device: torch.device | None = None,
        max_seq_len: int | None = None,
    ) -> float:
        return self.token_log_prob(self.encode(text), device, max_seq_len)

    @torch.no_grad()
    def token_log_prob(
        self,
        ids: list[int],
        device: torch.device | None = None,
        max_seq_len: int | None = None,
    ) -> float:
        if not ids:
            return float("-inf")

        device = self.device if device is None else torch.device(device)
        max_seq_len = self.max_seq_len if max_seq_len is None else max_seq_len

        if max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")

        # Score every token, including the first one conditioned on EOS.
        # Do not truncate sentences: use overlapping context windows.
        ids = [self.eos_id] + ids
        total_log_prob = 0.0
        stride = max(1, max_seq_len // 2)

        for start in range(1, len(ids), stride):
            end = min(start + stride, len(ids))
            left = max(0, end - 1 - max_seq_len)

            context = torch.tensor(
                ids[left:end - 1], dtype=torch.long, device=device
            ).unsqueeze(0)
            targets = torch.tensor(
                ids[left + 1:end], dtype=torch.long, device=device
            ).unsqueeze(0)

            logits, _, _ = self.model(context, targets)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            token_scores = log_probs.gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)

            # Only count new targets, not the overlapping context.
            total_log_prob += token_scores[0, -(end - start):].sum().item()

        return total_log_prob

    @torch.no_grad()
    def evaluate_blimp_subset(
        self,
        blimp_subset: str,
        device: str | torch.device | None = None,
        max_seq_len: int | None = None,
        split: str = "train",
    ) -> Dict[str, Any]:

        pairs = self.load_tokenized_subset(blimp_subset, split)

        correct = 0
        total = 0
        was_training = self.model.training
        self.model.eval()

        try:
            for good_ids, bad_ids in tqdm(
                pairs, desc=f"BLiMP/{blimp_subset}/{split}"
            ):
                log_p_good = self.token_log_prob(good_ids, device, max_seq_len)
                log_p_bad = self.token_log_prob(bad_ids, device, max_seq_len)

                if not (math.isfinite(log_p_good) and math.isfinite(log_p_bad)):
                    continue

                if log_p_good > log_p_bad:
                    correct += 1
                total += 1
        finally:
            self.model.train(was_training)

        accuracy = correct / total if total > 0 else float("nan")
        prefix = blimp_subset if split == "train" else f"{blimp_subset}_{split}"

        return {
            f"{prefix}_accuracy": accuracy,
            f"{prefix}_correct": correct,
            f"{prefix}_total": total,
        }

    def evaluate_blimp_all(
        self,
        device: str | torch.device | None = None,
        max_seq_len: int | None = None,
        subsets_contains: str | None = None,
    ) -> Dict[str, Any]:

        results: Dict[str, Any] = {}
        totals = {"train": [0, 0], "validation": [0, 0], "val": [0, 0]}

        ds_list = sorted(
            name for name in os.listdir(self.blim_save_path)
            if os.path.isdir(os.path.join(self.blim_save_path, name))
            and (
                os.path.isfile(os.path.join(self.blim_save_path, name, "dataset_dict.json"))
                or os.path.isfile(os.path.join(self.blim_save_path, name, "dataset_info.json"))
            )
        )

        if subsets_contains is not None:
            ds_list = [name for name in ds_list if subsets_contains in name]

        if self.n_datasets is not None:
            ds_list = ds_list[:self.n_datasets]

        was_training = self.model.training
        self.model.eval()

        try:
            for subset in ds_list:
                data = datasets.load_from_disk(
                    os.path.join(self.blim_save_path, subset)
                )
                splits = data.keys() if isinstance(data, datasets.DatasetDict) else ["train"]

                for split in ("train", "validation", "val"):
                    if split not in splits:
                        continue

                    res = self.evaluate_blimp_subset(
                        subset, device, max_seq_len, split
                    )

                    prefix = subset if split == "train" else f"{subset}_{split}"
                    correct = res[f"{prefix}_correct"]
                    total = res[f"{prefix}_total"]

                    totals[split][0] += correct
                    totals[split][1] += total
                    results[f"{prefix}_accuracy"] = res[f"{prefix}_accuracy"]

                    print(f"{prefix}_accuracy: {res[f'{prefix}_accuracy']:.4f}")
        finally:
            self.model.train(was_training)

        for split, (correct, total) in totals.items():
            if total:
                key = "overall_accuracy" if split == "train" else f"overall_{split}_accuracy"
                results[key] = correct / total
                print(f"{key}: {results[key]:.4f}")

        if "overall_accuracy" not in results:
            raise ValueError("No BLiMP train split found in local datasets")

        return results
