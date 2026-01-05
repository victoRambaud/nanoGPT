import os

os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
from datasets import load_dataset
from models.gpt import GPT
from datasets import load_dataset
from generate_tiktoken import load_local_tiktoken


def eval_pg19(model: GPT, max_len: int = 1024, batch_size: int = 8, device: str = "cuda"):

    dataset = load_dataset("emozilla/pg19", split="test", num_proc=1)
    enc = load_local_tiktoken("/lustre/fswork/projects/rech/fku/uir17ua/dev/nanoGPT/gpt2_tiktoken_full.json")
    model.eval()
    model.to(device)

    total_ppl = 0.0
    total_tokens = 0

    buffer = []  # stores token chunks of length max_len+1

    def flush_buffer(buffer):
        x = torch.stack([b[:-1] for b in buffer]).to(device)  # (bs, T)
        y = torch.stack([b[1:]  for b in buffer]).to(device)  # (bs, T)
        _, loss = model(x, y)
        ppl = torch.exp(loss)

        return ppl.item()

    with torch.no_grad():
        for example in dataset:
            ids = enc.encode_ordinary(example["text"])
            ids.append(enc.eot_token)
            ids = torch.tensor(ids, dtype=torch.long)
            print(len(ids), "leeeeen")

            for i in range(0, len(ids) - max_len - 1, max_len):
            # for i in range(0, len(ids) - max_len - 1, max_len):
                chunk = ids[i : i + max_len + 1]
                if len(chunk) != max_len + 1:
                    continue

                buffer.append(chunk)

                if len(buffer) == batch_size:
                    ppl = flush_buffer(buffer)
                    print(ppl)

            # break
                    total_ppl += ppl
                    total_tokens += 1
                    buffer.clear()
                if i == 2:
                    break

    # drop incomplete batch intentionally (standard for eval)

    ppl = total_ppl / total_tokens
    return ppl