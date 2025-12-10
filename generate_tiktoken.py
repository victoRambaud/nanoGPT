import tiktoken
import json
import base64


def load_local_tiktoken(load_path: str = "gpt2_tiktoken_full.json") -> tiktoken.Encoding:

    with open(load_path, "r") as f:
        data = json.load(f)

    mergeable_ranks = {
        base64.b64decode(k.encode()): v
        for k, v in data["mergeable_ranks"].items()
    }

    enc = tiktoken.Encoding(
        name=data["name"],
        pat_str=data["pat_str"],
        mergeable_ranks=mergeable_ranks,
        special_tokens=data["special_tokens"],
        explicit_n_vocab=data["n_vocab"],
    )
    return enc


if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")

    save_path = "gpt2_tiktoken_full.json"

    data = {
        "name": enc.name,
        "pat_str": enc._pat_str,
        "mergeable_ranks": {
            base64.b64encode(k).decode(): v
            for k, v in enc._mergeable_ranks.items()
        },
        "special_tokens": enc._special_tokens,
        "n_vocab": enc.n_vocab,
    }

    with open(save_path, "w") as f:
        json.dump(data, f)

    print("Saved GPT-2 tokenizer to", save_path)
    print(enc.encode("your sentence here"))

    enc = load_local_tiktoken()

    print("Loaded tokenizer:", enc)
    print(enc.encode("your sentence here"))