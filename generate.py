"""
REFERANS metin uretme betigi (kendi surumunu yazarken bakmak icin).

  python generate.py --ckpt out/ckpt_best.pt --prompt "merhaba arkadaslar" --n 3
"""
import argparse
import io
import sys

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig

EOT = "<|endoftext|>"


def turkish_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt_best.pt")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--prompt", default="merhaba")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--n", type=int, default=3, help="kac ornek uretilsin")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = Tokenizer.from_file(args.tokenizer)
    ck = torch.load(args.ckpt, map_location="cpu")
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"yuklendi: adim {ck['iter']}, val kaybi {ck['val_loss']:.3f}\n")

    ids = tok.encode(turkish_lower(args.prompt)).ids
    x = torch.tensor([ids] * args.n, device=device)           # n ornegi tek seferde uret
    out = model.generate(x, max_new_tokens=args.max_new, temperature=args.temperature, top_k=args.top_k)

    for i, row in enumerate(out.tolist(), 1):
        text = tok.decode(row, skip_special_tokens=False)
        text = text.split(EOT)[0]                              # belge bitince kes
        print(f"--- {i} ---\n{text}\n")


if __name__ == "__main__":
    main()
