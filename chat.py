"""
REFERANS interaktif sohbet betigi (kendi surumunu yazarken bakmak icin).

Diyalog ince ayarindan SONRA kullan (aksi halde model cevap vermez, forum yazisi surdurur):
  python chat.py --ckpt out_dialog/ckpt_best.pt --tokenizer tokenizer.json

Komutlar:  /reset  (gecmisi sil)    /exit  (cik)
"""
import argparse
import re
import sys

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def turkish_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()      # egitim verisiyle ayni on isleme


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", turkish_lower(s)).strip()


def build_prompt(history, user_msg) -> str:
    """Egitimdeki bicimin AYNISI: '[K] soru [A] cevap [K] soru2 [A]' (parcalar tek bosluklu)."""
    parts = []
    for u, a in history:
        parts += [f"[K] {u}", f"[A] {a}"]
    parts += [f"[K] {user_msg}", "[A]"]
    return " ".join(parts)


@torch.no_grad()
def sample_reply(model, block, ids, stop_ids, max_new, temperature, top_k, rep_penalty, device):
    """Token token uretir; [K] veya <|endoftext|> gelince durur (kullanici adina konusmasin).
    block: modelin baglam uzunlugu (modelin ic ozniteliklerine bagimli kalmamak icin disaridan verilir)."""
    out = []
    x = torch.tensor([ids], device=device)
    for _ in range(max_new):
        logits, _ = model(x[:, -block:])
        logits = logits[0, -1, :].float()
        if rep_penalty != 1.0:                       # sadece BU cevapta uretilen tokenlari cezalandir
            for t in set(out):
                logits[t] = logits[t] / rep_penalty if logits[t] > 0 else logits[t] * rep_penalty
        logits = logits / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        if nxt in stop_ids:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return out


def main():
    if not sys.stdout.isatty():                      # boruya yonlendirilince Turkce karakter bozulmasin
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out_dialog/ckpt_best.pt")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--rep-penalty", type=float, default=1.2)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--device", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(args.tokenizer)
    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    stop_ids = {tok.token_to_id("[K]"), tok.token_to_id("<|endoftext|>")}
    history = []                                     # [(kullanici, asistan), ...]
    budget = cfg.block_size - args.max_new           # prompt'a ayrilan token (cevaba yer birak)
    print(f"yuklendi (adim {ck['iter']}, val {ck['val_loss']:.3f}). /reset, /exit\n")

    while True:
        try:
            msg = input("sen > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        msg = msg.strip()
        if msg == "/exit":
            break
        if msg == "/reset":
            history.clear()
            print("(gecmis silindi)\n")
            continue
        msg = normalize(msg)
        if not msg:
            continue

        # Gecmis baglama sigmazsa en eski turlari at
        ids = tok.encode(build_prompt(history, msg)).ids
        while len(ids) > budget and history:
            history.pop(0)
            ids = tok.encode(build_prompt(history, msg)).ids
        if len(ids) > budget:                        # tek mesaj bile cok uzunsa sondan kirp
            ids = ids[-budget:]

        reply_ids = sample_reply(model, cfg.block_size, ids, stop_ids, args.max_new, args.temperature,
                                 args.top_k, args.rep_penalty, device)
        # Byte-level BPE bazen yarim UTF-8 dizisi uretir; ekranda '�' olarak gorunur, temizle.
        reply = tok.decode(reply_ids).replace("�", "").strip()
        if args.debug:
            print(f"[debug] prompt {len(ids)} token, gecmis {len(history)} tur, cevap {len(reply_ids)} token")
        print(f"bot > {reply if reply else '(bos cevap)'}\n")
        history.append((msg, reply))


if __name__ == "__main__":
    main()
