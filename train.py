"""
REFERANS egitim betigi (kendi surumunu yazarken bakmak icin).

Iki asamada da ayni betik kullanilir:
  1) On egitim   : python train.py
  2) Ince ayar   : python train.py --train-bin dialog_train.bin --val-bin dialog_val.bin \
                      --init-from out/ckpt_best.pt --lr 1e-4 --max-iters 300 --out out_dialog

Gereken: model.py ayni klasorde, data/bin/ (prepare_data.py ciktisi).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bin")
    ap.add_argument("--train-bin", default="train.bin")
    ap.add_argument("--val-bin", default="val.bin")
    ap.add_argument("--out", default="out")
    ap.add_argument("--init-from", default=None, help="ince ayar icin baslangic checkpoint'i")
    # model
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=5)
    ap.add_argument("--n-embd", type=int, default=320)
    ap.add_argument("--dropout", type=float, default=0.0)
    # egitim
    ap.add_argument("--batch-size", type=int, default=32, help="GPU'ya bir seferde giren dizi sayisi")
    ap.add_argument("--grad-accum", type=int, default=4, help="gradyani kac adimda biriktir")
    ap.add_argument("--max-iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--compile", action="store_true", help="torch.compile (Windows'ta Triton gerektirir)")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"cihaz: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if device == "cpu":
        print("UYARI: CUDA yok, egitim cok yavas olur (CUDA'li PyTorch kur).")
    torch.set_float32_matmul_precision("high")          # TF32: ucuz hiz kazanci
    # bf16: fp16'nin aksine kayip olcekleyiciye (GradScaler) gerek yok
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / "meta.json").read_text())     # sozluk boyutu buradan: elle yazma!

    # ---------------- veri ----------------
    def get_batch(split):
        path = data / (args.train_bin if split == "train" else args.val_bin)
        # memmap'i her seferinde yeniden ac: uzun egitimde bellek sizintisini onler (nanoGPT numarasi)
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        ix = np.random.randint(0, len(arr) - args.block_size - 1, size=args.batch_size)
        x = np.stack([arr[i : i + args.block_size] for i in ix]).astype(np.int64)
        y = np.stack([arr[i + 1 : i + 1 + args.block_size] for i in ix]).astype(np.int64)  # 1 kaymis
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y

    # ---------------- model ----------------
    cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg)
    if args.init_from:                                   # ince ayar: onceki agirliklardan basla
        ck = torch.load(args.init_from, map_location="cpu")
        cfg = GPTConfig(**ck["config"])                  # modelin icine (model.config) bagimli olma
        model = GPT(cfg)
        model.load_state_dict(ck["model"])
        args.block_size = cfg.block_size                # veri penceresi modelin baglamiyla ayni olmali
        print(f"{args.init_from} yuklendi (adim {ck['iter']}, val kaybi {ck['val_loss']:.4f})")
    model.to(device)
    print(f"parametre: {model.num_params():,}")
    fwd = torch.compile(model) if args.compile else model

    # Weight decay sadece matrislere (boyut >= 2); bias ve LayerNorm'a uygulanmaz.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    def get_lr(it):                                      # warmup -> cosine azalma
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        if it >= args.max_iters:
            return args.min_lr
        r = (it - args.warmup) / (args.max_iters - args.warmup)
        return args.min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (args.lr - args.min_lr)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        res = {}
        for split in ("train", "val"):
            ls = []
            for _ in range(args.eval_iters):
                x, y = get_batch(split)
                with amp:
                    _, loss = fwd(x, y)
                ls.append(loss.item())
            res[split] = float(np.mean(ls))
        model.train()
        return res

    def save(name, it, val_loss):
        torch.save({"model": model.state_dict(), "config": cfg.__dict__, "iter": it,
                    "val_loss": val_loss, "optimizer": opt.state_dict()}, out / name)

    # ---------------- egitim dongusu ----------------
    tok_per_step = args.batch_size * args.grad_accum * args.block_size
    print(f"adim basina {tok_per_step:,} token | toplam ~{tok_per_step * args.max_iters / 1e6:.0f}M token")
    best_val = float("inf")
    model.train()
    t0 = time.time()
    for it in range(args.max_iters + 1):
        lr = get_lr(it)
        for g in opt.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0:
            L = estimate_loss()
            print(f"[eval] adim {it:5d} | train {L['train']:.4f} | val {L['val']:.4f} "
                  f"| ppl {math.exp(L['val']):.1f}")
            save("ckpt_last.pt", it, L["val"])
            if L["val"] < best_val:
                best_val = L["val"]
                save("ckpt_best.pt", it, L["val"])
        if it == args.max_iters:
            break

        loss_sum = 0.0
        for _ in range(args.grad_accum):                 # kucuk batch'leri biriktirip buyuk batch gibi davran
            x, y = get_batch("train")
            with amp:
                _, loss = fwd(x, y)
            (loss / args.grad_accum).backward()          # bolmeyi unutma: ortalama gradyan istiyoruz
            loss_sum += loss.item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)   # patlayan gradyani sinirla
        opt.step()
        opt.zero_grad(set_to_none=True)

        if it % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            n = args.log_interval if it > 0 else 1
            print(f"adim {it:5d} | kayip {loss_sum:.4f} | lr {lr:.2e} | {tok_per_step * n / dt:,.0f} tok/s")

    print(f"bitti. en iyi val kaybi: {best_val:.4f} -> {out / 'ckpt_best.pt'}")


if __name__ == "__main__":
    main()
