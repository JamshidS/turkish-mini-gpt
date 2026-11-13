import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


BATCH = 5000 

def encode_file(tok: Tokenizer, lines, out_path: Path, eot_id: int) -> int:

    total = 0
    with open(out_path, "w") as out:
        batch = []

        def flush():
            nonlocal total
            if not batch:
                return

            for enc in tok.encode_batch(batch):
                ids = enc.ids + [eot_id]
                np.asarray(ids, dtype=np.uint16).tofile(out)
                total += len(ids)

            batch.clear()

        for line in lines:
            line = line.strip()
            if line:
                batch.append(line)
                if len(batch) >= BATCH:
                    flush()

        flush()
    return total

def read_lines(path: Path):
    with open(path, encoding="utf-8") as f:
        yield from f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/clean")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--out", default="data/bin")
    ap.add_argument("--dialog-val-ratio", type=float, default=0.02)
    ap.add_argument("--dialog-file", default="dialog.txt", help="data klasorundeki diyalog dosyasi")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(args.tokenizer)
    vocab = tok.get_vocab_size()
    # uint16 en fazla 65535'e kadar sayi tutar; sozluk bundan buyukse tasma olur (sessizce!)
    assert vocab < 2**16, f"sozluk {vocab} uint16'ya sigmaz, uint32 kullan"
    eot_id = tok.token_to_id("<|endoftext|>")
    assert eot_id is not None, "tokenizer'da <|endoftext|> yok"

    counts = {}
    counts["train"] = encode_file(tok, read_lines(data / "train.txt"), out / "train.bin", eot_id)
    counts["val"] = encode_file(tok, read_lines(data / "val.txt"), out / "val.bin", eot_id)

    # Diyalog (ince ayar icin): dialog.txt'nin sonundaki %2'yi dogrulama yap.
    # clean_data.py dosyayi zaten karistirmisti, o yuzden sondan kesmek yansiz.
    dlg = [l for l in read_lines(data / args.dialog_file) if l.strip()]
    n_val = max(1, int(len(dlg) * args.dialog_val_ratio))
    counts["dialog_train"] = encode_file(tok, dlg[:-n_val], out / "dialog_train.bin", eot_id)
    counts["dialog_val"] = encode_file(tok, dlg[-n_val:], out / "dialog_val.bin", eot_id)

    # Egitim kodu sozluk boyutunu ve eot'u buradan okuyabilsin:
    meta = {"vocab_size": vocab, "eot_id": eot_id, "dtype": "uint16", "tokens": counts}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for k, v in counts.items():
        print(f"{k:13s}: {v:>13,} token")
    print(f"cikti: {out.resolve()}")


if __name__ == "__main__":
    main()
