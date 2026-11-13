import argparse
import io
import sys
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|endoftext|>", "[K]", "[A]"]

def line_iterator(files):
    for path in files:
        with open(path, encoding='utf-8') as f:
            for l in f:
                l = l.strip()
                if l:
                    yield l


def build_tokenizer(vocab_size: int) -> tuple[Tokenizer, trainers.BpeTrainer]:
    tok = Tokenizer(model=models.BPE()) 

    # ByteLevel'in varsayilan regex'i GPT-2'ye ozgu: "'da" -> "'d" + "a" diye boler.
    # Bu yuzden onu kapatip (use_regex=False) once kendi Split regex'imizle boluyoruz:
    # harf grubu / rakam grubu / sembol grubu. Apostrof sembol olarak ayrilir, ek ("da") butun kalir.
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"), "isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),   # 256 byte'in hepsi baslangicta var
            min_frequency=2,                      # 2'den az gecen birlesimleri ogrenme
            show_progress=True,
    )

    return tok, trainer


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/clean")
    ap.add_argument("--out", default="tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=8000)
    args = ap.parse_args()

    data = Path(args.data)
    files = [data / "train.txt", data / "dialog.txt"]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError("File not found..", missing)

    tok, trainer = build_tokenizer(args.vocab_size)
    tok.train_from_iterator(line_iterator(files=files), trainer=trainer)
    tok.save(args.out)
    print(f"Saved: {args.out} | vocab size: {tok.get_vocab_size()}")

    sample = "İstanbul'da bugün hava çok güzel, ışıl ışıl şahane bir gün!"
    sample = sample.replace("I", "ı").replace("İ", "i").lower()   # veriyle ayni on isleme
    enc = tok.encode(sample)
    print("\nexample  :", sample)
    print("tokens:", enc.tokens)
    print("back    :", tok.decode(enc.ids))
    assert tok.decode(enc.ids) == sample, "decode(encode(x)) != x  -> tokenizer kayipli!"

    dlg = tok.encode("[K] merhaba [A] merhaba! nasılsın?")
    print("diyalog :", dlg.tokens)


    val = data / "val.txt"
    if val.exists():
        text = val.read_text(encoding="utf-8")[:2_000_000]
        n_tok = len(tok.encode(text).ids)
        print(f"\nval orani: {len(text)/n_tok:.2f} karakter/token ({n_tok:,} token)")


if __name__ == "__main__":
    main()
