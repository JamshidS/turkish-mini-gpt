"""
REFERANS diyalog temizleme betigi (kendi surumunu yazarken bakmak icin).

Sorunlar (analizden):
  - everyday: 3000 diyalogun HEPSI ayni selamlasma cifti ile basliyor -> model sablonu ezberliyor
  - qa: ~%87'si ansiklopedik bilgi sorusu (bilgi yarismasi tarzi), bir kismi uydurma
Cozum: selamlasma ciftini cogu diyalogdan at, qa'dan sadece sohbet tarzi olanlari al.

  python clean_dialog.py --raw . --out data/clean --name dialog_v2.txt
"""
import argparse
import json
import random
import re
from pathlib import Path

from clean_data import clean_text      # forum verisiyle AYNI on isleme (kucuk harf, normalize...)

GREETING = "Merhaba! Bugün size nasıl yardımcı olabilirim?"

# Bilgi sorusu kaliplari
TRIVIA_RE = re.compile(
    r"\b(kimdir|nedir|nerede|neresidir|hangi|kaç|kaçtır|nelerdir|ne zaman|kimin|"
    r"kimlerdir|hangisi|kimler|kimlere|nereye|neden|nasıl bir)\b", re.I)
FAKE_RE = re.compile(r"aeroattention", re.I)      # veride tekrar eden uydurma kutuphane


def keep_qa(q: dict) -> bool:
    i, o = q["input"].strip(), q["output"].strip()
    if FAKE_RE.search(i) or FAKE_RE.search(o):
        return False
    if TRIVIA_RE.search(i):
        return False
    if "**" in o or "\n" in o or re.search(r"(^|\n)\s*\d+\.", o):   # asistan tarzi liste / markdown
        return False
    if not (3 <= len(i) <= 100) or not (15 <= len(o) <= 300):
        return False
    # ilk kelime disinda buyuk harfle baslayan kelime = ozel isim = muhtemelen bilgi sorusu
    if re.findall(r"\b[A-ZÇĞİÖŞÜ]\w+", i)[1:]:
        return False
    if re.search(r"\d{3,}", i + " " + o):                           # yil, buyuk sayi -> olgu sorusu
        return False
    return True


def fmt(turns) -> str:
    """[(rol, metin), ...] -> '[K] ... [A] ...' (egitimdeki bicimin ayni, tek bosluklu)."""
    return " ".join(f"{'[K]' if r == 'user' else '[A]'} {clean_text(t)}" for r, t in turns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=".")
    ap.add_argument("--out", default="data/clean")
    ap.add_argument("--name", default="dialog_v2.txt")
    ap.add_argument("--keep-greeting", type=float, default=0.05,
                    help="selamlasma ciftini koruyacagimiz diyalog orani")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    raw = Path(args.raw) / "dialog"

    out_lines = []

    # ---- everyday ----
    ev = json.load(open(raw / "everyday-conversations-tur.json", encoding="utf-8"))
    n_greet = n_ev = 0
    for ex in ev:
        turns = [(m["role"], m["content"]) for m in ex["conversations"]]
        starts_with_template = (len(turns) >= 2 and turns[0][0] == "user"
                                and turns[1][1].strip() == GREETING)
        if starts_with_template:
            if rng.random() < args.keep_greeting:
                n_greet += 1                       # birkac diyalogda selamlasma kalsin
            else:
                turns = turns[2:]                  # ilk kullanici-asistan ciftini at
        if len(turns) >= 2 and turns[0][0] == "user":
            out_lines.append(fmt(turns))
            n_ev += 1

    # ---- qa ----
    qa = json.load(open(raw / "turkish-qa-multi-dialog.json", encoding="utf-8"))
    seen = set()
    n_qa = 0
    for q in qa:
        if not keep_qa(q):
            continue
        key = (q["input"].strip().lower(), q["output"].strip().lower())
        if key in seen:                            # tekrar edeni at
            continue
        seen.add(key)
        out_lines.append(fmt([("user", q["input"]), ("assistant", q["output"])]))
        n_qa += 1

    out_lines = [l for l in out_lines if len(l) > 20]
    rng.shuffle(out_lines)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / args.name).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"everyday: {len(ev)} -> {n_ev} (selamlasma korunan: {n_greet})")
    print(f"qa      : {len(qa)} -> {n_qa}")
    print(f"toplam  : {len(out_lines)} diyalog -> {out / args.name}")


if __name__ == "__main__":
    main()
