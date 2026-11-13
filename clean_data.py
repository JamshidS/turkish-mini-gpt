
import argparse
import glob
import hashlib
import html
import json
import random
import re
import unicodedata
from pathlib import Path


URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MAIL_RE = re.compile(r"\S+@\S+\.\S+")
MENTION_RE = re.compile(r"@\w+")
EMOJI_CODE_RE = re.compile(r":[a-z0-9_]{2,20}:", re.IGNORECASE)   # :roflol: 
TEMPLATE_RE = re.compile(r"@%\d+;|%\d+;")                          # '@%1;' 
REPEAT_RE = re.compile(r"(.)\1{2,}")                               # 3+ tekrar: haaaaaa -> haa
PUNCT_REPEAT_RE = re.compile(r"([!?.,])\1{2,}")                    # ?????? -> ??
SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]{1,100}>")                              # <br>, <b>, </div> ...
DOMAIN_RE = re.compile(                                            # imgyukle.com/2019/09/ (http/www yok)
    r"\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|info|biz|tr)\b(?:/\S*)?", re.IGNORECASE
)
QUOTE_RE = re.compile(r"\S+ yazdı:|alıntı:(?:\s*\S+)?")               # forum alinti kalintilari (kucuk harften sonra)
MAX_CHARS = 20000                                                  # bundan uzun belgeler kirpilir


def turkish_lower(s : str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def clean_text(s: str) -> str:
    s = html.unescape(s)                      # '&amp;' -> '&', '&quot;' -> '"'
    s = unicodedata.normalize("NFKC", s)      # farkli unicode gosterimlerini tekilleştirir
    s = TAG_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = MAIL_RE.sub(" ", s)                   # e-posta, alan adindan ONCE silinmeli
    s = DOMAIN_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = EMOJI_CODE_RE.sub(" ", s)
    s = TEMPLATE_RE.sub(" ", s)
    s = turkish_lower(s)                      # 'ReAL DRAW LA' -> 'real draw la'
    s = QUOTE_RE.sub(" ", s)
    s = REPEAT_RE.sub(r"\1\1", s)             # 'harikaaaaa' -> 'harikaa'
    s = PUNCT_REPEAT_RE.sub(r"\1\1", s)
    s = SPACE_RE.sub(" ", s).strip()          # tum bosluk/yeni satirlari tek bosluga indir
    return s


TR_CHARS = set("çğıöşü")

def is_good(s: str, min_chars: int = 80, min_alpha_ratio: float = 0.75) -> bool:
    if len(s) < min_chars:
        return False

    latters = sum(c.isalpha() for c in s)
    if latters / len(s) < min_alpha_ratio:
        return False

    if len(s) > 300 and not (set(s) & TR_CHARS):
        return False

    return True


def iter_forum(raw_dir: Path, limit: int | None):
    for path in sorted(glob.glob(str(raw_dir / "forum" / "*.jsonl"))):
        n = 0
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                t = d.get('texts', d.get('text', ''))
                yield " ".join(t) if isinstance(t, list) else str(t)
                n += 1
                if limit and n >= limit:
                    break


def iter_dialogs(raw_dir: Path, limit: int | None):
    p = raw_dir / "dialog" / "everyday-conversations-tur.json"
    if p.exists():
        data = json.load(open(p, encoding='utf-8'))
        for ex in data[:limit] if limit else data:
            parts = []
            for m in ex['conversations']:
                tag = "[K]" if m["role"] == "user" else "[A]"
                parts.append(f"{tag} {clean_text(m['content'])}")
            yield " ".join(parts)

    p = raw_dir / "dialog" / "turkish-qa-multi-dialog.json"
    if p.exists():
        data = json.load(open(p, encoding='utf-8'))
        for ex in data[:limit] if limit else data:
            yield f"[K] {clean_text(ex['input'])} [A] {clean_text(ex['output'])}"




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=".", help="forum/ ve dialog/ klasorlerinin bulundugu yer")
    ap.add_argument("--out", default="data/clean")
    ap.add_argument("--limit", type=int, default=None, help="hizli test: her dosyadan ilk N kayit")
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- forum ---
    seen = set()                            
    stats = {"okunan": 0, "filtrelenen": 0, "tekrar": 0, "kalan": 0}
    docs = []
    for raw_text in iter_forum(raw, args.limit):
        stats["okunan"] += 1
        t = clean_text(raw_text)
        if len(t) > MAX_CHARS:                # cok uzun belgeyi kelime sinirinda kirp
            t = t[:MAX_CHARS].rsplit(" ", 1)[0]
        if not is_good(t):
            stats["filtrelenen"] += 1
            continue
        h = hashlib.md5(t.encode("utf-8")).digest()  
        if h in seen:
            stats["tekrar"] += 1
            continue
        seen.add(h)
        docs.append(t)
    stats["kalan"] = len(docs)


    random.Random(args.seed).shuffle(docs)
    n_val = max(1, int(len(docs) * args.val_ratio))
    val, train = docs[:n_val], docs[n_val:]
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")

    dialogs = [d for d in iter_dialogs(raw, args.limit) if len(d) > 20]
    random.Random(args.seed).shuffle(dialogs)
    (out / "dialog.txt").write_text("\n".join(dialogs) + "\n", encoding="utf-8")

    chars = sum(len(d) for d in train)
    print(f"forum  : {stats}")
    print(f"train  : {len(train)} belge, {chars/1e6:.1f}M karakter | val: {len(val)} belge")
    print(f"dialog : {len(dialogs)} diyalog")
    print(f"cikti  : {out.resolve()}")


if __name__ == "__main__":
    main()