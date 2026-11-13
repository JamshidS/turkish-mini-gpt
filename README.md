# turkish-mini-gpt

Küçük bir Türkçe korpusla sıfırdan eğitilmiş, **~10M parametreli** GPT (öğrenme projesi).
Tek bir laptop GPU'sunda (RTX 4060 Laptop) eğitilebilir.

> **Durum:** Devam ediyor. Model akıcı Türkçe cümleler kuruyor ama gerçek bir sohbet asistanı **değil**. Bilgi sorularına çoğunlukla uydurma cevap verir.

## Model
| | |
|---|---|
| Mimari | decoder-only Transformer (pre-LN, GELU, ağırlık paylaşımı) |
| Parametre | ~10,1M (6 katman, 5 head, `n_embd=320`) |
| Bağlam | 512 token |
| Tokenizer | byte-level BPE, 8000 sözlük (Türkçe ekler apostroftan sonra bütün kalır) |
| Ön eğitim | ~71M token Türkçe forum metni, val kaybı **4,17** (perplexity ~64) |
| İnce ayar | küçük `[K] soru [A] cevap` diyalog seti |

## Dosyalar
| Dosya | Görevi |
|---|---|
| `clean_data.py` | Forum verisini temizler (küçük harf, URL/HTML/alıntı kalıntıları, tekilleştirme) |
| `clean_dialog.py` | Diyalog verisini sohbet tarzına göre filtreler |
| `train_tokenizer.py` | BPE tokenizer eğitir (`tokenizer.json`) |
| `prepare_data.py` | Metni token numaralarına çevirip `.bin` yazar |
| `model.py` | GPT modeli |
| `train.py` | Ön eğitim ve ince ayar |
| `generate.py` | Prompt'tan metin üretir |
| `chat.py` | Terminalde sohbet |

## Kullanım
Ham veriyi `forum/` ve `dialog/` klasörlerine koy (kaynaklar aşağıda), sonra:

```bash
pip install torch tokenizers numpy        # GPU için CUDA'lı PyTorch kur

python clean_data.py --raw . --out data/clean
python train_tokenizer.py --data data/clean --out tokenizer.json --vocab-size 8000
python prepare_data.py --data data/clean --out data/bin

python train.py                           # ön eğitim -> out/ckpt_best.pt
python train.py --data data/bin --train-bin dialog_train.bin --val-bin dialog_val.bin \
  --init-from out/ckpt_best.pt --out out_dialog --grad-accum 1 --max-iters 350 \
  --warmup 20 --lr 2e-4 --min-lr 2e-5 --eval-interval 50    # diyalog ince ayarı

python chat.py --ckpt out_dialog/ckpt_best.pt --tokenizer tokenizer.json
```

## Bilinen sınırlar
- 10M parametre bilgi tutmaya yetmez; cevaplar akıcı ama çoğu zaman anlamsız veya uydurma.
- Forum verisi teknik konulara (SEO, web) kayıyor ve bazı yazımlar aksansız (`gecmek`, `icin`).
- Küfür ve kişisel bilgi filtresi yok.
- Diyalog verisi küçük ve karışık kalitede.

## Veri kaynakları ve lisanslar
Veri bu repoda **yoktur**, kendin indirmelisin.

- [`turkish-nlp-suite/ForumSohbetleri`](https://huggingface.co/datasets/turkish-nlp-suite/ForumSohbetleri): CC-BY-SA-4.0 (kullanılan forumlar: forumum, iyinet, kadinlarklubu, turkiyeforum, wmaraci)
- [`SoAp9035/everyday-conversations-tur`](https://huggingface.co/datasets/SoAp9035/everyday-conversations-tur): Apache-2.0
- [`sixfingerdev/turkish-qa-multi-dialog-dataset`](https://huggingface.co/datasets/sixfingerdev/turkish-qa-multi-dialog-dataset): lisans belirtilmemiş

## Lisans
Kod: **MIT**. Model ağırlıkları henüz yayınlanmadı. Yayınlanırsa CC-BY-SA-4.0 verisiyle eğitildiği için ağırlıklar için CC-BY-SA-4.0 uygulanması planlanıyor.
