"""
run_39_conll_qa_vs_bio.py  —  Paper B experiment 6.22 (GPU; see PREREG_6.22_conll_tuned_bio.md)

CoNLL-2003 flat NER with two heads on the same encoder:
  --mode bio : first-subword token tagger (9 BIO labels), weighted CE (--o-weight on O), argmax AND BIO-valid
               Viterbi decodes scored from the same model.
  --mode mrc : Li et al. 2020-style MRC. One fixed query per entity type (PER/ORG/LOC/MISC), "query + sentence" as a
               text pair, two per-token start/end heads (2-way CE, --pos-weight on the positive class), flat-NER
               matching (each start paired with the nearest end at or after it). Simplified vs the paper: no
               start-end matching classifier; queries are ours, not theirs.
Metric: entity-level micro-F1, exact span + type, on the word-level tags (CoNLL is pre-tokenized, so the
SentencePiece trailing-punctuation artifact cannot occur here). Best epoch chosen on dev F1; test predictions
from that epoch are written. No checkpoints are saved (nothing to delete later).

    python run_39_conll_qa_vs_bio.py --mode bio --model bert-base-cased --seed 42 --o-weight 0.2 --out-dir RUN_DIR
    python run_39_conll_qa_vs_bio.py --selftest
Outputs in --out-dir: metrics.json, test_predictions.jsonl (gold + predicted entity lists per test sentence).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_33_absa_data_prep import get  # noqa: E402  (HF datasets-server fetch with 429/5xx backoff)

ID2TAG = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
TYPES = ["PER", "ORG", "LOC", "MISC"]
QUERY = {
    "PER": "find persons in the text including first names last names and nicknames",
    "ORG": "find organizations in the text including companies agencies sports teams and institutions",
    "LOC": "find locations in the text including countries cities regions and bodies of water",
    "MISC": "find miscellaneous entities in the text including nationalities events languages and other names",
}
ROWS = "https://datasets-server.huggingface.co/rows?dataset=tomaarsen/conll2003&config=default&split={}&offset={}&length=100"


# ── pure functions (selftested) ─────────────────────────────────────────────────

def tags_to_ents(tags):
    """conlleval-style: B-X starts an entity; I-X continues it, or starts one if the type changed / after O."""
    ents, cur = [], None
    for i, t in enumerate(tags):
        if t.startswith("B-") or (t.startswith("I-") and (cur is None or cur[2] != t[2:])):
            if cur:
                ents.append(tuple(cur))
            cur = [i, i, t[2:]]
        elif t.startswith("I-"):
            cur[1] = i
        else:
            if cur:
                ents.append(tuple(cur))
            cur = None
    if cur:
        ents.append(tuple(cur))
    return ents


def viterbi(logp):
    """Best BIO-valid tag path over [n, 9] log-probs: I-X only after B-X/I-X, no I- at position 0."""
    n, K = logp.shape
    if n == 0:
        return []
    allow = np.ones((K, K), bool)
    for cur, tc in enumerate(ID2TAG):
        if tc.startswith("I-"):
            for prev, tp in enumerate(ID2TAG):
                allow[prev, cur] = tp != "O" and tp[2:] == tc[2:]
    trans = np.where(allow, 0.0, -1e9)
    score = np.where([not t.startswith("I-") for t in ID2TAG], logp[0], -1e9)
    back = np.zeros((n, K), int)
    for i in range(1, n):
        cand = score[:, None] + trans
        back[i], score = cand.argmax(0), cand.max(0) + logp[i]
    path = [int(score.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    return path[::-1]


def match_flat(starts, ends):
    """Each start paired with the nearest end at or after it (correct for flat, non-overlapping entities)."""
    out = set()
    for s in sorted(starts):
        e = [x for x in sorted(ends) if x >= s]
        if e:
            out.add((s, e[0]))
    return out


def micro_f1(gold, pred):
    """gold/pred: list (per sentence) of sets of (s, e, type)."""
    tp = sum(len(g & p) for g, p in zip(gold, pred))
    fp = sum(len(p - g) for g, p in zip(gold, pred))
    fn = sum(len(g - p) for g, p in zip(gold, pred))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def selftest():
    assert tags_to_ents(["B-PER", "I-PER", "O", "I-LOC", "B-ORG"]) == [(0, 1, "PER"), (3, 3, "LOC"), (4, 4, "ORG")]
    assert [ID2TAG[i] for i in viterbi(np.log(np.array([[.1, .8, .02, .02, .02, .01, .01, .01, .01], [.4, .01, .5, .02, .02, .02, .01, .01, .01]])))] == ["B-PER", "I-PER"]
    # an unconstrained argmax would give I-LOC first (invalid); viterbi must not start with I-
    bad = np.log(np.array([[.1, .05, .05, .05, .05, .05, .55, .05, .05]]))
    assert not ID2TAG[viterbi(bad)[0]].startswith("I-")
    assert match_flat([0, 5], [2, 7]) == {(0, 2), (5, 7)} and match_flat([3], [1]) == set()
    g = [{(0, 1, "PER")}, {(2, 2, "LOC")}]
    assert abs(micro_f1(g, [{(0, 1, "PER")}, {(2, 2, "ORG")}]) - 0.5) < 1e-9
    assert micro_f1(g, g) == 1.0
    print("selftest ok")


# ── data ────────────────────────────────────────────────────────────────────────

def load_split(split, data_dir, max_n=None):
    f = Path(data_dir) / f"{split}.jsonl"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        rows, off = [], 0
        while True:
            page = json.loads(get(ROWS.format("validation" if split == "dev" else split, off)))
            rows += [{"tokens": x["row"]["tokens"], "tags": [ID2TAG[t] for t in x["row"]["ner_tags"]]} for x in page["rows"]]
            off += 100
            if off >= page["num_rows_total"]:
                break
        f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rows = [json.loads(l) for l in open(f)]
    return rows[:max_n] if max_n else rows


# ── model / training (imports torch lazily so --selftest runs anywhere) ───────

def run(a):
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"add_prefix_space": True} if "roberta" in a.model and "xlm" not in a.model else {}
    tok = AutoTokenizer.from_pretrained(a.model, **kw)
    train, dv, test = (load_split(s, a.data_dir, a.max_n) for s in ("train", "dev", "test"))
    use_tt = "token_type_ids" in tok.model_input_names

    def enc_bio(words):
        e = tok(words, is_split_into_words=True, truncation=True, max_length=a.max_len)
        wid, pos, widx, prev = e.word_ids(), [], [], None
        for p, w in enumerate(wid):
            if w is not None and w != prev:
                pos.append(p); widx.append(w)
            prev = w
        return e["input_ids"], pos, widx, e.get("token_type_ids")

    def enc_mrc(words, typ):
        e = tok(QUERY[typ].split(), words, is_split_into_words=True, truncation="only_second", max_length=a.max_len)
        wid, sid, pos, widx, prev = e.word_ids(), e.sequence_ids(), [], [], None
        for p, (w, s) in enumerate(zip(wid, sid)):
            if s == 1 and w != prev:
                pos.append(p); widx.append(w)
            prev = w if s == 1 else None
        return e["input_ids"], pos, widx, e.get("token_type_ids")

    def build(rows, with_labels):
        items = []
        for si, r in enumerate(rows):
            gold = tags_to_ents(r["tags"])
            if a.mode == "bio":
                ids, pos, widx, tt = enc_bio(r["tokens"])
                lab = [ID2TAG.index(r["tags"][w]) for w in widx]
                items.append(dict(s=si, ids=ids, pos=pos, widx=widx, tt=tt, lab=lab, typ=None, n=len(r["tokens"])))
            else:
                for typ in TYPES:
                    ids, pos, widx, tt = enc_mrc(r["tokens"], typ)
                    st = {e[0] for e in gold if e[2] == typ}; en = {e[1] for e in gold if e[2] == typ}
                    lab = [(int(w in st), int(w in en)) for w in widx]
                    items.append(dict(s=si, ids=ids, pos=pos, widx=widx, tt=tt, lab=lab, typ=typ, n=len(r["tokens"])))
        return items

    class Net(nn.Module):
        def __init__(s, out):
            super().__init__()
            s.enc = AutoModel.from_pretrained(a.model); s.drop = nn.Dropout(0.1)
            s.head = nn.Linear(s.enc.config.hidden_size, out)

        def forward(s, ids, mask, tt, pos):
            kwargs = {"token_type_ids": tt} if (tt is not None and s.enc.config.type_vocab_size > 1) else {}
            h = s.enc(input_ids=ids, attention_mask=mask, **kwargs).last_hidden_state
            lg = s.head(s.drop(h))
            return lg.gather(1, pos.unsqueeze(-1).expand(-1, -1, lg.size(-1)))

    out_dim = 9 if a.mode == "bio" else 4
    net = Net(out_dim).to(dev)
    pad = tok.pad_token_id

    def collate(b):
        L, W = max(len(x["ids"]) for x in b), max(max(len(x["pos"]) for x in b), 1)
        ids = torch.full((len(b), L), pad); mask = torch.zeros((len(b), L), dtype=torch.long)
        tt = torch.zeros((len(b), L), dtype=torch.long) if use_tt else None
        pos = torch.zeros((len(b), W), dtype=torch.long)
        lab = torch.full((len(b), W) if a.mode == "bio" else (len(b), W, 2), -100)
        for i, x in enumerate(b):
            n = len(x["ids"]); ids[i, :n] = torch.tensor(x["ids"]); mask[i, :n] = 1
            if use_tt and x["tt"] is not None:
                tt[i, :n] = torch.tensor(x["tt"])
            k = len(x["pos"])
            if k:
                pos[i, :k] = torch.tensor(x["pos"]); lab[i, :k] = torch.tensor(x["lab"])
        return ids.to(dev), mask.to(dev), (tt.to(dev) if use_tt else None), pos.to(dev), lab.to(dev)

    w_bio = torch.tensor([a.o_weight] + [1.0] * 8, device=dev)
    w_mrc = torch.tensor([1.0, a.pos_weight], device=dev)

    def loss_fn(lg, lab):
        if a.mode == "bio":
            return nn.functional.cross_entropy(lg.reshape(-1, 9), lab.reshape(-1), weight=w_bio, ignore_index=-100)
        ls = nn.functional.cross_entropy(lg[..., :2].reshape(-1, 2), lab[..., 0].reshape(-1), weight=w_mrc, ignore_index=-100)
        le = nn.functional.cross_entropy(lg[..., 2:].reshape(-1, 2), lab[..., 1].reshape(-1), weight=w_mrc, ignore_index=-100)
        return ls + le

    def predict(items, rows):
        net.eval(); res = {}
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
        with torch.no_grad(), amp:
            for i in range(0, len(items), 64):
                b = items[i:i + 64]; ids, mask, tt, pos, _ = collate(b)
                lg = net(ids, mask, tt, pos).float().cpu()
                for x, l in zip(b, lg):
                    if x["pos"]:
                        res.setdefault(x["s"], []).append((x, l[:len(x["pos"])]))
        arg, vit = [], []
        for si, r in enumerate(rows):
            n = len(r["tokens"])
            if a.mode == "bio":
                x, l = res[si][0] if si in res else ({"widx": []}, torch.zeros(0, 9))
                lp = torch.log_softmax(l, -1).numpy()
                ta, tv = ["O"] * n, ["O"] * n
                for w, k in zip(x["widx"], lp.argmax(-1)):
                    ta[w] = ID2TAG[k]
                for w, k in zip(x["widx"], viterbi(lp) if len(lp) else []):
                    tv[w] = ID2TAG[k]
                arg.append(set(tags_to_ents(ta))); vit.append(set(tags_to_ents(tv)))
            else:
                ents = set()
                for x, l in res.get(si, []):
                    p = torch.softmax(l.view(-1, 2, 2), -1)[..., 1].numpy()          # [words, (start,end)]
                    st = {w for w, v in zip(x["widx"], p[:, 0]) if v > 0.5}; en = {w for w, v in zip(x["widx"], p[:, 1]) if v > 0.5}
                    ents |= {(s, e, x["typ"]) for s, e in match_flat(st, en)}
                arg.append(ents)
        gold = [set(tags_to_ents(r["tags"])) for r in rows]
        return gold, arg, vit

    tr_items = build(train, True); dv_items = build(dv, True); te_items = build(test, True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(tr_items) + a.batch - 1) // a.batch)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    g = random.Random(a.seed); best, t0 = None, time.time()
    for ep in range(1, a.epochs + 1):
        net.train(); order = list(range(len(tr_items))); g.shuffle(order)
        for i in range(0, len(order), a.batch):
            ids, mask, tt, pos, lab = collate([tr_items[j] for j in order[i:i + a.batch]])
            with amp:
                lg = net(ids, mask, tt, pos)
            loss = loss_fn(lg.float(), lab)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
        gd, ad, vd = predict(dv_items, dv)
        f_arg, f_vit = micro_f1(gd, ad), (micro_f1(gd, vd) if vd else None)
        print(f"epoch {ep}: loss {loss.item():.4f} dev F1 {f_arg:.4f}" + (f" (viterbi {f_vit:.4f})" if f_vit is not None else "") + f"  [{time.time()-t0:.0f}s]", flush=True)
        if best is None or f_arg > best["dev_f1"]:
            gt, at, vt = predict(te_items, test)
            best = dict(epoch=ep, dev_f1=f_arg, dev_f1_viterbi=f_vit, test_f1=micro_f1(gt, at),
                        test_f1_viterbi=(micro_f1(gt, vt) if vt else None), gold=gt, arg=at, vit=vt)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "test_predictions.jsonl", "w") as f:
        for i in range(len(test)):
            row = {"i": i, "gold": sorted(best["gold"][i]), "pred": sorted(best["arg"][i])}
            if best["vit"]:
                row["pred_viterbi"] = sorted(best["vit"][i])
            f.write(json.dumps(row) + "\n")
    m = dict(mode=a.mode, model=a.model, seed=a.seed, lr=a.lr, epochs=a.epochs, o_weight=a.o_weight, pos_weight=a.pos_weight,
             best_epoch=best["epoch"], dev_f1=best["dev_f1"], dev_f1_viterbi=best["dev_f1_viterbi"], test_f1=best["test_f1"],
             test_f1_viterbi=best["test_f1_viterbi"], n_train=len(train), n_dev=len(dv), n_test=len(test), seconds=round(time.time() - t0))
    json.dump(m, open(out / "metrics.json", "w"), indent=1)
    print(json.dumps(m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bio", "mrc"]); ap.add_argument("--model", default="bert-base-cased")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--o-weight", type=float, default=1.0); ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--max-n", type=int, help="cap sentences per split (dry-run timing only)")
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[2] / "data" / "conll03"))
    ap.add_argument("--out-dir"); ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.fetch_only:
        print({s: len(load_split(s, a.data_dir)) for s in ("train", "dev", "test")}); return
    if not (a.mode and a.out_dir):
        ap.error("--mode and --out-dir are required")
    if (Path(a.out_dir) / "metrics.json").exists() and not a.force:
        print(f"skip (metrics.json exists in {a.out_dir}; pass --force to overwrite)"); return
    run(a)


if __name__ == "__main__":
    main()
