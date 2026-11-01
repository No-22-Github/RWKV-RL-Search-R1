"""拒绝采样自蒸馏（RAFT）打包：策略模型每题采 n 次 → 每题留答对且最省的轨迹 → MiSS segments，可和教师包拼接。

输入是 eval_base.py 的 samples.jsonl（榜单协议环境，--n 8 --temperature 1.0）。每条样本的
response = 生成1 + 检索1 + 生成2 + ... + 生成N，turns[i]["kept"] 是第 i 轮生成，按位置切开：
生成段 kept + "\\n\\nUser:" 训练，检索段（去掉开头的 "\\n\\nUser:"）不训练，和 distill_teacher.render 一致。

每题的筛选：
- 只要答对、end=answer、没有坏调用、没有被截断的轮次、没有完全相同的重复检索；
- 按（检索次数，生成 token 数）取最省的一条；组内通过率 ≤ --hard_rate 的难题再多留一条不同检索次数/最短的；
- 总长超过 --max_tokens 的丢掉（训练 ctx）；
- 组内全对的简单题只按 --easy_keep 的比例留（题池里随机新题多时，别让简单题占满数据）。

用法：python scripts/pack_raft.py outputs/raft/<run>/samples.jsonl --teacher segments-sft2.jsonl --out outputs/raft/<pack>
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

try:
    import rwkv
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    VOCAB_DIR = os.path.dirname(rwkv.__file__)
except ImportError:
    from rwkv_tokenizer import TRIE_TOKENIZER
    VOCAB_DIR = os.path.dirname(os.path.abspath(__file__))

SEP = "\n\nUser:"


def segments(s: dict) -> list[dict] | None:
    segs = [{"text": s["prompt"], "train": False}]
    rest = s["response"]
    for i, t in enumerate(s["turns"]):
        k = t["kept"]
        if not rest.startswith(k):
            return None
        rest = rest[len(k):]
        segs.append({"text": k + SEP, "train": True})
        if i + 1 < len(s["turns"]):
            nxt = s["turns"][i + 1]["kept"]
            j = rest.find(nxt)
            if j <= 0 or not rest.startswith(SEP):
                return None
            segs.append({"text": rest[len(SEP):j], "train": False})
            rest = rest[j:]
    return segs if rest == "" else None


def reject(s: dict) -> str | None:
    if not s["correct"]:
        return "wrong"
    if s["end"] != "answer" or not s["format_ok"]:
        return "end_" + str(s["end"])
    if s["bad_call"]:
        return "bad_call"
    if any(t["end"] == "length" or t.get("finish_reason") == "length" for t in s["turns"]):
        return "length"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("samples")
    p.add_argument("--teacher", default=None, help="教师 segments jsonl（SFT v2 训练包），拼在前面")
    p.add_argument("--out", required=True)
    p.add_argument("--max_tokens", type=int, default=6144)
    p.add_argument("--hard_rate", type=float, default=0.375, help="通过率 ≤ 它的题多留一条")
    p.add_argument("--easy_keep", type=float, default=1.0, help="全对的题保留比例")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    tok = TRIE_TOKENIZER(os.path.join(VOCAB_DIR, "rwkv_vocab_v20230424.txt"))

    groups = defaultdict(list)
    for line in open(a.samples):
        s = json.loads(line)
        groups[(str(s["qid"]), s["source"])].append(s)

    reasons, rows, meta = Counter(), [], []
    rng = random.Random(a.seed)
    for key, g in sorted(groups.items()):
        rate = sum(s["correct"] for s in g) / len(g)
        if rate == 1.0 and rng.random() >= a.easy_keep:
            reasons["q_easy_dropped"] += 1
            continue
        ok = []
        for s in g:
            why = reject(s)
            if why is None:
                segs = segments(s)
                if segs is None:
                    why = "split_fail"
            if why is None:
                n_tok = len(tok.encode("".join(x["text"] for x in segs)))
                gen = len(tok.encode("".join(x["text"] for x in segs if x["train"])))
                if n_tok > a.max_tokens:
                    why = "too_long"
            reasons[why or "ok"] += 1
            if why is None:
                ok.append((s["n_search"], gen, n_tok, segs, s))
        if not ok:
            reasons["q_no_keep"] += 1
            continue
        ok.sort(key=lambda x: (x[0], x[1]))
        keep = [ok[0]]
        if rate <= a.hard_rate and len(ok) > 1:
            keep.append(next((x for x in ok[1:] if x[0] != ok[0][0]), ok[1]))
        for n_search, gen, n_tok, segs, s in keep:
            rows.append({"segments": segs})
            meta.append({"qid": key[0], "source": key[1], "rate": rate, "n_search": n_search,
                         "gen_tokens": gen, "tokens": n_tok, "kind": "raft"})

    n_raft = len(rows)
    if a.teacher:
        for line in open(a.teacher):
            rows.append(json.loads(line))
            meta.append({"kind": "teacher"})
    order = list(range(len(rows)))
    random.Random(a.seed).shuffle(order)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train.jsonl"), "w") as f, open(os.path.join(a.out, "train.meta.jsonl"), "w") as fm:
        for i in order:
            f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
            fm.write(json.dumps(meta[i], ensure_ascii=False) + "\n")
    raft = [m for m in meta if m["kind"] == "raft"]
    sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest = {
        "inputs": [{"path": x, "sha256": sha(x)} for x in [a.samples] + ([a.teacher] if a.teacher else [])],
        "args": vars(a), "filter": dict(reasons.most_common()), "questions": len(groups),
        "rows": {"raft": n_raft, "teacher": len(rows) - n_raft},
        "raft_n_search": dict(sorted(Counter(m["n_search"] for m in raft).items())),
        "raft_by_source": dict(Counter(m["source"] for m in raft)),
        "raft_max_tokens": max((m["tokens"] for m in raft), default=0),
        "train_sha256": sha(os.path.join(a.out, "train.jsonl")),
    }
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: manifest[k] for k in ("filter", "questions", "rows", "raft_n_search", "raft_by_source",
                                                "raft_max_tokens")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
