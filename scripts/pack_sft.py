"""把 distill_teacher.py 的轨迹打成 MiSS 训练包：train.jsonl / validation.jsonl / manifest.json。

过滤和渲染与 distill_teacher.py 相同；验证集按 data_source 分层抽取，按题目切分。
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from types import SimpleNamespace

import rwkv
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

from distill_teacher import keep, render
from eval_base import G1KFormat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val_per_source", type=int, default=20)
    p.add_argument("--min_search", type=int, default=1)
    p.add_argument("--max_think_chars", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    fmt = G1KFormat("full")
    tok = TRIE_TOKENIZER(os.path.join(os.path.dirname(rwkv.__file__), "rwkv_vocab_v20230424.txt"))
    filt = SimpleNamespace(min_search=args.min_search, max_think_chars=args.max_think_chars)

    kept, seen = [], set()
    for path in args.trajectories:
        for line in open(path):
            t = json.loads(line)
            if not t["steps"] or t["qid"] in seen or keep(t, filt) is not None:
                continue
            seen.add(t["qid"])
            kept.append(t)
    rng = random.Random(args.seed)
    rng.shuffle(kept)

    val_ids = set()
    for src in sorted({t["source"] for t in kept}):
        val_ids |= {t["qid"] for t in [t for t in kept if t["source"] == src][: args.val_per_source]}

    os.makedirs(args.out, exist_ok=True)
    manifest = {"inputs": [{"path": p, "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()} for p in args.trajectories],
                "filter": vars(filt), "splits": {}}
    for split in ("train", "validation"):
        rows = [t for t in kept if (t["qid"] in val_ids) == (split == "validation")]
        path = os.path.join(args.out, f"{split}.jsonl")
        lens = []
        # rwkv_miss_tune 的解析器只接受 {"segments": [...]} 一个字段，meta 按行号另存
        with open(path, "w") as f, open(os.path.join(args.out, f"{split}.meta.jsonl"), "w") as fm:
            for t in rows:
                segs = render(t, fmt)
                lens.append(len(tok.encode("".join(s["text"] for s in segs))))
                f.write(json.dumps({"segments": segs}, ensure_ascii=False) + "\n")
                fm.write(json.dumps({"qid": t["qid"], "source": t["source"], "n_search": t["n_search"]}) + "\n")
        lens.sort()
        manifest["splits"][split] = {
            "path": path, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "rows": len(rows),
            "by_source": dict(Counter(t["source"] for t in rows)),
            "by_n_search": dict(sorted(Counter(t["n_search"] for t in rows).items())),
            "tokens": {"p50": lens[len(lens) // 2], "p99": lens[int(len(lens) * 0.99)], "max": lens[-1]} if lens else None,
        }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
