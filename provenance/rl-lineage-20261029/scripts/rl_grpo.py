"""Search-R1 多轮检索 GRPO 客户端，对接 rwkv_miss_tune_server（在线 RL 服务）。

服务端用 Alic-Li/rwkv_lightning_cuda main（b79aca6 起）的 CI 产物：
合批采样、字符串 stop、/v1/logprobs 参考策略打分、checkpoint 发布；
rollout 和训练默认走 BF16 Tensor Core（--gemm-mode bf16，比 strict 快约一个数量级）。
注意字符串 stop 与 stop_token_ids/stop_token_sequences 互斥（同时发返回 400），
本客户端只发字符串 stop；截断的轮次由 finish_reason=length 反映，cut_turn 归为 end=length。

每一步：
1. 取 questions 道题，每题 group 次采样，所有轨迹放进同一批多轮 rollout：
   server 合批生成一轮 → 本地截出动作 → BM25 检索 → 把检索结果接到历史后面，最多 max_turns 轮。
2. 奖励：EM 答对 1；格式正确但答错 r_wrong；没给出合法答案（max_turns/截断/越界/坏调用）r_invalid。
3. 组内归一化得到优势（GRPO），可选按 token 减去 kl_coef·(logp - logp_ref)。
4. 一次 /v1/train（一次 Adam），再 /v1/publish 换成新策略。

token 级拼接：生成部分直接用 server 返回的 token id（loss_mask=1），prompt 和检索结果
用 server 的 /v1/tokenize 编码（loss_mask=0），整条历史从不重新编码，训练的 token
和采样时完全一致。格式（System 提示、<tool_call>、<tool_response>、最后一轮提示）
复用 eval_base.G1KFormat，与 SFT 数据和评测一致。

服务端（在解压后的 rwkv_lighting_cuda 目录）：
  ./rwkv_miss_tune_server --model rwkv7-g1k-2.9b-20260930-ctx25600.pth \
    --adapter rl-sr1-v1-s100.pth \
    --vocab ./rwkv_vocab_v20230424.txt --ctx 8192 --chunk 128 --max-batch 256 \
    --cache-capacity 640 --keep-versions 2 --lr 2e-6 --output rl-out/server
  --cache-capacity 至少约 2×每批轨迹数：每条轨迹每轮会存“prompt”和“生成完”两个前缀，
  默认 16 会让多轮检索每轮都从头 prefill。--max-batch 要不小于每步轨迹数（题数×组大小）。
客户端（检索服务要先起：uv run python retriever/bm25_server.py）：
  uv run python scripts/rl_grpo.py --server http://127.0.0.1:8848 --out outputs/rl/sr1-v2
建议先用 scripts/filter_questions.py 筛出组内有对有错的题，--data 指向筛出的 parquet。
中断后重跑同一条命令即可：启动时若 learner 已训练未发布会先补发布，
数据位置从 metrics.jsonl 的最后一步接着走（Adam 状态只有 checkpoint 发布过才能 --resume 恢复）。
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_base import G1KFormat, cut_turn, em, extract_answers, normalize_answer, to_docs, truncate_docs  # noqa: E402

STOPS = ["</tool_call>", "</tool_calls>", "</answer>", "\n\nUser:"]


class Server:
    def __init__(self, url: str, timeout: float):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.sess = requests.Session()

    def get(self, path: str) -> dict:
        r = self.sess.get(self.url + path, timeout=60)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: dict) -> dict:
        r = self.sess.post(self.url + path, json=body, timeout=(10, self.timeout))
        if r.status_code != 200:
            raise RuntimeError(f"{path} {r.status_code}: {r.text[:500]}")
        return r.json()

    def tokenize(self, text: str) -> list[int]:
        return self.post("/v1/tokenize", {"text": text})["tokens"]


def f1(pred: str | None, golds) -> float:
    if not pred:
        return 0.0
    p = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        g = normalize_answer(g).split()
        common = sum(min(p.count(w), g.count(w)) for w in set(p))
        if common:
            pr, rc = common / len(p), common / len(g)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


def new_rows(df: pd.DataFrame, fmt: G1KFormat, srv: Server, group: int) -> list[dict]:
    rows = []
    for gi, (_, q) in enumerate(df.iterrows()):
        prompt = srv.tokenize(fmt.prompt(q))
        for k in range(group):
            rows.append({
                "group": gi, "qid": q.id, "source": q.data_source, "question": q.question,
                "golden": list(q.golden_answers), "sample": k,
                # turns 就是发给 /v1/train 的分段；hist 是它们拼起来的完整 token 历史
                "turns": [{"tokens": list(prompt), "loss_mask": [0] * len(prompt)}],
                "hist": list(prompt), "gen_pos": [], "logprobs": [], "kept": [],
                "end": None, "n_search": 0, "bad_call": 0, "gen_tokens": 0,
            })
    return rows


def rollout(rows: list[dict], fmt: G1KFormat, srv: Server, version: str, a, seed: int) -> dict:
    stats = {"gen_seconds": 0.0, "retrieve_seconds": 0.0, "generated_tokens": 0, "cached_prompt_tokens": 0,
             "prefill_tokens": 0, "turn_rows": []}
    active = list(range(len(rows)))
    for turn in range(a.max_turns):
        # 历史太长、放不下下一轮生成的直接结束（按无效答案算）
        fits = []
        for i in active:
            if len(rows[i]["hist"]) + a.max_turn_tokens > a.ctx:
                rows[i]["end"] = "overflow"
            else:
                fits.append(i)
        active = fits
        if not active:
            break
        stats["turn_rows"].append(len(active))
        t = time.time()
        outs = []
        for c in range(0, len(active), a.max_batch):
            chunk = active[c : c + a.max_batch]
            res = srv.post("/v1/completions", {
                "policy_version": version, "prompts": [rows[i]["hist"] for i in chunk],
                "max_tokens": a.max_turn_tokens, "temperature": a.temperature,
                "stop": STOPS, "seed": seed * 1000 + turn * 100 + c, "use_cache": True,
            })
            if res["policy_version"] != version:
                raise RuntimeError(f"rollout policy {res['policy_version']} != {version}")
            outs += res["completions"]
            for k in ("generated_tokens", "cached_prompt_tokens", "prefill_tokens"):
                stats[k] += res.get(k, 0)
        stats["gen_seconds"] += time.time() - t

        to_search = []
        for i, out in zip(active, outs):
            r = rows[i]
            toks = out["tokens"]
            r["gen_pos"] += range(len(r["hist"]), len(r["hist"]) + len(toks))
            r["turns"].append({"tokens": toks, "loss_mask": [1] * len(toks)})
            r["hist"] += toks
            r["logprobs"] += out["logprobs"]
            r["gen_tokens"] += len(toks)
            text, end = cut_turn(out["text"], fmt)
            r["kept"].append({"text": out["text"], "kept": text, "end": end,
                              "finish_reason": out["finish_reason"], "matched_stop": out.get("matched_stop")})
            if end == "search" and turn < a.max_turns - 1:
                q, _ = fmt.parse_call(text)
                r["n_search"] += 1
                r["bad_call"] += int(q is None)
                to_search.append((i, q or ""))
            else:
                r["end"] = "max_turns" if end == "search" else end

        if to_search:
            t = time.time()
            resp = requests.post(f"{a.retriever}/retrieve", json={"queries": [q for _, q in to_search], "topk": a.topk},
                                 timeout=600)
            resp.raise_for_status()
            last = turn + 1 == a.max_turns - 1
            for (i, _), docs in zip(to_search, resp.json()["result"]):
                docs = to_docs(docs)
                if getattr(a, "max_obs_tokens", 0):
                    docs = truncate_docs(docs, a.max_obs_tokens)
                obs = srv.tokenize(fmt.observation(docs, last))
                rows[i]["turns"].append({"tokens": obs, "loss_mask": [0] * len(obs)})
                rows[i]["hist"] += obs
            stats["retrieve_seconds"] += time.time() - t
        active = [i for i, _ in to_search]
    for i in active:
        rows[i]["end"] = rows[i]["end"] or "max_turns"
    return stats


def score(rows: list[dict], a) -> None:
    for r in rows:
        answers = extract_answers(r["kept"])
        r["pred"] = answers[-1].strip() if answers else None
        r["format_ok"] = r["end"] == "answer" and len(answers) == 1 and r["bad_call"] == 0
        r["correct"] = em(r["pred"], r["golden"])
        if r["correct"]:
            r["reward"] = 1.0
        elif r["format_ok"]:
            r["reward"] = a.r_wrong + a.f1_weight * f1(r["pred"], r["golden"])
        else:
            r["reward"] = a.r_invalid


def advantages(rows: list[dict], a) -> None:
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    for g in groups.values():
        rs = np.array([r["reward"] for r in g])
        std = rs.std()
        for r, x in zip(g, rs):
            if std < 1e-6:
                r["adv"] = 0.0
            else:
                r["adv"] = float((x - rs.mean()) / (std + 1e-6) if a.std_norm else x - rs.mean())


def kl_shaping(rows: list[dict], srv: Server, ref_version: str, a) -> float:
    """per-token 优势 = A - kl_coef·(logp - logp_ref)，logp 用采样时返回的值。返回平均 k1 KL。"""
    todo = [r for r in rows if r["adv"] != 0.0 or not a.drop_zero_adv]
    kls = []
    for c in range(0, len(todo), a.max_batch):
        chunk = todo[c : c + a.max_batch]
        res = srv.post("/v1/logprobs", {"policy_version": ref_version, "sequences": [r["hist"] for r in chunk],
                                        "temperature": a.temperature})
        for r, ref in zip(chunk, res["logprobs"]):
            # ref[t-1] 是 hist[t] 在参考策略下的 logprob
            r["kl"] = [lp - ref[p - 1] for p, lp in zip(r["gen_pos"], r["logprobs"])]
            kls += r["kl"]
    return float(np.mean(kls)) if kls else 0.0


def scale_adv(rows: list[dict], a) -> None:
    """--neg_scale 缩小负优势；--seq_norm 按生成长度把每条轨迹的总权重拉平。
    服务端 loss 按全批 token 数归一化，错的轨迹更长，负梯度的 token 占多数会推高熵、越写越长。"""
    live = [r for r in rows if r["adv"] != 0.0 and r["gen_pos"]]
    if not live:
        return
    mean_len = float(np.mean([len(r["gen_pos"]) for r in live]))
    for r in live:
        if r["adv"] < 0:
            r["adv"] *= a.neg_scale
        if a.seq_norm:
            r["adv"] *= mean_len / len(r["gen_pos"])


def train_samples(rows: list[dict], a) -> list[dict]:
    samples = []
    for r in rows:
        if r["adv"] == 0.0 and a.drop_zero_adv:
            continue
        if not r["gen_pos"]:
            continue
        # 末尾不带梯度的分段（overflow 轨迹最后接上的检索结果）去掉不影响 loss，
        # 但会让历史超出服务端 ctx 直接 400；rollout 已保证生成部分都在 ctx 内
        turns = list(r["turns"])
        while not any(turns[-1]["loss_mask"]):
            turns.pop()
        n_tok = sum(len(t["tokens"]) for t in turns)
        if n_tok > a.ctx:
            print(f"  skip sample {r['qid']}/{r['sample']}: {n_tok} tokens > ctx {a.ctx}", flush=True)
            continue
        if "kl" in r:
            per_tok = iter([r["adv"] - a.kl_coef * k for k in r["kl"]])
            turns = [{**t, "advantages": [next(per_tok) if m else 0.0 for m in t["loss_mask"]]} for t in turns]
            samples.append({"turns": turns, "advantages": 0.0})
        else:
            samples.append({"turns": turns, "advantages": r["adv"]})
    return samples


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r["correct"])
    ends = {}
    for r in rows:
        ends[r["end"]] = ends.get(r["end"], 0) + 1
    return {
        "reward": round(float(np.mean([r["reward"] for r in rows])), 4),
        "acc": round(sum(r["correct"] for r in rows) / n, 4),
        "format_ok": round(sum(r["format_ok"] for r in rows) / n, 4),
        "pass@group": round(sum(any(g) for g in groups.values()) / len(groups), 4),
        "mixed_groups": sum(0 < sum(g) < len(g) for g in groups.values()),
        "nonzero_adv": sum(r["adv"] != 0.0 for r in rows),
        "avg_search": round(float(np.mean([r["n_search"] for r in rows])), 3),
        "avg_gen_tokens": round(float(np.mean([r["gen_tokens"] for r in rows])), 1),
        "max_len": max(len(r["hist"]) for r in rows),
        "end": ends,
    }


def last_step(path: str) -> int:
    if not os.path.exists(path):
        return 0
    step = 0
    with open(path) as f:
        for line in f:
            step = max(step, json.loads(line)["step"] + 1)
    return step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True, help="rwkv_miss_tune_server 地址，如 $TUNE_SERVER 或 http://127.0.0.1:8848")
    p.add_argument("--retriever", default="http://127.0.0.1:8000")
    p.add_argument("--data", default="data/search_r1/train.parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=200, help="总步数（含已完成的）")
    p.add_argument("--questions", type=int, default=16, help="每步题数")
    p.add_argument("--group", type=int, default=8, help="每题采样数（GRPO 组大小）")
    p.add_argument("--max_turns", type=int, default=4)
    p.add_argument("--max_turn_tokens", type=int, default=1024)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--max_obs_tokens", type=int, default=0, help="每次检索结果截到前 N 个 token（榜单协议 500），0 不截")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="前 N 步学习率线性升到 --lr（冷启动的 Adam 头几步约等于 lr·sign(g)，更新最猛）")
    p.add_argument("--abort_fmt", type=float, default=0.0,
                   help=">0 时格式正确率连续 2 步低于它就停训（探针用，崩了不浪费时间）")
    p.add_argument("--r_wrong", type=float, default=0.0, help="格式正确但答错")
    p.add_argument("--r_invalid", type=float, default=-0.2, help="没有合法的最终答案")
    p.add_argument("--f1_weight", type=float, default=0.0, help="答错时按 F1 给部分分")
    p.add_argument("--seq_norm", action="store_true", help="优势乘 批内平均生成长度/本条长度，每条轨迹总权重相等")
    p.add_argument("--neg_scale", type=float, default=1.0, help="负优势乘这个系数（<1 抑制熵上涨）")
    p.add_argument("--no_std_norm", dest="std_norm", action="store_false", help="优势只减组均值，不除标准差")
    p.add_argument("--keep_zero_adv", dest="drop_zero_adv", action="store_false",
                   help="组内无差异的轨迹也送去训练（只增加 loss 分母，不贡献梯度）")
    p.add_argument("--kl_coef", type=float, default=0.0, help=">0 时用 /v1/logprobs 对参考策略做逐 token KL 惩罚")
    p.add_argument("--save_every", type=int, default=10, help="每 N 步发布时导出 adapter.pth")
    p.add_argument("--checkpoint_every", type=int, default=50, help="每 N 步导出含 Adam 状态的完整检查点")
    p.add_argument("--dump_every", type=int, default=5, help="每 N 步保存完整轨迹")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--timeout", type=float, default=3600)
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    metrics_path = os.path.join(a.out, "metrics.jsonl")
    srv = Server(a.server, a.timeout)
    health = srv.get("/health")
    a.ctx, a.max_batch = health["ctx"], health["max_batch"]
    if a.questions * a.group > a.max_batch:
        sys.exit(f"questions*group={a.questions * a.group} 超过服务端 --max-batch {a.max_batch}，一步只能一次 train")
    if a.max_turn_tokens >= a.ctx:
        sys.exit(f"max_turn_tokens {a.max_turn_tokens} >= ctx {a.ctx}")
    if health["step"] > health["published_step"]:
        # 上次在 train 之后、publish 之前断了：先把这次更新发布出来，不要重训
        print(f"learner step {health['step']} > published {health['published_step']}，补发布", flush=True)
        srv.post("/v1/publish", {"expected_step": health["step"], "save": True})
        health = srv.get("/health")
    version, ref_version = health["policy_version"], health["reference_version"]
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump({"args": vars(a), "health": health}, f, ensure_ascii=False, indent=2)

    df = pd.read_parquet(a.data)
    order = np.random.default_rng(a.seed).permutation(len(df))
    fmt = G1KFormat("full")
    start = last_step(metrics_path)
    print(f"start at step {start}, learner step {health['step']}, policy {version}", flush=True)

    low_fmt = 0
    for step in range(start, a.steps):
        t0 = time.time()
        idx = order[(step * a.questions) % len(df) :][: a.questions]
        rows = new_rows(df.iloc[idx], fmt, srv, a.group)
        rs = rollout(rows, fmt, srv, version, a, a.seed + step)
        score(rows, a)
        advantages(rows, a)
        scale_adv(rows, a)
        m = {"step": step, "policy_version": version, **summarize(rows), "rollout": {
            k: (round(v, 1) if isinstance(v, float) else v) for k, v in rs.items()}}
        if a.kl_coef > 0:
            m["kl"] = round(kl_shaping(rows, srv, ref_version, a), 5)
        samples = train_samples(rows, a)
        t1 = time.time()
        if samples:
            lr = a.lr * min(1.0, (step + 1) / a.warmup_steps) if a.warmup_steps else a.lr
            res = srv.post("/v1/train", {"policy_version": version, "temperature": a.temperature,
                                         "learning_rate": lr, "samples": samples})
            m["train"] = {k: res.get(k) for k in ("loss", "grad_norm", "learning_rate", "step", "trained_tokens",
                                                  "updated", "seconds")}
            if res.get("updated"):
                save = (step + 1) % a.save_every == 0
                ckpt = (step + 1) % a.checkpoint_every == 0
                pub = srv.post("/v1/publish", {"expected_step": res["step"], "save": save or ckpt, "checkpoint": ckpt})
                version = pub["policy_version"]
                m["publish"] = {k: pub.get(k) for k in ("policy_version", "step", "directory", "checkpoint")}
        else:
            m["train"] = None  # 没有任何组有差异：不更新，策略不变
        m["seconds"] = {"rollout": round(t1 - t0, 1), "train+publish": round(time.time() - t1, 1),
                        "total": round(time.time() - t0, 1)}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
        if step % a.dump_every == 0:
            with open(os.path.join(a.out, f"rollout-{step:05d}.jsonl"), "w") as f:
                for r in rows:
                    keep = {k: r[k] for k in ("qid", "source", "question", "golden", "sample", "end", "n_search",
                                              "bad_call", "gen_tokens", "pred", "format_ok", "correct", "reward",
                                              "adv", "kept")}
                    f.write(json.dumps(keep, ensure_ascii=False) + "\n")
        tr = m["train"] or {}
        print(f"step {step}: reward {m['reward']} acc {m['acc']} fmt {m['format_ok']} mixed {m['mixed_groups']}/"
              f"{a.questions} loss {tr.get('loss')} gnorm {tr.get('grad_norm')} lr {tr.get('learning_rate')} "
              f"gen {m['avg_gen_tokens']} | {m['seconds']}", flush=True)
        low_fmt = low_fmt + 1 if a.abort_fmt and m["format_ok"] < a.abort_fmt else 0
        if low_fmt >= 2:
            sys.exit(f"格式正确率连续 2 步低于 {a.abort_fmt}，停训")


if __name__ == "__main__":
    main()
