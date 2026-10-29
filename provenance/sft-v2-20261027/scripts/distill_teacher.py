"""用 teacher（Qwen3.7-Flash）在 Search-R1 train 集上做题，生成 G1K 格式的 SFT 语料。

teacher 每步只出一个动作（<tool_call> 或 <answer>），检索由本地 BM25 真实执行；思考取自
reasoning_content。轨迹按 RWKV G1 模板重新渲染，字节和 eval_base.py 的 G1KFormat（think=full）一致。

训练行格式照 RWKV-Agent v2 系列：{"segments":[{"text","train"}]}，上下文以 "Assistant: <think" 结尾，
训练片段为 ">思考</think>\\n动作\\n\\nUser:"。我们的历史保留思考，所以一条轨迹一行、每步一个训练片段。

凭据从环境变量读：API_BASE、API_KEY（先 source 自己的密钥脚本，不要入库）。
"""

import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from eval_base import G1K_NUDGE, G1KFormat, clean_txt, contains, em, to_docs, truncate_docs

# 只给 teacher 看、不进训练语料（同 RWKV-Agent 的 SYSTEM_SUFFIX）
TEACHER_SUFFIX = (
    "Teacher notes: always run at least one search before giving the final answer, even if you think you know it. "
    "The final answer must be as short as possible: only the entity, name, number or date; for yes/no questions answer yes or no. "
    "Keep your thinking brief and plain: a few sentences, no Markdown lists."
)
# 试点里 6–8% 的思考步会复述上面的附加说明，而学生 prompt 里没有这段
TEACHER_SUFFIX_QUIET = " These teacher notes are private: never mention, quote or refer to them (or to any rule about searching or answer length) in your thinking."

ACTION_RE = re.compile(r"^(<tool_call>(.*)</tool_call>|<answer>(.*)</answer>)$", re.S)


class Teacher:
    def __init__(self, args):
        self.args = args
        self.base = os.environ["API_BASE"].rstrip("/")
        self.key = os.environ["API_KEY"]

    def chat(self, messages: list[dict]) -> tuple[str, str]:
        a = self.args
        body = {
            "model": a.model, "messages": messages, "temperature": a.temperature, "top_p": 1,
            "max_tokens": a.max_tokens, "chat_template_kwargs": {"enable_thinking": True},
        }
        if a.reasoning_effort:
            body["reasoning_effort"] = a.reasoning_effort
        for attempt in range(5):
            try:
                r = requests.post(f"{self.base}/chat/completions", json=body, timeout=600,
                                  headers={"Authorization": f"Bearer {self.key}"})
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                m = r.json()["choices"][0]["message"]
                return (m.get("content") or "").strip(), (m.get("reasoning_content") or m.get("reasoning") or "").strip()
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))


class Retriever:
    def __init__(self, url: str, topk: int):
        self.url, self.topk = url, topk
        self.lock = threading.Lock()  # pyserini 走 JNI，串行调用更稳；单次检索只要几十毫秒

    def __call__(self, query: str) -> list[dict]:
        with self.lock:
            r = requests.post(f"{self.url}/retrieve", json={"queries": [query], "topk": self.topk}, timeout=120)
        r.raise_for_status()
        return to_docs(r.json()["result"][0])


def parse_action(content: str) -> dict | None:
    """teacher 输出必须恰好是一个动作；工具调用 JSON 规范成紧凑写法。"""
    m = ACTION_RE.match(content.strip())
    if not m:
        return None
    if m.group(3) is not None:
        ans = m.group(3).strip()
        return {"type": "answer", "answer": ans, "text": f"<answer>{ans}</answer>"} if ans else None
    try:
        obj = json.loads(m.group(2))
        q = obj["arguments"]["query"].strip()
        assert obj["name"] == "search" and q
    except Exception:
        return None
    call = json.dumps({"name": "search", "arguments": {"query": q}}, ensure_ascii=False, separators=(",", ":"))
    return {"type": "search", "query": q, "text": f"<tool_call>{call}</tool_call>"}


def solve(row, teacher: Teacher, retrieve: Retriever, fmt: G1KFormat, args) -> dict:
    suffix = TEACHER_SUFFIX + (TEACHER_SUFFIX_QUIET if args.quiet_suffix else "")
    system = fmt.system.removeprefix("System: ") + (f"\n{suffix}" if args.teacher_suffix else "")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": clean_txt(row.question)}]
    steps, status = [], "max_steps"
    for step in range(args.max_steps):
        content, reasoning = teacher.chat(messages)
        act = parse_action(content)
        if act is None:
            steps.append({"reasoning": reasoning, "raw": content, "action": None})
            status = "bad_action"
            break
        st = {"reasoning": reasoning, "raw": content, "action": act}
        steps.append(st)
        messages.append({"role": "assistant", "content": act["text"], "reasoning_content": reasoning})
        if act["type"] == "answer":
            status = "answer"
            break
        if step == args.max_steps - 1:
            break  # 最后一步还在调工具：不收尾，丢弃
        st["docs"] = retrieve(act["query"])
        if args.max_obs_tokens:
            st["docs"] = truncate_docs(st["docs"], args.max_obs_tokens)
        last = step + 1 == args.max_steps - 1
        st["last"] = last
        payload = json.dumps({"ok": True, "tool": "search", "result": st["docs"]}, ensure_ascii=False, separators=(",", ":"))
        obs = f"<tool_response>{payload}</tool_response>" + (f"\n{G1K_NUDGE}" if last else "")
        messages.append({"role": "user", "content": obs})
    pred = steps[-1]["action"]["answer"] if status == "answer" else None
    return {
        "qid": row.id, "source": row.data_source, "question": row.question, "golden": list(row.golden_answers),
        "status": status, "pred": pred, "correct": em(pred, row.golden_answers), "hit": contains(pred, row.golden_answers),
        "n_search": sum(1 for s in steps if s["action"] and s["action"]["type"] == "search"), "steps": steps,
    }


def render(traj: dict, fmt: G1KFormat) -> list[dict]:
    """渲染成 segments：上下文以 "Assistant: <think" 结尾，每步训练 ">思考</think>\\n动作\\n\\nUser:"。"""
    row = pd.Series({"question": traj["question"]})
    segs = [{"text": fmt.prompt(row), "train": False}]
    for st in traj["steps"]:
        think = clean_txt(re.sub(r"^\s*Thinking Process:\s*", "", st["reasoning"]))
        segs.append({"text": f">{think}</think>\n{st['action']['text']}\n\nUser:", "train": True})
        if st["action"]["type"] == "search":
            obs = fmt.observation(st["docs"], st["last"])
            assert obs.startswith("\n\nUser:")
            segs.append({"text": obs[len("\n\nUser:"):], "train": False})
    return segs


def keep(traj: dict, args) -> str | None:
    """返回丢弃原因；None 表示保留。"""
    if traj["status"] != "answer":
        return traj["status"]
    if not traj["correct"]:
        return "wrong"
    if traj["n_search"] < args.min_search:
        return "no_search"
    if any(len(s["reasoning"]) > args.max_think_chars for s in traj["steps"]):
        return "long_think"
    if any(not s["reasoning"] for s in traj["steps"]):
        return "empty_think"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/search_r1/train.parquet")
    p.add_argument("--n_per_source", type=int, default=20, help="0 表示用 --data 里的全部题（已筛好的题单）")
    p.add_argument("--sources", nargs="+", default=["nq", "hotpotqa"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen3.7-Flash")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--reasoning_effort", default=None, help="如 low / medium / high，空则不传")
    p.add_argument("--teacher_suffix", action="store_true", help="给 teacher 追加只它可见的附加指令")
    p.add_argument("--quiet_suffix", action="store_true", help="附加指令里再要求思考中不得提及这些指令")
    p.add_argument("--max_steps", type=int, default=4)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--retriever", default="http://127.0.0.1:8000")
    p.add_argument("--max_obs_tokens", type=int, default=0, help="每次检索结果截到前 N 个 token，0 不截（榜单协议 500）")
    p.add_argument("--parallel", type=int, default=16)
    p.add_argument("--min_search", type=int, default=1)
    p.add_argument("--max_think_chars", type=int, default=3000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.read_parquet(args.data, columns=["id", "question", "golden_answers", "data_source"])
    if args.n_per_source:
        df = pd.concat([df[df.data_source == s].sample(n=args.n_per_source, random_state=args.seed) for s in args.sources])
    os.makedirs(args.out, exist_ok=True)

    fmt = G1KFormat("full")
    teacher, retrieve = Teacher(args), Retriever(args.retriever, args.topk)
    trajs, t0 = [], time.time()
    with ThreadPoolExecutor(args.parallel) as ex, open(os.path.join(args.out, "trajectories.jsonl"), "w") as f:
        futs = {ex.submit(solve, row, teacher, retrieve, fmt, args): row.id for row in df.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                traj = fut.result()
            except Exception as e:
                traj = {"qid": futs[fut], "status": f"error: {e}", "steps": []}
            trajs.append(traj)
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} done, {time.time() - t0:.0f}s", flush=True)

    reasons = Counter()
    with open(os.path.join(args.out, "segments.jsonl"), "w") as f:
        kept = []
        for traj in trajs:
            why = keep(traj, args) if traj["steps"] else traj["status"]
            reasons[why or "kept"] += 1
            if why is None:
                kept.append(traj)
        random.Random(args.seed).shuffle(kept)
        for traj in kept:
            f.write(json.dumps({"segments": render(traj, fmt)}, ensure_ascii=False) + "\n")

    ok = [t for t in trajs if t["steps"]]
    think = [len(s["reasoning"]) for t in ok for s in t["steps"]]
    stats = {
        "questions": len(trajs),
        "filter": dict(reasons),
        "teacher_em": round(sum(t.get("correct", False) for t in ok) / max(len(ok), 1), 4),
        "avg_search": round(sum(t["n_search"] for t in ok) / max(len(ok), 1), 3),
        "search_hist": dict(Counter(t["n_search"] for t in ok)),
        "think_chars_p50_p90_max": [sorted(think)[len(think) // 2], sorted(think)[int(len(think) * 0.9)], max(think)] if think else None,
        "args": vars(args),
    }
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
