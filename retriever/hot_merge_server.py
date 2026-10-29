"""FictionalHot 检索：wiki-18（远端 e5 Flat 服务）+ 虚构文档（本地 e5 Flat），按分数合并。

ReSeek 的 FictionalHot 把 GPT-5 写的虚构实体文档插进 wiki-18 再建 e5 Flat 索引（TencentBAC/ReSeek-corpus）。
两个库互不重叠、都是精确内积检索，各取 top-k 再按分数合并，等价于在合并后的库上检索，
所以不用下载 64.7GB 的 emb_e5_hot.index，也不用动给训练供数的 8002 服务。

虚构文档的编码方式照 Search-R1 / FlashRAG 的 e5 索引：前缀 "passage: "、mean pooling、L2 归一化。
启动时自检：本地重编码远端返回的 wiki-18 文档，分数应与远端一致（远端是 fp16，差 1e-3 量级）。

用法（Mac，8003）：
  uv run --group retriever python retriever/hot_merge_server.py \
      --upstream $RETRIEVER_URL --extra_jsonl data/fictionalhot/corpus/fictional_docs.jsonl
"""

import argparse
import json
import os
import time

import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


class Encoder:
    def __init__(self, model_name: str, batch: int):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(self.device).eval()
        self.batch = batch

    @torch.no_grad()
    def encode(self, texts: list[str], prefix: str) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch):
            enc = self.tok([prefix + t for t in texts[i : i + self.batch]], padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            emb = (h * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(emb, dim=-1).float().cpu().numpy())
        return np.concatenate(out)


class MergedIndex:
    def __init__(self, args):
        self.upstream = args.upstream.rstrip("/")
        self.sess = requests.Session()
        self.sess.trust_env = False  # tailnet 请求不走系统代理
        self.enc = Encoder(args.model, args.encode_batch)
        with open(args.extra_jsonl) as f:
            self.docs = [json.loads(line) for line in f if line.strip()]
        cache = args.extra_jsonl + ".e5.npy"
        if os.path.exists(cache):
            self.emb = np.load(cache)
        else:
            t0 = time.monotonic()
            self.emb = self.enc.encode([d["contents"] for d in self.docs], "passage: ")
            np.save(cache, self.emb)
            print(f"[hot] encoded {len(self.docs)} extra docs ({time.monotonic() - t0:.0f}s)", flush=True)
        assert len(self.emb) == len(self.docs), f"{cache} 与 {args.extra_jsonl} 条数不一致，删掉缓存重编码"
        print(f"[hot] extra docs: {len(self.docs)}, upstream {self.upstream}", flush=True)
        self.self_check()

    def remote(self, queries: list[str], topk: int) -> list[list[dict]]:
        r = self.sess.post(f"{self.upstream}/retrieve",
                           json={"queries": queries, "topk": topk, "return_scores": True}, timeout=1200)
        r.raise_for_status()
        return r.json()["result"]

    def self_check(self):
        qs = ["who wrote the novel 1984", "capital city of australia", "when did the berlin wall fall"]
        res = self.remote(qs, 1)
        qemb = self.enc.encode(qs, "query: ")
        pemb = self.enc.encode([r[0]["document"]["contents"] for r in res], "passage: ")
        local = (qemb * pemb).sum(1)
        remote = np.array([r[0]["score"] for r in res])
        print(f"[hot] self-check remote {np.round(remote, 4)} local {np.round(local, 4)}", flush=True)
        assert np.abs(local - remote).max() < 5e-3, "本地编码与远端索引不一致（前缀/pooling/归一化不同），合并分数不可比"

    def retrieve(self, queries: list[str], topk: int) -> list[list[dict]]:
        remote = self.remote(queries, topk)
        scores = self.enc.encode(queries, "query: ") @ self.emb.T
        k = min(topk, scores.shape[1])
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        out = []
        for qi, docs in enumerate(remote):
            extra = [{"document": {"id": self.docs[j]["id"], "contents": self.docs[j]["contents"]},
                      "score": float(scores[qi, j])} for j in top[qi]]
            out.append(sorted(docs + extra, key=lambda d: -d["score"])[:topk])
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default=os.environ.get("RETRIEVER_URL", "http://127.0.0.1:8002"),
                   help="上游 e5 检索地址，默认读环境变量 RETRIEVER_URL")
    p.add_argument("--extra_jsonl", required=True, help="虚构文档，每行 {id, contents}")
    p.add_argument("--model", default="intfloat/e5-base-v2")
    p.add_argument("--encode_batch", type=int, default=64)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8003)
    args = p.parse_args()

    index = MergedIndex(args)
    app = FastAPI()

    @app.post("/retrieve")
    def retrieve(req: QueryRequest):
        t0 = time.monotonic()
        result = index.retrieve(req.queries, req.topk or args.topk)
        if not req.return_scores:
            result = [[{k: v for k, v in d.items() if k != "score"} for d in docs] for docs in result]
        print(f"[retrieve] n={len(req.queries)} {time.monotonic() - t0:.1f}s", flush=True)
        return {"result": result}

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
