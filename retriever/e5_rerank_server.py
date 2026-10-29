"""e5 重排探针：BM25 召回 top-N 候选 → e5-base-v2 语义重排 → 返回 top-k。

不建全量索引就能测 e5 相对 BM25 的核心增量之一：同一召回池内更好的排序
（另一半增量"词面全漏的语义召回"只有全量 faiss 索引才有，此探针测不到，
两者差值即全量索引的剩余上限）。

接口与 bm25_server.py 完全一致（POST /retrieve），默认端口 8001。
文档正文原样透传（contents 不动），只重排顺序。

v2（2026-10-26）：修复大批量 500——
  1. 查询分块处理（--chunk，默认 256）：BM25 调用与编码都按块走，单个 pyserini
     巨批不再有；块内异常带详细信息返回（不再是无信息的 500）。
  2. CUDA 上模型转 fp16、编码批加大（--encode_batch 128），冷缓存一轮从 ~400s 降一个量级。
  3. 每个请求打一行日志（query 数、新文档数、耗时），服务端日志即诊断信息。
"""

import argparse
import threading
import time
import traceback

import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


class E5Reranker:
    def __init__(self, model_name: str, bm25_url: str, candidates: int,
                 device: str, encode_batch: int, chunk: int, use_fp16: bool):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        if use_fp16 and device == "cuda":
            model = model.half()
        self.model = model.to(device).eval()
        self.device = device
        self.bm25_url = bm25_url.rstrip("/")
        self.candidates = candidates
        self.encode_batch = encode_batch
        self.chunk = chunk
        self._doc_emb: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    @torch.no_grad()
    def _encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        prefix = "query: " if is_query else "passage: "
        feed = [prefix + t for t in texts]
        out = []
        for i in range(0, len(feed), self.encode_batch):
            enc = self.tok(feed[i : i + self.encode_batch], padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            emb = (h * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), dtype="float32")

    def _bm25(self, queries: list[str]) -> list[list[dict]]:
        r = requests.post(f"{self.bm25_url}/retrieve",
                          json={"queries": queries, "topk": self.candidates},
                          timeout=900)
        r.raise_for_status()
        return r.json()["result"]

    def retrieve(self, queries: list[str], topk: int) -> list[list[dict]]:
        t0 = time.monotonic()
        new_docs = 0
        result: list[list[dict]] = []
        with self._lock:
            for c in range(0, len(queries), self.chunk):
                qs = queries[c : c + self.chunk]
                try:
                    cand = self._bm25(qs)
                except Exception as e:
                    raise HTTPException(500, f"bm25 chunk failed (n={len(qs)}): {e!r}") from e
                todo, seen = [], set()
                for docs in cand:
                    for d in docs:
                        did = d["document"]["id"]
                        if did not in self._doc_emb and did not in seen:
                            seen.add(did)
                            todo.append(d["document"])
                new_docs += len(todo)
                if todo:
                    try:
                        embs = self._encode([d["contents"] for d in todo], is_query=False)
                    except Exception as e:
                        raise HTTPException(
                            500, f"e5 encode failed (n_docs={len(todo)}, "
                                 f"first={todo[0]['contents'][:80]!r}): {e!r}") from e
                    for d, e in zip(todo, embs):
                        self._doc_emb[d["id"]] = e
                try:
                    qembs = self._encode(qs, is_query=True)
                except Exception as e:
                    raise HTTPException(500, f"e5 query encode failed (n={len(qs)}): {e!r}") from e
                for qemb, docs in zip(qembs, cand):
                    if not docs:  # 个别 query 在 BM25 零命中（空/怪字符），直接返回空结果
                        result.append([])
                        continue
                    ids = [d["document"]["id"] for d in docs]
                    try:
                        scores = np.stack([self._doc_emb[i] for i in ids]) @ qemb
                    except KeyError as e:
                        raise HTTPException(500, f"docid {e} missing from emb cache") from e
                    order = np.argsort(-scores)[:topk]
                    result.append([{"document": docs[i]["document"], "score": float(scores[i])}
                                   for i in order])
        dt = time.monotonic() - t0
        print(f"[retrieve] n={len(queries)} new_docs={new_docs} "
              f"cache={len(self._doc_emb)} {dt:.1f}s", flush=True)
        return result


def build_app(model_name: str, bm25_url: str, candidates: int, default_topk: int,
              encode_batch: int, chunk: int) -> FastAPI:
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    use_fp16 = device == "cuda"
    print(f"[e5-rerank] device={device} fp16={use_fp16} model={model_name} "
          f"bm25={bm25_url} candidates={candidates} chunk={chunk}", flush=True)
    reranker = E5Reranker(model_name, bm25_url, candidates, device, encode_batch, chunk, use_fp16)
    app = FastAPI()

    @app.post("/retrieve")
    def retrieve(req: QueryRequest):
        topk = req.topk or default_topk
        result = reranker.retrieve(req.queries, topk)
        if not req.return_scores:
            result = [[{k: v for k, v in d.items() if k != "score"} for d in docs]
                       for docs in result]
        return {"result": result}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument("--bm25", default="http://127.0.0.1:8000")
    parser.add_argument("--candidates", type=int, default=50, help="BM25 召回候选数")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--chunk", type=int, default=256, help="单次 BM25/编码的查询分块大小")
    parser.add_argument("--encode_batch", type=int, default=128, help="e5 编码批大小")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    app = build_app(args.model, args.bm25, args.candidates, args.topk,
                    args.encode_batch, args.chunk)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
