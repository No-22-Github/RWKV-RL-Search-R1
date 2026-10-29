"""Search-R1 兼容的 BM25 检索服务（pyserini + wiki-18）。

接口与 Search-R1 的 retrieval_server.py 一致：
    POST /retrieve  {"queries": [...], "topk": 3, "return_scores": true}
    -> {"result": [[{"document": {"id", "contents"}, "score"}, ...], ...]}
"""

import argparse
import os

JAVA_HOME_DEFAULT = "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
if "JAVA_HOME" not in os.environ and os.path.isdir(JAVA_HOME_DEFAULT):
    os.environ["JAVA_HOME"] = JAVA_HOME_DEFAULT

import json
import math
import zlib
from collections import Counter
from functools import lru_cache

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pyserini.index.lucene import LuceneIndexReader
from pyserini.search.lucene import LuceneSearcher


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


class CorpusStore:
    """读 build_corpus_store.py 生成的分块压缩语料，按行号取正文。"""

    def __init__(self, store_dir: str):
        with open(os.path.join(store_dir, "meta.json")) as f:
            self.block = json.load(f)["block"]
        self.offsets = np.load(os.path.join(store_dir, "offsets.npy"))
        self.f = open(os.path.join(store_dir, "corpus.bin"), "rb")
        self._read_block = lru_cache(maxsize=1024)(self._read_block)

    def _read_block(self, b: int) -> list[str]:
        start, end = int(self.offsets[b]), int(self.offsets[b + 1])
        return json.loads(zlib.decompress(os.pread(self.f.fileno(), end - start, start)))

    def __getitem__(self, i: int) -> dict:
        return {"id": str(i), "contents": self._read_block(i // self.block)[i % self.block]}


def _int4(i: int) -> int:
    bits = i.bit_length()
    if bits < 4:
        return i
    shift = bits - 4
    return ((i >> shift) & 7) | ((shift + 1) << 3)


def _uint4(i: int) -> int:
    bits, shift = i & 7, (i >> 3) - 1
    return bits if shift == -1 else (bits | 8) << shift


_FREE = 255 - _int4(2**31 - 1)


def lucene_doc_len(n: int) -> int:
    """Lucene 按 SmallFloat.intToByte4 有损存文档长度，BM25 用的是解码后的值。"""
    return n if n < _FREE else _FREE + _uint4(_int4(n - _FREE))


class ExtraBM25:
    """额外文档（如 FictionalHot 的虚构文档）按“插进主索引后”的口径打 BM25，分数与主索引可直接合并。

    用主索引的分析器和 N/avgdl，df 取主索引 + 额外文档之和（虚构实体名在主索引里 df≈0，不加会虚高）。
    与真正重建索引的差别只在 N/avgdl 少算了额外文档（5.6 万 / 2100 万，约 0.3%）。
    分数实现与 pyserini 默认 BM25（k1=0.9, b=0.4）逐位一致，启动时会自检。
    """

    def __init__(self, path: str, reader: LuceneIndexReader, k1: float = 0.9, b: float = 0.4):
        self.reader, self.k1, self.b = reader, k1, b
        st = reader.stats()
        self.n, self.avgdl = st["documents"], st["total_terms"] / st["documents"]
        with open(path) as f:
            self.docs = [json.loads(line) for line in f if line.strip()]
        self.tf = [Counter(reader.analyze(d["contents"])) for d in self.docs]
        self.dl = [lucene_doc_len(sum(c.values())) for c in self.tf]
        self.df_extra = Counter(t for c in self.tf for t in c)
        self.postings = {}
        for i, c in enumerate(self.tf):
            for t in c:
                self.postings.setdefault(t, []).append(i)
        self._df = lru_cache(maxsize=200_000)(self._df)
        print(f"[bm25] extra docs: {len(self.docs)} from {path}", flush=True)

    def _df(self, term: str) -> int:
        return self.reader.get_term_counts(term, analyzer=None)[0] + self.df_extra[term]

    def _idf(self, df: int) -> float:
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query: str, i: int, df=None) -> float:
        tf, norm = self.tf[i], self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
        return sum(qn * self._idf((df or self._df)(t)) * tf[t] / (tf[t] + norm)
                   for t, qn in Counter(self.reader.analyze(query)).items() if t in tf)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        # 按词累加，和 score() 同一公式；查询只分析一次
        acc = {}
        for t, qn in Counter(self.reader.analyze(query)).items():
            if t not in self.postings:
                continue
            w = qn * self._idf(self._df(t))
            for i in self.postings[t]:
                f = self.tf[i][t]
                acc[i] = acc.get(i, 0.0) + w * f / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return sorted(acc.items(), key=lambda x: -x[1])[:k]

    def self_check(self, searcher: LuceneSearcher, corpus: "CorpusStore"):
        """用主索引自己的文档验证打分公式：只用主索引 df 时应与 Lucene 分数一致。"""
        for q in ["who wrote harry potter", "when did the berlin wall fall"]:
            for h in searcher.search(q, k=2):
                tf = Counter(self.reader.analyze(corpus[int(h.docid)]["contents"]))
                saved = self.tf, self.dl
                self.tf, self.dl = [tf], [lucene_doc_len(sum(tf.values()))]
                mine = self.score(q, 0, df=lambda t: self.reader.get_term_counts(t, analyzer=None)[0])
                self.tf, self.dl = saved
                assert abs(mine - h.score) < 1e-3, f"BM25 自检失败：{q} {h.docid} lucene {h.score} mine {mine}"
        print("[bm25] extra-doc scoring self-check ok", flush=True)


def build_app(index_path: str, store_dir: str, default_topk: int, threads: int,
              extra_jsonl: str | None = None) -> FastAPI:
    searcher = LuceneSearcher(index_path)
    # 索引里没有存正文，按 docid（即行号）回语料存储里取
    corpus = CorpusStore(store_dir)
    extra = None
    if extra_jsonl:
        extra = ExtraBM25(extra_jsonl, LuceneIndexReader(index_path))
        extra.self_check(searcher, corpus)
    app = FastAPI()

    @app.post("/retrieve")
    def retrieve(req: QueryRequest):
        topk = req.topk or default_topk
        qids = [str(i) for i in range(len(req.queries))]
        hits = searcher.batch_search(req.queries, qids, k=topk, threads=threads)
        result = []
        for q, qid in zip(req.queries, qids):
            docs = []
            for h in hits[qid]:
                doc = corpus[int(h.docid)]
                docs.append({"document": {"id": doc["id"], "contents": doc["contents"]}, "score": h.score})
            if extra:
                docs += [{"document": {"id": extra.docs[i]["id"], "contents": extra.docs[i]["contents"]}, "score": s}
                         for i, s in extra.search(q, topk)]
                docs = sorted(docs, key=lambda d: -d["score"])[:topk]
            if not req.return_scores:
                docs = [{"document": d["document"]} for d in docs]
            result.append(docs)
        return {"result": result}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", default="data/retrieval/wiki-18-bm25-index/bm25")
    parser.add_argument("--store_dir", default="data/retrieval/wiki-18-store")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--extra_jsonl", default=None, help="额外文档（每行 {id, contents}），按插入主索引的口径合并检索")
    args = parser.parse_args()

    app = build_app(args.index_path, args.store_dir, args.topk, args.threads, args.extra_jsonl)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
