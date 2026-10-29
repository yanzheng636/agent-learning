"""资料库 18：中文 BM25 + 向量召回 + RRF，封装成 LangChain BaseRetriever。

uv run p18a_hybrid_retrieval.py --backend both
uv run p18a_hybrid_retrieval.py --backend qdrant --query '买回来坏了，寄回去谁付钱？'
uv run p18a_hybrid_retrieval.py --check   真实数据库契约检查，不下载 Embedding

首次正常运行会下载 BAAI/bge-small-zh-v1.5（约 100 MB，缓存后复用），需要能访问模型托管站。
只用本地 Embedding，不假定聊天供应商支持 Embedding；两后端使用同一组向量。
Chroma 临时集合；Qdrant 默认内存模式，设 QDRANT_URL 可接远程服务（只删除本例自建集合）。
先按系统租户、可选分类过滤，再分别召回，RRF 合并排名而非直接相加原始分数。
"""

import argparse
import math
import re
import uuid
from typing import Any

import jieba
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from rank_bm25 import BM25Okapi

from setting import Settings


class RetrievalSettings(Settings):
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    qdrant_url: str = ""
    qdrant_api_key: str = Field(default="", repr=False)


# 固定的教学资料与人工相关性标签，租户不是模型可传入的参数。
CORPUS = [
    {
        "id": "return/quality",
        "tenant": "shop",
        "category": "return",
        "text": "商品有质量问题、到货损坏或故障，退货寄回运费由小店承担。",
    },
    {
        "id": "return/change",
        "tenant": "shop",
        "category": "return",
        "text": "签收后七天内商品完好可无理由退货；不喜欢、买错了，寄回运费由顾客承担。",
    },
    {
        "id": "return/evidence",
        "tenant": "shop",
        "category": "return",
        "text": "质量问题申请售后时，提供订单号和故障照片，客服核实后给出退货地址。",
    },
    {
        "id": "shipping/time",
        "tenant": "shop",
        "category": "shipping",
        "text": "付款后四十八小时内发货，默认顺丰，偏远地区三到五天送达。",
    },
    {
        "id": "shipping/address",
        "tenant": "shop",
        "category": "shipping",
        "text": "未发货时可申请修改收货地址，已发货需联系快递，不保证能拦截。",
    },
    {
        "id": "warranty/headphone",
        "tenant": "shop",
        "category": "warranty",
        "text": "耳机保修半年；保修期内非人为损坏可免费维修，人为损坏不在保修范围。",
    },
    {
        "id": "warranty/appliance",
        "tenant": "shop",
        "category": "warranty",
        "text": "电饭煲等电器保修一年；超过保修期的维修需报价，确认后处理。",
    },
    {
        "id": "invoice/request",
        "tenant": "shop",
        "category": "invoice",
        "text": "电子发票在订单完成后申请，需提供抬头和税号，开票金额以实付为准。",
    },
    {
        "id": "refund/time",
        "tenant": "shop",
        "category": "return",
        "text": "退货入库验收后原路退款，银行到账通常需要一到三个工作日。",
    },
    {
        "id": "other/private",
        "tenant": "other",
        "category": "return",
        "text": "商品质量故障损坏退货运费，另一租户的内部政策不可对小店顾客展示。",
    },
]
EVAL = [
    ("买回来坏了，寄回去谁付钱？", {"return/quality"}),
    ("不喜欢这个商品可以退吗，邮费谁出？", {"return/change"}),
    ("耳机免费修多久？", {"warranty/headphone"}),
    ("电饭煲用了八个月坏了还有保修吗？", {"warranty/appliance"}),
    ("退货后钱什么时候回到银行卡？", {"refund/time"}),
]


def tokens(text: str) -> list[str]:
    return [t.lower() for t in jieba.lcut(text) if re.search(r"[\w\u4e00-\u9fff]", t)]


def rrf(rankings: list[list[dict]], k: int = 60) -> list[dict]:
    if k < 1:
        raise ValueError("RRF k 必须为正数")
    scores, rows = {}, {}
    for ranking in rankings:
        seen = set()
        rank = 0
        for row in ranking:
            doc_id = row["id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            rows[doc_id] = row
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    return [dict(rows[i], score=scores[i]) for i in sorted(scores, key=lambda i: (-scores[i], i))]


def metrics(ranked: list[dict], relevant: set[str], k: int) -> tuple[float, float]:
    ids = [d["id"] for d in ranked[:k]]
    recall = len(set(ids) & relevant) / len(relevant) if relevant else 0.0
    reciprocal = next((1 / rank for rank, doc_id in enumerate(ids, 1) if doc_id in relevant), 0.0)
    return recall, reciprocal


class LocalEmbedding:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        print(f"加载真实 Embedding：{model_name}（首次需要下载）")
        self.model = TextEmbedding(model_name=model_name)

    def documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.passage_embed(texts)]

    def query(self, text: str) -> list[float]:
        return next(self.model.query_embed(text)).tolist()


class HybridIndex:
    def __init__(self, embedding, backends: list[str], config: RetrievalSettings, *, tenant: str = "shop"):
        self.embedding, self.tenant = embedding, tenant  # 应用授权身份，不能被工具参数覆盖
        self.clients = {}
        self.collection = "teaching_" + uuid.uuid4().hex
        self.rows = {row["id"]: row for row in CORPUS}
        vectors = embedding.documents([d["text"] for d in CORPUS])
        try:
            for backend in backends:
                if backend == "chroma":
                    import chromadb

                    client = chromadb.EphemeralClient()
                    collection = client.create_collection(
                        self.collection, embedding_function=None, metadata={"hnsw:space": "cosine"}
                    )
                    self.clients[backend] = (client, collection)
                    collection.add(
                        ids=[d["id"] for d in CORPUS],
                        embeddings=vectors,
                        documents=[d["text"] for d in CORPUS],
                        metadatas=[{"tenant": d["tenant"], "category": d["category"]} for d in CORPUS],
                    )
                elif backend == "qdrant":
                    from qdrant_client import QdrantClient, models

                    client = (
                        QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key or None)
                        if config.qdrant_url
                        else QdrantClient(":memory:")
                    )
                    client.create_collection(
                        self.collection,
                        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
                    )
                    self.clients[backend] = (client, None)
                    client.upsert(
                        self.collection,
                        points=[
                            models.PointStruct(id=i, vector=v, payload=row)
                            for i, (row, v) in enumerate(zip(CORPUS, vectors, strict=True))
                        ],
                    )
                else:
                    raise ValueError(f"未知后端：{backend}")
        except Exception:
            self.close()
            raise

    def close(self):
        for backend, (client, _) in self.clients.items():
            try:
                client.delete_collection(self.collection)
            finally:
                if backend == "qdrant":
                    client.close()
        self.clients.clear()

    def search(self, query: str, backend: str, *, category: str | None = None, limit: int = 3) -> dict:
        if not query.strip() or limit < 1:
            raise ValueError("query 不能为空且 limit 必须大于零")
        eligible = [d for d in CORPUS if d["tenant"] == self.tenant and (category is None or d["category"] == category)]
        if not eligible:
            return {"bm25": [], "vector": [], "rrf": []}
        wordlists = [tokens(d["text"]) for d in eligible]
        scores = BM25Okapi(wordlists).get_scores(tokens(query))
        # 没有关键词交集时 BM25 无召回；向量仍会有近邻，所以不能把 top-k 当相关性证明。
        lexical = [
            dict(d, score=float(score))
            for d, words, score in zip(eligible, wordlists, scores, strict=True)
            if set(tokens(query)) & set(words)
        ]
        lexical.sort(key=lambda d: (-d["score"], d["id"]))
        client, collection = self.clients[backend]
        vector = self.embedding.query(query)
        if backend == "chroma":
            conditions = [{"tenant": self.tenant}] + ([{"category": category}] if category else [])
            where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
            result = collection.query(
                query_embeddings=[vector], where=where, n_results=min(limit, len(eligible)), include=["distances"]
            )
            dense = [
                dict(self.rows[i], score=1 - float(distance))
                for i, distance in zip(result["ids"][0], result["distances"][0], strict=True)
            ]
        else:
            from qdrant_client import models

            must = [models.FieldCondition(key="tenant", match=models.MatchValue(value=self.tenant))]
            if category:
                must.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))
            result = client.query_points(
                self.collection, query=vector, query_filter=models.Filter(must=must), limit=limit
            )
            dense = [dict(self.rows[p.payload["id"]], score=float(p.score)) for p in result.points]
        lexical = lexical[:limit]
        return {"bm25": lexical, "vector": dense, "rrf": rrf([lexical, dense])[:limit]}


class HybridRetriever(BaseRetriever):
    engine: Any
    backend: str
    category: str | None = None
    limit: int = 3

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        rankings = self.engine.search(query, self.backend, category=self.category, limit=self.limit)
        print(f"[retriever/{self.backend}] {query}")
        for name, rows in rankings.items():
            print(f"  {name}: " + ", ".join(f"{i}:{r['id']}={r['score']:.4f}" for i, r in enumerate(rows, 1)))
        return [
            Document(
                page_content=d["text"], metadata={"source": d["id"], "score": d["score"], "category": d["category"]}
            )
            for d in rankings["rrf"]
        ]


def check():
    # 固定数字向量只用于验证数据库过滤/排序接口，不模拟语义效果，也不会用于正常运行。
    class ContractEmbedding:
        def documents(self, texts):
            return [[1.0, (i + 1) / 10, 0.2] for i in range(len(texts))]

        def query(self, text):
            return [1.0, 0.15, 0.2]

    a, b = {"id": "a"}, {"id": "b"}
    merged = rrf([[a, a, b], [b]])
    assert len(merged) == 2 and merged[0]["id"] == "b"
    assert math.isclose(next(d["score"] for d in merged if d["id"] == "a"), 1 / 61)
    assert math.isclose(next(d["score"] for d in merged if d["id"] == "b"), 1 / 62 + 1 / 61)
    assert metrics([a, b], {"b"}, 2) == (1.0, 0.5)
    config = RetrievalSettings(qdrant_url="", qdrant_api_key="")  # 检查永远只操作本地
    engine = HybridIndex(ContractEmbedding(), ["chroma", "qdrant"], config)
    try:
        ranks = []
        for backend in engine.clients:
            result = engine.search("质量故障退货运费", backend, category="return")
            for rows in result.values():
                assert rows and all(d["tenant"] == "shop" and d["category"] == "return" for d in rows)
            assert engine.search("运费", backend, category="missing") == {"bm25": [], "vector": [], "rrf": []}
            docs = HybridRetriever(engine=engine, backend=backend, category="return").invoke("质量故障退货运费")
            assert all(isinstance(d, Document) and d.metadata["source"] != "other/private" for d in docs)
            ranks.append([d["id"] for d in result["vector"]])
        assert ranks[0] == ranks[1]
    finally:
        engine.close()
    print("真实 Chroma/Qdrant 本地过滤、相同向量排序、RRF 去重、指标与 LangChain Retriever 检查通过；未下载模型。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["chroma", "qdrant", "both"], default="both")
    parser.add_argument("--query", default="买回来坏了，寄回去谁付钱？")
    parser.add_argument("--category", choices=["return", "shipping", "warranty", "invoice"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    config = RetrievalSettings()
    backends = ["chroma", "qdrant"] if args.backend == "both" else [args.backend]
    engine = HybridIndex(LocalEmbedding(config.embedding_model), backends, config)
    try:
        for backend in backends:
            docs = HybridRetriever(engine=engine, backend=backend, category=args.category).invoke(args.query)
            for doc in docs:
                print(f"[{doc.metadata['source']}] {doc.page_content}")
            print("人工标签评测（未设分类过滤，Recall@3 / MRR@3）：")
            totals = {name: [0.0, 0.0] for name in ("bm25", "vector", "rrf")}
            for query, relevant in EVAL:
                ranked = engine.search(query, backend)
                print("问题：", query, "相关：", sorted(relevant))
                for name, rows in ranked.items():
                    recall, reciprocal = metrics(rows, relevant, 3)
                    totals[name][0] += recall / len(EVAL)
                    totals[name][1] += reciprocal / len(EVAL)
                    print(f"  {name}: {[d['id'] for d in rows]} | {recall:.2f}/{reciprocal:.2f}")
            print("平均指标：", totals, "；小样本不能证明后端优劣。向量近邻也可能不相关，请检查原文。")
    finally:
        engine.close()


if __name__ == "__main__":
    main()

# 试试改：
# - 写代码：增加候选数、最终 top-k 两个独立参数，对照不同 RRF k 的 Recall 和 MRR。
# - 写代码：增加拼错商品名、跨租户和无答案样本，做拒答阈值评测；不要直接把距离当置信概率。
