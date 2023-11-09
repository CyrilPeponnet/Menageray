from ray import serve
from fastapi.responses import JSONResponse
from sentence_transformers import CrossEncoder


@serve.deployment(
    name="Reranker",
    autoscaling_config={"min_replicas": 0, "max_replicas": 1})
class RerankingModel:
    def __init__(self):
        self.model = "cross-encoder/ms-marco-MiniLM-L-2-v2"
        self.reranker = CrossEncoder(self.model)

    def rerank(self, data):
        query = data.get("query", "")
        documents = data.get("documents", [])
        model = data.get("model", "cross-encoder/ms-marco-MiniLM-L-2-v2")

        try:
            if model and self.model != model:
                self.model = model
                self.reranker = CrossEncoder(model)

            scores = self.reranker.predict([(query, d) for d in documents])

        except Exception as ex:
            return JSONResponse(f"unble to rerank: {ex}", status_code=500)

        results = [x.tolist() for x in scores]

        return JSONResponse(results)
