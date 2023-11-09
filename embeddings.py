from ray import serve
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer


@serve.deployment(
    name="Embedder",
    autoscaling_config={"min_replicas": 0, "max_replicas": 1})
class EmbeddingsModel:
    def __init__(self):
        self.model = "sentence-transformers/all-mpnet-base-v2"
        self.embedder = SentenceTransformer(self.model)

    def embed(self, data):

        sentences = data.get("input", [])
        model = data.get("model", "sentence-transformers/all-mpnet-base-v2")

        try:
            if model and self.model != model:
                self.model = model
                self.embedder = SentenceTransformer(model)

            tokens = self.embedder.tokenize(sentences)
            # No idea if we need to exclude padding tokens
            tokenCount = sum([len(x) for x in tokens.get("input_ids", [])])

            embeddings = self.embedder.encode(sentences, batch_size=512)

        except Exception as ve:
            return JSONResponse(f"uanble to embed query: {ve}", status_code=500)

        embeddings = [x.tolist() for x in embeddings]

        embedding_resp = {
            "object": "embeddingResponse",
            "model": model,
            "data": [],
            "usage": {
                "prompt_tokens": tokenCount,
                "total_tokens": tokenCount
            }
        }

        for index, emb in enumerate(embeddings):
            embedding_resp["data"].append({
                "object": "embedding",
                "embedding": emb,
                "index": index
            })

        return JSONResponse(embedding_resp)
