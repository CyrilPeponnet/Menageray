from fastapi import FastAPI, Request
from ray import serve
from ray.serve.handle import DeploymentHandle


from llamacpp import LlamaCPPModel
from embeddings import EmbeddingsModel
from reranking import RerankingModel

app = FastAPI(title="MyLLM")


@serve.deployment()
@serve.ingress(app)
class APIIngress:

    def __init__(self,
                 embedding_responder,
                 reranking_responder,
                 llm_responder) -> None:

        self.embedder: DeploymentHandle = embedding_responder.options(
            use_new_handle_api=True,
        )
        self.reranker = reranking_responder.options(
            use_new_handle_api=True,
        )
        self.llm = llm_responder.options(
            use_new_handle_api=True,
        )

    @app.post("/v1/embeddings")
    async def embed(self, request: Request):
        req = await request.json()
        return await self.embedder.embed.remote(req)

    @app.post("/v1/reranking")
    async def rerank(self, request: Request):
        req = await request.json()
        return await self.reranker.rerank.remote(req)

    @app.post("/v1/completions")
    @app.post("/v1/engines/copilot-codex/completions")
    async def complete(self, request: Request):
        req = await request.json()
        return await self.llm.complete.remote(req)

    @app.post("/v1/chat/completions")
    async def chatcomplete(self, request: Request):
        req = await request.json()
        return await self.llm.chatcomplete.remote(req)


entrypoint = APIIngress.bind(
                            EmbeddingsModel.bind(),
                            RerankingModel.bind(),
                            LlamaCPPModel.bind()
                            )
