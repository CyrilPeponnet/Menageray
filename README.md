# MenageRay

MenageRay (contraption between Menagerie and Ray) is a hacky collection of LLM models exposed through a Ray serve ingress to perform the following:

- embeddings with Encoders models (CPU)
- re-ranking with Cross Encoders models (CPU)
- text generation with Quantized models through LLamacpp (Metal on Apple Silicon)

This is an attempt to have a working setup on your laptop.

## Usage

Install conda and create an env as:

```console
conda create -n menageray python=3.11
conda activa menageray
conda install -c conda-forge pip
pip install -r requirements.txt
```

Now you can simply run `serve run menageray:entrypoint` (or deploy it see Ray documentation)

This will expose the following API on http://localhost:8000

- /v1/embeddings (openAI api format)
- /v1/reranking (query as `{"query": "your query", "documents: ["doc1","doc2"], "model": "model"}` returns `[float,float]`) 
- /v1/completions (openAI api format)
- /v1/chat/completions (openAI api format)

## Models

While embeddings and reranking takes usual sentence transformer models, the LLamacpp endpoints for completions need Quantized moddels. For now its dumb and opinionated see the `models.py` as example. 

