from ray import serve
from fastapi.responses import JSONResponse
from llama_cpp.server.app import Settings, make_logit_bias_processor, CreateCompletionRequest, CreateChatCompletionRequest

from models import models
import llama_cpp
import yaml
import os
import urllib
import pathlib


@serve.deployment(
    name="LlamaCPP",
    autoscaling_config={"min_replicas": 0, "max_replicas": 1})
class LlamaCPPModel:
    def __init__(self):
        # we stat empty as we dont really need anything right now
        # we just load our models
        self.model = ""
        self.llm = None
        self.models = yaml.safe_load(models)
        self.__modelspath = ".models/"

    def startLLM(self, model):
        # check if model exists
        if model not in self.models:
            raise Exception("model not found")

        m = self.models[model]
        # build our model path
        modelPath = os.path.abspath(self.__modelspath + "/" + os.path.basename(m["path"]))
        # check if model is local or download it
        if not os.path.isfile(modelPath):
            pathlib.Path(self.__modelspath).mkdir(parents=True, exist_ok=True)

            def show_progress(block_num, block_size, total_size):
                print(round(block_num * block_size / total_size * 100, 2), end="\r")
            print(f"> Downloading {model} from {m['path']}...")
            urllib.request.urlretrieve(m["path"], modelPath, reporthook=show_progress)

        # set our settings on top of default
        settings = Settings(**{"model": modelPath, **m["settings"]})

        # The following is borowed from https://github.com/abetlen/llama-cpp-python/blob/82072802ea0eb68f7f226425e5ea434a3e8e60a0/llama_cpp/server/app.py#L384C5-L441
        chat_handler = None
        if settings.chat_format == "llava-1-5":
            assert settings.clip_model_path is not None
            chat_handler = llama_cpp.llama_chat_format.Llava15ChatHandler(
                clip_model_path=settings.clip_model_path,
                verbose=settings.verbose)

        self.llm = llama_cpp.Llama(
            model_path=settings.model,
            # Model Params
            n_gpu_layers=settings.n_gpu_layers,
            main_gpu=settings.main_gpu,
            tensor_split=settings.tensor_split,
            vocab_only=settings.vocab_only,
            use_mmap=settings.use_mmap,
            use_mlock=settings.use_mlock,
            # Context Params
            seed=settings.seed,
            n_ctx=settings.n_ctx,
            n_batch=settings.n_batch,
            n_threads=settings.n_threads,
            n_threads_batch=settings.n_threads_batch,
            rope_scaling_type=settings.rope_scaling_type,
            rope_freq_base=settings.rope_freq_base,
            rope_freq_scale=settings.rope_freq_scale,
            yarn_ext_factor=settings.yarn_ext_factor,
            yarn_attn_factor=settings.yarn_attn_factor,
            yarn_beta_fast=settings.yarn_beta_fast,
            yarn_beta_slow=settings.yarn_beta_slow,
            yarn_orig_ctx=settings.yarn_orig_ctx,
            mul_mat_q=settings.mul_mat_q,
            f16_kv=settings.f16_kv,
            logits_all=settings.logits_all,
            embedding=settings.embedding,
            # Sampling Params
            last_n_tokens_size=settings.last_n_tokens_size,
            # LoRA Params
            lora_base=settings.lora_base,
            lora_path=settings.lora_path,
            # Backend Params
            numa=settings.numa,
            # Chat Format Params
            chat_format=settings.chat_format,
            chat_handler=chat_handler,
            # Misc
            verbose=settings.verbose,
        )

        if settings.cache:
            if settings.cache_type == "disk":
                if settings.verbose:
                    print(f"Using disk cache with size {settings.cache_size}")
                cache = llama_cpp.LlamaDiskCache(capacity_bytes=settings.cache_size)
            else:
                if settings.verbose:
                    print(f"Using ram cache with size {settings.cache_size}")
                cache = llama_cpp.LlamaRAMCache(capacity_bytes=settings.cache_size)

        cache = llama_cpp.LlamaCache(capacity_bytes=settings.cache_size)
        self.llm.set_cache(cache)

    def complete(self, data):
        body = CreateCompletionRequest(**data)

        # how model swapping if needed
        if body.model is not None and body.model != self.model:
            try:
                self.startLLM(body.model)
            except Exception as ex:
                return JSONResponse(f"unable to start llm for model {body.model}: {ex}", status_code=500)

            self.model = body.model

        if isinstance(body.prompt, list):
            assert len(body.prompt) <= 1
            body.prompt = body.prompt[0] if len(body.prompt) > 0 else ""

        exclude = {
            "n",
            "best_of",
            "logit_bias",
            "logit_bias_type",
            "user",
        }

        kwargs = body.model_dump(exclude=exclude)
        # update with our default if any
        kwargs.update(self.models[self.model].get("default") or {})

        if body.logit_bias is not None:
            kwargs["logits_processor"] = llama_cpp.LogitsProcessorList(
                [
                    make_logit_bias_processor(self.llm, body.logit_bias, body.logit_bias_type),
                ]
            )

        if body.grammar is not None:
            kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(body.grammar)

        return self.llm(**kwargs)

    def chatcomplete(self, data):
        body = CreateChatCompletionRequest(**data)
        exclude = {
            "n",
            "logit_bias",
            "logit_bias_type",
            "user",
        }
        # how model swapping if needed
        if body.model is not None and body.model != self.model:
            try:
                self.startLLM(body.model)
            except Exception as ex:
                return JSONResponse(f"unable to start llm for model {body.model}: {ex}", status_code=500)

            self.model = body.model

        kwargs = body.model_dump(exclude=exclude)
        # update with our default if any
        kwargs.update(self.models[self.model].get("default") or {})

        if body.logit_bias is not None:
            kwargs["logits_processor"] = llama_cpp.LogitsProcessorList(
                [
                    make_logit_bias_processor(self.llm, body.logit_bias, body.logit_bias_type),
                ]
            )

        if body.grammar is not None:
            kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(body.grammar)

        return self.llm.create_chat_completion(**kwargs)
