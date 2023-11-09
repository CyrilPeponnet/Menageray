# list of models to support with some paramters link etc

'''
Default query options for ref

model:
    description="The model to use for generating completions.", default=None
)

max_tokens:
    default=16, ge=1, description="The maximum number of tokens to generate."
)

temperature:
    default=0.8,
    ge=0.0,
    le=2.0,
    description="Adjust the randomness of the generated text.\n\n"
    + "Temperature is a hyperparameter that controls the randomness of the generated text. It affects the probability distribution of the model's output tokens. A higher temperature (e.g., 1.5) makes the output more random and creative, while a lower temperature (e.g., 0.5) makes the output more focused, deterministic, and conservative. The default value is 0.8, which provides a balance between randomness and determinism. At the extreme, a temperature of 0 will always pick the most likely next token, leading to identical outputs in each run.",
)

top_p:
    default=0.95,
    ge=0.0,
    le=1.0,
    description="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P.\n\n"
    + "Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.",
)

stop:
    default=None,
    description="A list of tokens at which to stop generation. If None, no stop tokens are used.",
)

stream:
    default=False,
    description="Whether to stream the results as they are generated. Useful for chatbots.",
)

top_k:
    default=40,
    ge=0,
    description="Limit the next token selection to the K most probable tokens.\n\n"
    + "Top-k sampling is a text generation method that selects the next token only from the top k most likely tokens predicted by the model. It helps reduce the risk of generating low-probability or nonsensical tokens, but it may also limit the diversity of the output. A higher value for top_k (e.g., 100) will consider more tokens and lead to more diverse text, while a lower value (e.g., 10) will focus on the most probable tokens and generate more conservative text.",
)

repeat_penalty:
    default=1.1,
    ge=0.0,
    description="A penalty applied to each token that is already generated. This helps prevent the model from repeating itself.\n\n"
    + "Repeat penalty is a hyperparameter used to penalize the repetition of token sequences during text generation. It helps prevent the model from generating repetitive or monotonous text. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.",
)

presence_penalty:
    default=0.0,
    ge=-2.0,
    le=2.0,
    description="Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
)

frequency_penalty:
    default=0.0,
    ge=-2.0,
    le=2.0,
    description="Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
)

mirostat_mode:
    default=0,
    ge=0,
    le=2,
    description="Enable Mirostat constant-perplexity algorithm of the specified version (1 or 2; 0 = disabled)",
)

mirostat_tau:
    default=5.0,
    ge=0.0,
    le=10.0,
    description="Mirostat target entropy, i.e. the target perplexity - lower values produce focused and coherent text, larger values produce more diverse and less coherent text",
)

mirostat_eta:
    default=0.1, ge=0.001, le=1.0, description="Mirostat learning rate"
)

grammar = Field(
    default=None,
    description="A CBNF grammar (as string) to be used for formatting the model's output."
)
'''

'''
Default settings options for ref

model:
    description="The path to the model to use for generating completions."
)
model_alias:
    default=None,
    description="The alias of the model to use for generating completions.",
)
# Model Params
n_gpu_layers:
    default=0,
    ge=-1,
    description="The number of layers to put on the GPU. The rest will be on the CPU. Set -1 to move all to GPU.",
)
main_gpu: int
    default=0,
    ge=0,
    description="Main GPU to use.",
)
tensor_split:
    default=None,
    description="Split layers across multiple GPUs in proportion.",
)
vocab_only: bool
    default=False, description="Whether to only return the vocabulary."
)
use_mmap: bool =
    default=llama_cpp.llama_mmap_supported(),
    description="Use mmap.",
)
use_mlock: bool =
    default=llama_cpp.llama_mlock_supported(),
    description="Use mlock.",
)
# Context Params
seed: int = Field(default=llama_cpp.LLAMA_DEFAULT_SEED,
n_ctx: int = Field(default=2048, ge=1, description="The context size.")
n_batch:
    default=512, ge=1, description="The batch size to use per eval."
)
n_threads:
    default=max(multiprocessing.cpu_count() // 2, 1),
    ge=1,
    description="The number of threads to use.",
)
n_threads_batch:
    default=max(multiprocessing.cpu_count() // 2, 1),
    ge=0,
    description="The number of threads to use when batch processing.",
)
rope_scaling_type:
    default=llama_cpp.LLAMA_ROPE_SCALING_UNSPECIFIED
)
rope_freq_base:
    default=0.0, description="RoPE base frequency"
)
rope_freq_scale:
    default=0.0, description="RoPE frequency scaling factor"
)
yarn_ext_factor:
    default=-1.0
)
yarn_attn_factor:
    default=1.0
)
yarn_beta_fast:
    default=32.0
)
yarn_beta_slow:
    default=1.0
)
yarn_orig_ctx:
    default=0
)
mul_mat_q:
    default=True, description="if true, use experimental mul_mat_q kernels"
)
f16_kv: bool = Field(default=True, description="Whether to use f16 key/value.")
logits_all: bool = Field(default=True, description="Whether to return logits.")
embedding: bool = Field(default=True, description="Whether to use embeddings.")
# Sampling Params
last_n_tokens_size:
    default=64,
    ge=0,
    description="Last n tokens to keep for repeat penalty calculation.",
)
# LoRA Params
lora_base:
    default=None,
    description="Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model."
)
lora_path:
    default=None,
    description="Path to a LoRA file to apply to the model.",
)
# Backend Params
numa:
    default=False,
    description="Enable NUMA support.",
)
# Chat Format Params
chat_format:
    default="llama-2",
    description="Chat format to use.",
)
clip_model_path:
    default=None,
    description="Path to a CLIP model to use for multi-modal chat completion.",
)
# Cache Params
cache:
    default=False,
    description="Use a cache to reduce processing times for evaluated prompts.",
)
cache_type:
    default="ram",
    description="The type of cache to use. Only used if cache is True.",
)
cache_size:
    default=2 << 30,
    description="The size of the cache in bytes. Only used if cache is True.",
)
# Misc
verbose:
    default=True, description="Whether to print debug information."
)
'''


# TODO: We put that here for now to avoid usinng a shared storage of some
# sort to make it ray cluster wise available
# Here were define our model alias that will be downloaded if doesnt exists locally
# with some startup option and default query options if needed.
models = """
zephyr:
  path: https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf
  settings:
     n_gpu_layers: 1
     n_ctx: 2048
  default:
"""
