from dataclasses import dataclass

from . import register_model_config


@dataclass
class GPT2Config:
    vocab_size: int = 10000
    context_length: int = 256

    d_model: int = 512
    d_ff: int = 1344

    num_layers: int = 4
    num_heads: int = 16
    rope_theta: float = 10000


register_model_config("gpt2", GPT2Config)
