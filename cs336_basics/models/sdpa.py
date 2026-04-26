import math

import einx
import torch
import torch.nn as nn
from jaxtyping import Bool, Float

from cs336_basics.models.softmax import softmax


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "... seq_len_q d_k"],
    K: Float[torch.Tensor, "... seq_len_k d_k"],
    V: Float[torch.Tensor, "... seq_len_k d_v"],
    mask: Bool[torch.Tensor, "... seq_len_q seq_len_k"] | None = None,
) -> Float[torch.Tensor, "... seq_len_q d_v"]:
    d_k = Q.shape[-1]

    qk_T = einx.dot(
        "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k", Q, K
    )
    qk_T /= math.sqrt(d_k)
    if mask is not None:
        qk_T[~mask] = float("-inf")
    attention_map = softmax(qk_T, dim=-1)
    attention_value = einx.dot(
        "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v",
        attention_map,
        V,
    )
    return attention_value
