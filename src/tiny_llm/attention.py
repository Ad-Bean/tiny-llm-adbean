import mlx.core as mx
import math
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    d = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    scores = (query @ key.swapaxes(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask

    attention = softmax(scores, axis=-1)
    return attention @ value

# https://nlp.seas.harvard.edu/annotated-transformer
class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # query, key, value: N x L x E
        N, L, E = query.shape
        H = self.num_heads
        D = self.head_dim

        # Project and reshape to N x L x H x D
        q = linear(query, self.wq).reshape(N, L, H, D)
        k = linear(key, self.wk).reshape(N, L, H, D)
        v = linear(value, self.wv).reshape(N, L, H, D)

        # Transpose to N x H x L x D
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply attention
        # Output: N x H x L x D
        x = scaled_dot_product_attention_simple(q, k, v, mask=mask)

        # Transpose back to N x L x H x D
        x = x.transpose(0, 2, 1, 3)

        # Flatten heads to N x L x (H * D) -> N x L x E
        # Note: In most implementations E = H * D. Here we assume so as well for the reshape.
        # However, if wq maps E to something else, we should rely on the projection output dimension.
        # But standard MHA maps to H*D.
        x = x.reshape(N, L, H * D)

        # Output projection
        return linear(x, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
