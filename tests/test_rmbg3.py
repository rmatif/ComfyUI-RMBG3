import torch

from rmbg3 import diffusers_to_comfy


def test_fuses_attention_into_comfy_fibo2_layout():
    q, k, v, out = (torch.full((2, 2), float(i)) for i in range(4))
    sd = {
        "layers.0.attention.to_q.weight": q,
        "layers.0.attention.to_k.weight": k,
        "layers.0.attention.to_v.weight": v,
        "layers.0.attention.to_out.0.weight": out,
        "layers.0.attention.norm_q.weight": q[0],
        "layers.0.attention.norm_k.weight": k[0],
        "layers.0.cross_attention.to_q.weight": q,
        "layers.0.cross_attention.to_k.weight": k,
        "layers.0.cross_attention.to_v.weight": v,
        "layers.0.cross_attention.to_out.0.weight": out,
        "layers.0.cross_attention.norm_q.weight": q[0],
        "layers.0.feed_forward.w1.weight": q,
    }

    sd = diffusers_to_comfy(sd)

    assert sorted(sd) == [
        "layers.0.attention.k_norm.weight",
        "layers.0.attention.out.weight",
        "layers.0.attention.q_norm.weight",
        "layers.0.attention.qkv.weight",
        "layers.0.cross_attention.norm_q.weight",
        "layers.0.cross_attention.to_kv.weight",
        "layers.0.cross_attention.to_out.weight",
        "layers.0.cross_attention.to_q.weight",
        "layers.0.feed_forward.w1.weight",
    ]
    assert torch.equal(sd["layers.0.attention.qkv.weight"], torch.cat([q, k, v]))
    assert torch.equal(sd["layers.0.cross_attention.to_kv.weight"], torch.cat([k, v]))
