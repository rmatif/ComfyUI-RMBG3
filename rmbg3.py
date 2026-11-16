import re

import torch

# The Fibo 2.0 DiT in ComfyUI fuses its main attention into qkv and every other attention's k/v into to_kv.
MAIN_ATTENTION = re.compile(r"(layers|noise_refiner|context_refiner)\.\d+\.attention")


def diffusers_to_comfy(sd):
    for prefix in {key.rsplit(".to_k.", 1)[0] for key in sd if ".to_k." in key}:
        main = MAIN_ATTENTION.fullmatch(prefix) is not None
        for kind in ("weight", "bias"):
            if f"{prefix}.to_k.{kind}" in sd:
                k, v = sd.pop(f"{prefix}.to_k.{kind}"), sd.pop(f"{prefix}.to_v.{kind}")
                if main:
                    sd[f"{prefix}.qkv.{kind}"] = torch.cat([sd.pop(f"{prefix}.to_q.{kind}"), k, v])
                else:
                    sd[f"{prefix}.to_kv.{kind}"] = torch.cat([k, v])
            if f"{prefix}.to_out.0.{kind}" in sd:
                sd[f"{prefix}.{'out' if main else 'to_out'}.{kind}"] = sd.pop(f"{prefix}.to_out.0.{kind}")
        if main:
            for n in "qk":
                if f"{prefix}.norm_{n}.weight" in sd:
                    sd[f"{prefix}.{n}_norm.weight"] = sd.pop(f"{prefix}.norm_{n}.weight")
    return sd
