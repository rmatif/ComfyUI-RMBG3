import os

import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io

from .rmbg3 import diffusers_to_comfy


def resolve(path):
    path = path.strip()
    return path if os.path.isabs(path) else folder_paths.get_full_path_or_raise("diffusion_models", path)


class RMBG3ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RMBG3ModelLoader",
            display_name="Load RMBG 3.0 Model",
            category="advanced/loaders",
            description="Loads Bria RMBG 3.0, or any Fibo 2.0 transformer saved in Diffusers' sharded format, as a native Fibo 2.0 model.",
            inputs=[io.String.Input("shards", multiline=True, tooltip="The transformer's safetensors shards, one per line: absolute paths or names in diffusion_models.")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, shards):
        sd = {}
        for line in shards.splitlines():
            if line.strip():
                sd.update(comfy.utils.load_torch_file(resolve(line), safe_load=True))
        return io.NodeOutput(comfy.sd.load_diffusion_model_state_dict(diffusers_to_comfy(sd)))


class RMBG3Extension(ComfyExtension):
    async def get_node_list(self):
        return [RMBG3ModelLoader]


async def comfy_entrypoint():
    return RMBG3Extension()
