# ComfyUI RMBG 3.0

Loads [Bria RMBG 3.0](https://huggingface.co/briaai/RMBG-3.0) in ComfyUI. RMBG 3.0 is a Fibo 2.0 Edit Turbo fine-tune that edits a photo into its alpha matte, so everything except loading runs on ComfyUI's native Fibo 2.0 nodes. No additional dependencies.

Clone this repository into ComfyUI's `custom_nodes` directory and restart ComfyUI.

Download the four `transformer/diffusion_pytorch_model-0000N-of-00004.safetensors` shards from `briaai/RMBG-3.0` (gated; accept Bria's terms first) into `models/diffusion_models`, and the Fibo 2.0 text encoder and VAE from [Comfy-Org/Fibo-2.0](https://huggingface.co/Comfy-Org/Fibo-2.0). `Load RMBG 3.0 Model` takes the shards one per line and fuses their attention weights into ComfyUI's Fibo 2.0 layout at load time.

To remove a background:

1. `Load CLIP` (type `fibo2`) and `Load VAE`.
2. `TextEncodeFibo2` with the image as `image_1`, width and height 0, and exactly this prompt, which RMBG 3.0 was trained on:
   ```
   {"edit_instruction": "Generate a detailed grayscale alpha matte. Map the opaque foreground to white and the background to black. Produce soft, anti-aliased grayscale gradients at the edges of the subject to represent fine details and transparency."}
   ```
3. `KSampler` at 4 steps, CFG 1, `euler`, `simple`, then `VAE Decode`. The result is the matte, white where the subject is.
4. `Convert Image to Mask` (red), `InvertMask`, and `Join Image with Alpha` onto the original image for a transparent cutout at the source size.

The code is MIT. The RMBG 3.0 weights are under Bria's license, which requires an agreement with Bria for commercial use.
