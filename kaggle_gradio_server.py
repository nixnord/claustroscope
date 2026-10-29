"""Module 4 Server: kaggle_gradio_server.py

Run this code in a Kaggle notebook with GPU enabled (T4 x 2 or P100).
It exposes the Multi-ControlNet Stable Diffusion img2img inference service
via a public Gradio live tunnel.

Required Kaggle environment packages:
!pip install -q diffusers transformers accelerate gradio xformers
"""

import gradio as gr
from PIL import Image
import torch
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
)

# 1. Load ControlNet Adapters
controlnets = [
    ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16,
    ),
    ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=torch.float16,
    ),
]

# 2. Load Base Checkpoint
base_model_id = "Lykon/dreamshaper-8"
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnets,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

# 3. Configure DPM++ 2M Karras Scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,
)
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pipe.enable_attention_slicing()


def generate_visual_canvas(
    base_image: Image.Image,
    depth_image: Image.Image,
    canny_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    denoising_strength: float,
    cfg_scale: float,
    depth_scale: float,
    edge_scale: float,
    num_steps: int,
) -> Image.Image:
    """Executes Multi-ControlNet Img2Img inference."""
    w, h = base_image.size
    w_adj, h_adj = (w // 8) * 8, (h // 8) * 8

    base_pil = base_image.resize((w_adj, h_adj)).convert("RGB")
    depth_pil = depth_image.resize((w_adj, h_adj)).convert("RGB")
    canny_pil = canny_image.resize((w_adj, h_adj)).convert("RGB")

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=base_pil,
        control_image=[depth_pil, canny_pil],
        strength=float(denoising_strength),
        guidance_scale=float(cfg_scale),
        controlnet_conditioning_scale=[float(depth_scale), float(edge_scale)],
        num_inference_steps=int(num_steps),
    ).images[0]

    return output


# 4. Launch Gradio Interface
demo = gr.Interface(
    fn=generate_visual_canvas,
    inputs=[
        gr.Image(type="pil", label="Vignetted Base Image"),
        gr.Image(type="pil", label="Warped Depth Image"),
        gr.Image(type="pil", label="Modulated Canny Image"),
        gr.Textbox(label="Positive Prompt"),
        gr.Textbox(label="Negative Prompt"),
        gr.Slider(0.1, 1.0, value=0.75, label="Denoising Strength"),
        gr.Slider(1.0, 15.0, value=9.0, label="CFG Scale"),
        gr.Slider(0.1, 2.0, value=0.65, label="ControlNet Depth Scale"),
        gr.Slider(0.1, 2.0, value=0.95, label="ControlNet Edge Scale"),
        gr.Slider(15, 50, value=30, step=1, label="Inference Steps"),
    ],
    outputs=gr.Image(type="pil", label="Generated Visual Canvas"),
    title="Claustrophobia Visualizer Diffusion Worker",
)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
