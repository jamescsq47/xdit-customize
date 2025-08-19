import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import diffusers
import time
import shutil
import argparse
import logging
from functools import wraps

import diffusers
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

# from wan import profile_WanTransformer3DModel
# # Special for wan
# diffusers.models.WanTransformer3DModel = profile_WanTransformer3DModel

from diffusers import WanPipeline
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
# from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging

import numpy as np
def seed_everything(seed):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

'''
input: F,H,W: fixed as standard.
permute_plan:
    - empty_plan: [N_block(42), N_head(48)]
    - permute_plan: [N_block(42), N_head(48)]
sparse_plan:
    - sparse: [N_timestep(10), N_block(42), N_head(48), N_downsamplertd(278)]
        - 17550 img_token
        - 226   text_token + 30 -> (256//64=4)
        - (17550-30)/64 = 273.75 -> 274
        - 278 = 274 + 4
    - dense_rate: [N_timestep(10), N_block(42), N_head(48)]
---
PAROAttention
    - self.prefetch = False
    - self.sparse_mask. (in python wrapper., updated each iter.)
--- 
transformer_block.norm1 -> LayerNormWithPermute
    - self.FHW
    - self.permute_order (update each iter.)
transformer_block.norm2 -> LayerNormWithInversePermute
    - self.FHW
    - self.inv_permute_order
apply_rotary_embedding 
    - self.FHW
---
'''


def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device="cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = args.ckpt if args.ckpt is not None else "/home/models/Wan2.1-T2V-14B-Diffusers"
    model_id = ckpt_path
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 5.0 # 5.0 for 720P, 3.0 for 480P
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)

    pipe = WanPipeline.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        vae=vae
    )
    pipe.scheduler = scheduler
    pipe = pipe.to(device)

    # INFO: if memory intense
    # pipe.enable_model_cpu_offload()
    # pipe.vae.enable_tiling()

    # read the promts
    prompt_path = args.prompt if args.prompt is not None else "./prompts.txt"
    prompts = []
    with open(prompt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            prompts.append(line.strip())

    for i, prompt in enumerate(prompts):        
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=args.num_sampling_steps, # 50
            num_frames=81,
            guidance_scale=args.cfg_scale,
            height=720,  # 720P
            width=1280,  # 720P
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).frames[0]
        
        save_path = os.path.join(args.log, "generated_videos")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        outpath = os.path.join(save_path, f"output_{i}.mp4")
        export_to_video(video, outpath, fps=8)
        print(f"Export video to {outpath}")
        
        # total_time = 0.0
        # for b in range(40):
        #     stats = pipe.transformer.blocks[b].attn1.processor.get_time_stats()
        #     total_time += stats['total_ms']
        # print(f"Total attention time: {total_time:.2f} ms")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default='./log')
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--num-sampling-steps", type=int, default=30)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--prefetch", action="store_true")
    args = parser.parse_args()
    main(args)
