#!/bin/bash
set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# CogVideoX configuration
SCRIPT="cogvideox_usp_example.py"
MODEL_ID="/home/models/CogVideoX1.5"
INFERENCE_STEP=50

mkdir -p ./results

# CogVideoX specific task args
TASK_ARGS="--height 768 --width 1360 --num_frames 17 --guidance_scale 6.0"

# CogVideoX parallel configuration
N_GPUS=8
PARALLEL_ARGS="--ulysses_degree 2 --ring_degree 2"
CFG_ARGS="--use_cfg_parallel"

# Uncomment and modify these as needed
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
# OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
ENABLE_TILING="--enable_tiling"
# COMPILE_FLAG="--use_torch_compile"

torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
--prompt "A serene art gallery with polished wooden floors and soft, ambient lighting showcases an array of captivating artworks. The camera pans across vibrant abstract paintings, intricate sculptures, and detailed portraits, each piece telling its own unique story. Visitors, dressed in elegant attire, move gracefully through the space, pausing to admire the masterpieces. The gallery's high ceilings and large windows allow natural light to flood in, enhancing the colors and textures of the art. A close-up reveals the delicate brushstrokes of a painting, while another shot captures the intricate details of a marble sculpture. The atmosphere is one of quiet reverence and inspiration, as art enthusiasts immerse themselves in the beauty and creativity surrounding them." \
$CFG_ARGS \
$PARALLLEL_VAE \
$ENABLE_TILING \
$COMPILE_FLAG
