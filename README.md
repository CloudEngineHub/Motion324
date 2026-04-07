# 
<h2 align="center"> <a href="https://motion3-to-4.github.io/">Motion 3-to-4: 3D Motion Reconstruction for 4D Synthesis</a>
</h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2503.24391-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2601.14253) 
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://motion3-to-4.github.io/) 
[![HuggingFace](https://img.shields.io/badge/🤗%20Dataset-River--Chen%2FMotion324-yellow)](https://huggingface.co/datasets/River-Chen/Motion324) 

[Hongyuan Chen](https://hyuanChen.github.io/),
[Xingyu Chen](https://rover-xingyu.github.io/),
[Youjia Zhang](https://youjiazhang.github.io/),
[Zexiang Xu](https://zexiangxu.github.io/),
[Anpei Chen](https://apchenstu.github.io/),
</h5>

<div align="center">

Motion 3-to-4 reconstructs 3D motion from videos for 4D synthesis in a <b>feedforward</b> manner within seconds.

</div>

<br>

<div align="center">
<img src="https://github.com/user-attachments/assets/1b21f991-501c-440d-9504-0ea35395bdfe" width="90%">
</div>


## Quick Start

For users who want to quickly try the inference:

```bash
git clone https://github.com/Inception3D/Motion324.git
cd Motion324

# 1. Setup environment
conda create -n Motion324 python=3.11
conda activate Motion324
pip install -r requirements.txt
# Install Hunyuan3D-2.0 components(optional)
cd scripts/hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../../..
cd scripts/hy3dgen/texgen/differentiable_renderer
python3 setup.py install
cd ../../../..

# 2. Download pre-trained checkpoints and place in experiments/checkpoints/

# 3. Run inference
chmod +x ./scripts/4D_from_existing.sh
./scripts/4D_from_existing.sh ./examples/chili.glb ./examples/chili.mp4 ./examples/chili

# Hunyuan needed
chmod +x ./scripts/4D_from_video.sh
./scripts/4D_from_video.sh ./examples/tiger.mp4

# 4. Render results
# Render output from 4D_from_existing.sh:
python scripts/render_results.py -- --output_dir ./examples/chili
# Render output from 4D_from_video.sh:
python scripts/render_results.py -- --output_dir ./examples/tiger_processed
```

## 1. Preparation

### Checkpoints

**Download**: Please download the pre-trained checkpoint from [here](https://huggingface.co/River-Chen/Motion324/tree/main) and place it in `experiments/checkpoints/`.

### Environment Details
#### Setup up base environment
```bash
conda create -n Motion324 python=3.11
conda activate Motion324
pip install -r requirements.txt
```
The code has been tested with Python 3.11 + Pytorch 2.4.1 + CUDA 12.4.

#### Setup Hunyuan3D-2.0 Components
```bash
# Install custom rasterizer
cd scripts/hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../../..

# Install differentiable renderer
cd scripts/hy3dgen/texgen/differentiable_renderer
python3 setup.py install
cd ../../../..
```
#### Setup Blender
Download and install Blender for 4D asset rendering.

Our results is rendered with [blender-4.0.0-linux-x64](https://download.blender.org/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz), using the scripts which is modified from [bpy-renderer](https://github.com/huanngzh/bpy-renderer).

`scripts/render_results.py` provides basic visualization of results. It supports two modes:

- **Batch mode** (default): scans all subdirectories under `examples/` and renders each one.
- **Single-directory mode**: pass `--output_dir` to render the output of a specific pipeline run.

Installation steps:
```bash
# Download Blender
wget https://download.blender.org/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz
tar -xf blender-4.0.0-linux-x64.tar.xz

# Add Blender to PATH (optional, or use full path in scripts)
export PATH=$PATH:$(pwd)/blender-4.0.0-linux-x64
```

Usage:
```bash
# Batch mode — render all results under examples/
python scripts/render_results.py
# Single-directory mode — render the output of 4D_from_existing.sh
python scripts/render_results.py -- --output_dir ./examples/output
# Single-directory mode — render the output of 4D_from_video.sh
python scripts/render_results.py -- --output_dir ./examples/tiger_processed/
```

The rendered video (`animation.mp4`) is saved inside the specified output directory.

**Note**: As we use [xformers](https://github.com/facebookresearch/xformers) `memory_efficient_attention` with [flash_attn](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1), the GPU device compute capability needs > 8.0. Otherwise, it would pop up an error. Check your GPU compute capability in [CUDA GPUs Page](https://developer.nvidia.com/cuda-gpus#compute).

### Dataset

The **Motion80 benchmark** and the **training dataset** is available [here](https://huggingface.co/datasets/River-Chen/Motion324/tree/main).

Update the dataset path in `configs/dyscene.yaml`:
```yaml
training:
  dataset_path: /path/to/your/dataset
  train_lst: /path/to/name_list
```

## 2. Training

Before training, you need to follow the instructions [here](https://docs.wandb.ai/guides/track/public-api-guide/#:~:text=You%20can%20generate%20an%20API,in%20the%20upper%20right%20corner.) to generate the Wandb key file for logging and save it in the `configs` folder as `api_keys.yaml`.

### Training Command

The default training uses `configs/dyscene.yaml`:

```bash
torchrun --nproc_per_node 8 --nnodes 1 --master_port 12344 train.py --config configs/dyscene.yaml
```

### Training Configuration

Key training parameters in `configs/dyscene.yaml`:
You can override any config parameter via command line:

```bash
torchrun --nproc_per_node 8 --nnodes 1 --master_port 12346 train.py --config configs/dyscene.yaml \
    training.batch_size_per_gpu=32
```

## 3. Inference
> We use `rembg` for simple background removal from videos.  
> However, we strongly recommend using [SAM2](https://github.com/facebookresearch/sam2) for best video background removal.  

### Generate 4D animation from a single video input

**Input**: Video file (`.mp4/.avi/.mov`) or image directory (use `./scripts/images2video.py` to convert images to video first)

**Output**: 
- Processed frames and mesh files in `{video_name}_processed/`
- Animation output in `{video_name}_processed/animation/` (FBX format)

**Example**:
```bash
chmod +x ./scripts/4D_from_video.sh
./scripts/4D_from_video.sh ./examples/tiger.mp4
```

### Reconstruct 4D from an existing mesh and video

**Inputs**:
- `data_dir`: Mesh file (`.glb` or `.fbx`) - FBX files will be automatically converted to GLB
- `video_path`: Video file (`.mp4/.avi/.mov`) or image directory
- `output_dir`: Output directory for results

**Output**: 
- Animated mesh files (GLB format) in the specified output directory
- Segmented videos if segmentation is enabled

**Example**:
```bash
chmod +x ./scripts/4D_from_existing.sh
./scripts/4D_from_existing.sh ./examples/chili.glb ./examples/chili.mp4 ./examples/output
```

## 4. Evaluation

For a fair comparison across methods, it is recommended to initialize with the **same mesh**, typically generated from an image at 512×512 resolution.

### Mesh Geometry Metrics (Chamfer Distance, F-score)

For a quick test, use `batch_eval_all.sh` to run both pipelines and evaluate all samples:

```bash
# Run both GT and video pipelines on the default benchmark
bash scripts/batch_eval_all.sh --release_dir /path/to/release_dir

# Run only the GT pipeline (inference_with_gt.py + evaluation)
bash scripts/batch_eval_all.sh --gt_only

# Run only the video pipeline (4D_from_video.sh + evaluation)
bash scripts/batch_eval_all.sh --video_only

# Skip inference and only run evaluation on existing outputs
bash scripts/batch_eval_all.sh --skip_inference
```

The default benchmark path is `examples/release_80`. Sample lists are read from `dataset/short_videos.txt` and `dataset/long_videos.txt`.

This compares the predicted mesh (GLB/FBX file or directory of `frame_*.npy` files) with the ground-truth point cloud. 
It outputs metrics such as Chamfer Distance and F-score.

### Video Metrics (FVD, LPIPS, DreamSim, CLIP Loss)

Re-render both the generated GLB/FBX animation and the original GLB/FBX animation for comparison, **all with a white background and at 512×512 resolution**. Other rendering settings (such as lighting and materials) have little impact on the final scores, just ensure the background is white.

After rendering, evaluate using `evaluation.py`:

```bash
python ./evaluation/evaluation.py \
--gt_paths /paths/to/gt_videos.mp4 \
--result_paths /paths/to/rendered_results_videos.mp4
```

This script compares the rendered videos to the ground-truth, and reports metrics including FVD, LPIPS, DreamSim, and CLIP Loss.

## 5. Citation 

If you find this work useful in your research, please consider citing:

```bibtex
@article{chen2026motion3to4,
    title={Motion 3-to-4: 3D Motion Reconstruction for 4D Synthesis},
    author={Hongyuan, Chen and Xingyu, Chen and Youjia Zhang, and Zexiang, Xu and Anpei, Chen},
    journal={arXiv preprint arXiv:2601.14253},
    year={2026}
}
```
## 6. Acknowledgments
- [LVSM](https://github.com/Haian-Jin/LVSM) (for code architecture reference)
- [V2M4](https://github.com/WindVChen/V2M4), [AnimateAnyMesh](https://github.com/JarrentWu1031/AnimateAnyMesh) (for code reference)
- [bpy-renderer](https://github.com/huanngzh/bpy-renderer) (for rendering results)
- [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) (for 3D generation)
  
## 7. License

This project is licensed under the [CC BY-NC-SA 4.0 License](LICENSE.md) - see the LICENSE.md file for details.
