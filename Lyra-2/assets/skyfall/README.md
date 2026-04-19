---
license: apache-2.0
task_categories:
- image-to-3d
---

# The datasets for Skyfall-GS

[Project Page](https://skyfall-gs.jayinnn.dev/) | [Paper](https://arxiv.org/abs/2510.15869) | [GitHub](https://github.com/jayin92/skyfall-gs)

Skyfall-GS is a hybrid framework that synthesizes immersive, city-block scale 3D urban scenes by combining satellite reconstruction with diffusion refinement. This repository contains the JAX and NYC datasets used for training and evaluation.

## Dataset Structure

According to the [official GitHub documentation](https://github.com/jayin92/skyfall-gs), the datasets should be organized in the `data/` directory as follows:

```
data/
├── datasets_JAX/
│   ├── JAX_004
│   ├── JAX_068
│   └── ...
└── datasets_NYC/
    ├── NYC_004
    ├── NYC_010
    └── ...
```

Each individual scene directory (e.g., `JAX_068`) contains the following structure:

```
your_dataset/
├── images/                    # RGB images
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── masks/                   # Binary masks for valid pixels (optional)
│   ├── *.npy               # NumPy format
│   ├── *.png               # PNG format
│   └── ...
├── transforms_train.json      # Training camera parameters
├── transforms_test.json       # Testing camera parameters (optional)
└── points3D.txt              # 3D point cloud
```

## Citation

If you find this work or the datasets useful, please consider citing:

```bibtex
@article{lee2025SkyfallGS,
  title = {{Skyfall-GS}: Synthesizing Immersive {3D} Urban Scenes from Satellite Imagery},
  author = {Jie-Ying Lee and Yi-Ruei Liu and Shr-Ruei Tsai and Wei-Cheng Chang and Chung-Ho Wu and Jiewen Chan and Zhenjun Zhao and Chieh Hubert Lin and Yu-Lun Liu},
  journal = {arXiv preprint},
  year = {2025},
  eprint = {2510.15869},
  archivePrefix = {arXiv}
}
```