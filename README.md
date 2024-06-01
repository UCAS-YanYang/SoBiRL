# SoBiRL-Implementation
Implementation of "Bilevel Reinforcement Learning via the Development of Hyper-gradient without Lower-Level Convexity".

## Dependencies

- Ubuntu 20.04 
- Python 3.9 
- PyTorch 1.13.0 
- CUDA 11.6
- imitation 1.0.0
- stable_baselines3 2.3.2
- wandb 0.16.0 (We heavily rely on Weights & Biases for visualization and monitoring)


## Get Started

You can create a conda environment by simply running the following commands.

```bash
$ conda create -n SoBiRL python=3.9
$ conda activate SoBiRL
$ pip install imitation
$ pip freeze | grep -E 'torch|nvidia' | xargs pip uninstall -y
$ pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install stable-baselines3
$ pip install wandb
```

To start training, run

```bash
$ conda activate SoBiRL
```