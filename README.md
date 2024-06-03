# SoBiRL-Implementation
Code repository of "Bilevel Reinforcement Learning via the Development of Hyper-gradient without Lower-Level Convexity"

## Dependencies

- Ubuntu 20.04 
- Python 3.9 
- PyTorch 1.13.0 
- CUDA 11.6
- imitation 1.0.0
- stable_baselines 3 2.3.2
- wandb 0.17.0 (We heavily rely on Weights & Biases for visualization and monitoring)


## Get Started

You can create a conda environment by simply running the following commands.

```bash
$ conda create -n SoBiRL python=3.9
$ conda activate SoBiRL
$ pip install imitation
$ pip freeze | grep -E 'torch|nvidia' | xargs pip uninstall -y
$ pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install stable-baselines3
$ pip install -r requirements/requirements-atari.txt
$ pip install wandb
```

To start training,

+ step into `stable_baseliens3.common.monitor.py` and annotate the code in line 96-109:

  ```python
  # if terminated or truncated:
  #     self.needs_reset = True
  #     ep_rew = sum(self.rewards)
  #     ep_len = len(self.rewards)
  #     ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
  #     for key in self.info_keywords:
  #         ep_info[key] = info[key]
  #     self.episode_returns.append(ep_rew)
  #     self.episode_lengths.append(ep_len)
  #     self.episode_times.append(time.time() - self.t_start)
  #     ep_info.update(self.current_reset_info)
  #     if self.results_writer:
  #         self.results_writer.write_row(ep_info)
  #     info["episode"] = ep_info
  ```

+ run the following command

  ```bash
  $ conda activate SoBiRL
  $ python atari_SoBiRL.py
  ```

  
