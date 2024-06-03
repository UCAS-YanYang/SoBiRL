import torch as th
import gymnasium as gym
import numpy as np

from seals.util import AutoResetWrapper

from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

import RLHF_util.RLHF as RLHF
from sac_subroutine import SAC
from reward_util.reward_construction import CusCnnRewardNet
from dataclasses import dataclass
import tyro
from alg_config.config_loader import load_config
import os
import time


@dataclass
class Args:
    bialg: str = 'SoBiRL'             
    """bilevel RL algorithm"""
    env_id: str = "BeamRiderNoFrameskip-v4"
    """the enviroment id"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Implementation of SoBiRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""  # smaller than in original paper but evaluation is done only for 100k steps anyway
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2e4
    """timestep to start learning"""
    policy_lr: float = 1e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-4
    """the learning rate of the Q network network optimizer"""
    reward_lr: float = 3e-4
    """the learning rate of the reward network network optimizer"""
    lr_anneal: float = 8e6
    """the learning rate decays linearly"""
    update_frequency: int = 4
    """the frequency of training updates"""
    target_network_frequency: int = 8000
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""
    alternate: int = 10000
    """optimization alternating frequency between the agent networks and the reward networks"""

    # to be filled in runtime
    total_timesteps: int = 0
    """total timesteps of the experiments"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # debug
    debug: bool = False
    """debug: adjust the args manully"""
    test: bool = False

args = tyro.cli(Args)
if args.debug:
    pass
else:
    args.config_path = "alg_config/" + args.bialg + "_config.yaml"   
    # "alg_config/SoBiRL_config.yaml" 
    """algorithm config path"""
    args = load_config(args.config_path,args)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
rng = np.random.default_rng()


def construct_env():
    atari_env = gym.make(args.env_id,render_mode="rgb_array")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.capture_video:
        atari_env = gym.wrappers.RecordVideo(atari_env, f"RL_log/videos/{run_name}")
    atari_env = gym.wrappers.RecordEpisodeStatistics(atari_env)
    preprocessed_env = AtariWrapper(atari_env)
    endless_env = AutoResetWrapper(preprocessed_env)
    return endless_env


# For real training, you will want a vectorized environment with 8 environments in parallel.
# This can be done by passing in n_envs=8 as an argument to make_vec_env.
# The seed needs to be set to 1 for reproducibility and also to avoid win32
# np.random.randint high bound error.
venv = make_vec_env(construct_env, seed=1)
venv = VecFrameStack(venv, n_stack=4)


# reward net to prediect rewards
reward_net = CusCnnRewardNet(
    venv.observation_space,
    venv.action_space,
).to(device)

agent = SAC(
    reward_model=reward_net,
    env=venv,
    args=args,
)

# randomly select fragment pairs
fragmenter = RLHF.RandomFragmenter(warning_threshold=0, rng=rng)
# generate preference based on the ground-truth rewards
gatherer = RLHF.SyntheticGatherer(rng=rng)

# predict preference probobility based on the reward net
preference_model = RLHF.SoBiPreferenceModel(reward_net,agent)

# reward trainer
reward_trainer = RLHF.SoBiRewardTrainer(
    preference_model=preference_model,
    loss=RLHF.SoBiRewardLoss(),
    epochs=1,
    rng=rng,
    batch_size=64,
    lr = args.reward_lr
)

trajectory_generator = RLHF.AgentTrainer(
    algorithm=agent,
    reward_fn=reward_net,
    venv=venv,
    exploration_frac=0.0,
    rng=rng,
)

pref_comparisons = RLHF.PreferenceComparisons(
    trajectory_generator,
    reward_net,
    num_iterations=2000,
    fragmenter=fragmenter,
    preference_gatherer=gatherer,
    reward_trainer=reward_trainer,
    comparison_queue_size=3000,
    fragment_length=25,
    transition_oversampling=1,
    initial_comparison_frac=0.07,
    allow_variable_horizon=False,
    initial_epoch_multiplier=512,
    alternate=args.alternate
)

# Begin training
pref_comparisons.train(
    total_timesteps=20000000,
    total_comparisons=9000,
)