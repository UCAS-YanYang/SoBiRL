# SAC oracle adapted from https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[2], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *(4,84,84))).shape[1]


        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_q = layer_init(nn.Linear(512, envs.action_space.n))

    def forward(self, x):
        if x.shape[1] > 20:
            x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv(x / 255.0))
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *(4,84,84))).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_logits = layer_init(nn.Linear(512, envs.action_space.n))

    def forward(self, x):
        if x.shape[1] > 20:
            x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        if x.shape[1] > 20:
            x = x.permute(0, 3, 1, 2)
        logits = self(x / 255.0)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1) # = torch.log(action_probs)
        return action, log_prob, action_probs

class SAC():
    def __init__(self,reward_model,env,args):
        self.reward_model = reward_model
        self.envs = env
        self.args =args
        self.envs.seed(seed=args.seed)
        self._initial_sac()

    def _initial_sac(self):
        self.alg_name = "SAC"
        args = self.args
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        if args.track:
            import wandb
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        self.writer = SummaryWriter(f"./RL_log/runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # network setup
        self.actor = Actor(self.envs).to(self.device)
        self.qf1 = SoftQNetwork(self.envs).to(self.device)
        self.qf2 = SoftQNetwork(self.envs).to(self.device)
        self.qf1_target = SoftQNetwork(self.envs).to(self.device)
        self.qf2_target = SoftQNetwork(self.envs).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr, eps=1e-4)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr, eps=1e-4)

        # Automatic entropy tuning
        if args.autotune:
            self.target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(self.envs.action_space.n))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr, eps=1e-4)
        else:
            self.alpha = args.alpha

        # Replay buffer
        self.rb = ReplayBuffer(
            args.buffer_size,
            self.envs.observation_space,
            self.envs.action_space,
            self.device,
            handle_timeout_termination=False,
        )

        # inner_iteration conter        
        self.inner_count = 0

        # global_timestep conter
        self.global_step = 0

    def set_env(self, venv):
        self.envs = venv
    
    def get_env(self):
        return self.envs
    
    def get_actor(self):
        return self.actor

    def train(self, total_timesteps):
        """Train the agent using the reward function specified during instantiation.

        Args:
            steps: number of environment timesteps to train for
            **kwargs: other keyword arguments to pass to BaseAlgorithm.train()

        Raises:
            RuntimeError: Transitions left in `self.buffering_wrapper`; call
                `self.sample` first to clear them.
        """

        args = self.args
        args.num_envs = self.envs.num_envs
        args.step_iterations = total_timesteps

        device = self.device

        start_time = time.time()

        episodic_returns = np.zeros(self.envs.num_envs)
        episodic_lengths = np.zeros(self.envs.num_envs)

        # TRY NOT TO MODIFY: start the game
        obs = self.envs.reset()
        for step in range(args.step_iterations):
            self.global_step += 1

            if args.lr_anneal>0:
                frac = 1.0 - (self.global_step - 1.0) / args.lr_anneal
                q_lrnow = frac * args.q_lr
                policy_lrnow = frac * args.policy_lr
                self.actor_optimizer.param_groups[0]["lr"] = policy_lrnow
                self.q_optimizer.param_groups[0]["lr"] = q_lrnow
                if args.autotune:
                    self.a_optimizer.param_groups[0]["lr"] = q_lrnow

            # ALGO LOGIC: put action logic here
            if self.global_step < args.learning_starts:
                actions = np.array([self.envs.action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                actions, _, _ = self.actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, next_dones, infos = self.envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for env_id, info in enumerate(infos):
                if "episode" in info:
                    episodic_returns[env_id] += info['episode']['r']
                    episodic_lengths[env_id] += info['episode']['l']
                    if info["lives"] == 0:
                        print(f"global_step={self.global_step}, episodic_return={episodic_returns[env_id]}")
                        self.writer.add_scalar("charts/episodic_return", episodic_returns[env_id], self.global_step)
                        self.writer.add_scalar("charts/episodic_length", episodic_lengths[env_id], self.global_step)
                        episodic_returns[env_id] = 0
                        episodic_lengths[env_id] = 0
                if info["TimeLimit.truncated"]:
                    next_dones[env_id] = 0

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, done in enumerate(next_dones):
                if done:
                    # if trunc, real_next_obs is the obs reset by env, instead of the true s^\prime
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            self.rb.add(obs, real_next_obs, actions, rewards, next_dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if self.global_step > args.learning_starts:
                if step % args.update_frequency == 0:
                    data = self.rb.sample(args.batch_size)
                    # CRITIC training
                    with torch.no_grad():
                        _, next_state_log_pi, next_state_action_probs = self.actor.get_action(data.next_observations)
                        qf1_next_target = self.qf1_target(data.next_observations)
                        qf2_next_target = self.qf2_target(data.next_observations)
                        # we can use the action probabilities instead of MC sampling to estimate the expectation
                        min_qf_next_target = next_state_action_probs * (
                            torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                        )
                        # adapt Q-target for discrete Q-function
                        min_qf_next_target = min_qf_next_target.sum(dim=1)

                        # compute the target values
                        one_hot_actions = torch.nn.functional.one_hot(data.actions.squeeze(), num_classes=9)
                        cur_rewards = self.reward_model(data.observations,one_hot_actions,data.next_observations,data.dones).unsqueeze(1)
                        cur_rewards = torch.clamp(cur_rewards, min=-1, max=1)
                        next_q_value = cur_rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)


                    # use Q-values only for the taken actions
                    qf1_values = self.qf1(data.observations)
                    qf2_values = self.qf2(data.observations)
                    qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
                    qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    self.q_optimizer.zero_grad()
                    qf_loss.backward()
                    self.q_optimizer.step()

                    # ACTOR training
                    _, log_pi, action_probs = self.actor.get_action(data.observations)
                    with torch.no_grad():
                        qf1_values = self.qf1(data.observations)
                        qf2_values = self.qf2(data.observations)
                        min_qf_values = torch.min(qf1_values, qf2_values)
                    # no need for reparameterization, the expectation can be calculated for discrete actions
                    actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    if args.autotune:
                        # re-use action probabilities for temperature loss
                        alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())).mean()

                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()

                # update the target networks
                if self.global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                # monitor by wandb
                if step % 100 == 0:
                    self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), self.global_step)
                    self.writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), self.global_step)
                    self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), self.global_step)
                    self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), self.global_step)
                    self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, self.global_step)
                    self.writer.add_scalar("losses/actor_loss", actor_loss.item(), self.global_step)
                    self.writer.add_scalar("losses/alpha", self.alpha, self.global_step)
                    self.writer.add_scalar("losses/q_lrnow", q_lrnow, self.global_step)
                    self.writer.add_scalar("losses/policy_lrnow", policy_lrnow, self.global_step)
                    print("SPS:", int(step / (time.time() - start_time)))
                    self.writer.add_scalar("charts/SPS", int(step / (time.time() - start_time)), self.global_step)
                    if args.autotune:
                        self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), self.global_step)







