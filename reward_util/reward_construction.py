# Reward model parameterized by a deep network

import collections
from typing import Dict, Iterable, Optional, Type, Union, Any

import torch as th
from torch import nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common import preprocessing

from imitation.util.networks import SqueezeLayer
from imitation.util import networks, util
from imitation.rewards.reward_nets import RewardNet

def cnn_transpose(tens: th.Tensor) -> th.Tensor:
    """Transpose a (b,h,w,c)-formatted tensor to (b,c,h,w) format."""
    if len(tens.shape) == 4:
        return th.permute(tens, (0, 3, 1, 2))
    else:
        raise ValueError(
            f"Invalid input: len(tens.shape) = {len(tens.shape)} != 4.",
        )

def build_cnn(
    in_channels: int,
    hid_channels: Iterable[int],
    out_size: int = 1,
    name: Optional[str] = None,
    activation: Type[nn.Module] = nn.ReLU,
    kernel_size: int = 3,
    stride: int = 1,
    padding: Union[int, str] = "same",
    dropout_prob: float = 0.0,
    squeeze_output: bool = False,
) -> nn.Module:
    """Constructs a Torch CNN.

    Args:
        in_channels: number of channels of individual inputs; input to the CNN will have
            shape (batch_size, in_size, in_height, in_width).
        hid_channels: number of channels of hidden layers. If this is an empty iterable,
            then we build a linear function approximator.
        out_size: size of output vector.
        name: Name to use as a prefix for the layers ID.
        activation: activation to apply after hidden layers.
        kernel_size: size of convolutional kernels.
        stride: stride of convolutional kernels.
        padding: padding of convolutional kernels.
        dropout_prob: Dropout probability to use after each hidden layer. If 0,
            no dropout layers are added to the network.
        squeeze_output: if out_size=1, then squeeze_input=True ensures that CNN
            output is of size (B,) instead of (B,1).

    Returns:
        nn.Module: a CNN mapping from inputs of size (batch_size, in_size, in_height,
            in_width) to (batch_size, out_size), unless out_size=1 and
            squeeze_output=True, in which case the output is of size (batch_size, ).

    Raises:
        ValueError: if squeeze_output was supplied with out_size!=1.
    """
    layers: Dict[str, nn.Module] = {}

    if name is None:
        prefix = ""
    else:
        prefix = f"{name}_"

    prev_channels = in_channels
    for i, n_channels in enumerate(hid_channels):
        layers[f"{prefix}conv{i}"] = nn.Conv2d(
            prev_channels,
            n_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        prev_channels = n_channels
        if activation:
            layers[f"{prefix}act{i}"] = activation()
        if dropout_prob > 0.0:
            layers[f"{prefix}dropout{i}"] = nn.Dropout(dropout_prob)

    # final dense layer
    layers[f"{prefix}avg_pool"] = nn.AdaptiveAvgPool2d(1)
    layers[f"{prefix}flatten"] = nn.Flatten()
    layers[f"{prefix}dense_final"] = nn.Linear(prev_channels, out_size)

    if squeeze_output:
        if out_size != 1:
            raise ValueError("squeeze_output is only applicable when out_size=1")
        layers[f"{prefix}squeeze"] = SqueezeLayer()

    model = nn.Sequential(collections.OrderedDict(layers))
    return model


class CusCnnRewardNet(RewardNet):
    """CNN that takes as input the state, action, next state and done flag.

    Inputs are boosted to tensors with channel, height, and width dimensions, and then
    concatenated. Image inputs are assumed to be in (h,w,c) format, unless the argument
    hwc_format=False is passed in. Each input can be enabled or disabled by the `use_*`
    constructor keyword arguments, but either `use_state` or `use_next_state` must be
    True.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        hwc_format: bool = True,
        **kwargs,
    ):
        """Builds reward CNN.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: Should the current state be included as an input to the CNN?
            use_action: Should the current action be included as an input to the CNN?
            use_next_state: Should the next state be included as an input to the CNN?
            use_done: Should the "done" flag be included as an input to the CNN?
            hwc_format: Are image inputs in (h,w,c) format (True), or (c,h,w) (False)?
                If hwc_format is False, image inputs are not transposed.
            kwargs: Passed straight through to `build_cnn`.

        Raises:
            ValueError: if observation or action space is not easily massaged into a
                CNN input.
        """
        super().__init__(observation_space, action_space)
        self.use_state = use_state
        self.use_action = use_action
        self.use_next_state = use_next_state
        self.use_done = use_done
        self.hwc_format = hwc_format

        if not (self.use_state or self.use_next_state):
            raise ValueError("CnnRewardNet must take current or next state as input.")

        if not preprocessing.is_image_space(observation_space):
            raise ValueError(
                "CnnRewardNet requires observations to be images.",
            )
        assert isinstance(observation_space, spaces.Box)  # Note: hint to mypy

        if self.use_action and not isinstance(action_space, spaces.Discrete):
            raise ValueError(
                "CnnRewardNet can only use Discrete action spaces.",
            )

        input_size = 0
        output_size = 1

        if self.use_state:
            input_size += self.get_num_channels_obs(observation_space)

        if self.use_action:
            assert isinstance(action_space, spaces.Discrete)  # Note: hint to mypy
            output_size = int(action_space.n)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=7, stride=3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=16 * 7 * 7, out_features=64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(in_features=64, out_features=output_size)
        )

        self.squee_layer = SqueezeLayer()

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        hyper_estimate: bool = False,
    ) -> th.Tensor:
        """Computes rewardNet value on input state, action, next_state, and done flag.

        Takes inputs that will be used, transposes image states to (c,h,w) format if
        needed, reshapes inputs to have compatible dimensions, concatenates them, and
        inputs them into the CNN.

        Args:
            state: current state.
            action: current action.
            next_state: next state.
            done: flag for whether the episode is over.

        Returns:
            th.Tensor: reward of the transition.
        """
        inputs = []

        state_ = cnn_transpose(state) if self.hwc_format else state
        inputs.append(state_)

        inputs_concat = th.cat(inputs, dim=1)

        x = self.conv_layers(inputs_concat/255)
        x = th.flatten(x, 1) 
        x = self.fc_layers(x)


        if hyper_estimate:
            return x

        rewards = th.sum(x * action, dim=1)

        return rewards
    
    def get_num_channels_obs(self, space: spaces.Box) -> int:
        """Gets number of channels for the observation."""
        return space.shape[-1] if self.hwc_format else space.shape[0]
