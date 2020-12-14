from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union
import math

import gym
import torch as th
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.utils import get_device

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super(BaseFeaturesExtractor, self).__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        raise NotImplementedError()


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super(FlattenExtractor, self).__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(NatureCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        '''
        assert is_image_space(observation_space), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        '''
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            #nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=1, padding=0),
            nn.ReLU(),
            #nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def create_mlp(
    input_dim: int, output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module] = nn.ReLU, squash_output: bool = False
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    Adapted from Stable Baselines.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device:
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ):
        super(MlpExtractor, self).__init__()
        device = get_device(device)
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim

        # Iterate through the shared layers and build the shared parts of the network
        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):  # Check that this is a shared layer
                layer_size = layer
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(last_layer_dim_shared, layer_size))
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer_size
            else:
                if not isinstance(layer, dict): continue
                #assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)


def get_actor_critic_arch(net_arch: Union[List[int], Dict[str, List[int]]]) -> Tuple[List[int], List[int]]:
    """
    Get the actor and critic network architectures for off-policy actor-critic algorithms (SAC, TD3, DDPG).

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers,
    which can be different for the actor and the critic.
    It is assumed to be a list of ints or a dict.

    1. If it is a list, actor and critic networks will have the same architecture.
        The architecture is represented by a list of integers (of arbitrary length (zero allowed))
        each specifying the number of units per layer.
       If the number of ints is zero, the network will be linear.
    2. If it is a dict,  it should have the following structure:
       ``dict(qf=[<critic network architecture>], pi=[<actor network architecture>])``.
       where the network architecture is a list as described in 1.

    For example, to have actor and critic that share the same network architecture,
    you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).

    If you want a different architecture for the actor and the critic,
    then you can specify ``net_arch=dict(qf=[400, 300], pi=[64, 64])``.

    .. note::
        Compared to their on-policy counterparts, no shared layers (other than the features extractor)
        between the actor and the critic are allowed (to prevent issues with target networks).

    :param net_arch: The specification of the actor and critic networks.
        See above for details on its formatting.
    :return: The network architectures for the actor and the critic
    """
    if isinstance(net_arch, list):
        actor_arch, critic_arch = net_arch, net_arch
    else:
        assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
    return actor_arch, critic_arch

def lstm(input_tensor, mask_tensor, cell_state_hidden, scope, n_hidden, init_scale=1.0, layer_norm=False):
    """
    Creates an Long Short Term Memory (LSTM) cell for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the LSTM cell
    :param mask_tensor: (TensorFlow Tensor) The mask tensor for the LSTM cell
    :param cell_state_hidden: (TensorFlow Tensor) The state tensor for the LSTM cell
    :param scope: (str) The TensorFlow variable scope
    :param n_hidden: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :param layer_norm: (bool) Whether to apply Layer Normalization or not
    :return: (TensorFlow Tensor) LSTM cell
    """
    _, n_input = [v.value for v in input_tensor[0].get_shape()]
    with tf.variable_scope(scope):
        weight_x = tf.get_variable("wx", [n_input, n_hidden * 4], initializer=ortho_init(init_scale))
        weight_h = tf.get_variable("wh", [n_hidden, n_hidden * 4], initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", [n_hidden * 4], initializer=tf.constant_initializer(0.0))

        if layer_norm:
            # Gain and bias of layer norm
            gain_x = tf.get_variable("gx", [n_hidden * 4], initializer=tf.constant_initializer(1.0))
            bias_x = tf.get_variable("bx", [n_hidden * 4], initializer=tf.constant_initializer(0.0))

            gain_h = tf.get_variable("gh", [n_hidden * 4], initializer=tf.constant_initializer(1.0))
            bias_h = tf.get_variable("bh", [n_hidden * 4], initializer=tf.constant_initializer(0.0))

            gain_c = tf.get_variable("gc", [n_hidden], initializer=tf.constant_initializer(1.0))
            bias_c = tf.get_variable("bc", [n_hidden], initializer=tf.constant_initializer(0.0))

    cell_state, hidden = tf.split(axis=1, num_or_size_splits=2, value=cell_state_hidden)
    for idx, (_input, mask) in enumerate(zip(input_tensor, mask_tensor)):
        cell_state = cell_state * (1 - mask)
        hidden = hidden * (1 - mask)
        if layer_norm:
            gates = _ln(tf.matmul(_input, weight_x), gain_x, bias_x) \
                    + _ln(tf.matmul(hidden, weight_h), gain_h, bias_h) + bias
        else:
            gates = tf.matmul(_input, weight_x) + tf.matmul(hidden, weight_h) + bias
        in_gate, forget_gate, out_gate, cell_candidate = tf.split(axis=1, num_or_size_splits=4, value=gates)
        in_gate = tf.nn.sigmoid(in_gate)
        forget_gate = tf.nn.sigmoid(forget_gate)
        out_gate = tf.nn.sigmoid(out_gate)
        cell_candidate = tf.tanh(cell_candidate)
        cell_state = forget_gate * cell_state + in_gate * cell_candidate
        if layer_norm:
            hidden = out_gate * tf.tanh(_ln(cell_state, gain_c, bias_c))
        else:
            hidden = out_gate * tf.tanh(cell_state)
        input_tensor[idx] = hidden
    cell_state_hidden = tf.concat(axis=1, values=[cell_state, hidden])
    return input_tensor, cell_state_hidden

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    #def __init__(self,in_dim,activation):
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        #self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(th.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  th.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = th.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return attention[:, None, :,:,]

class AttnCnn(nn.Module):
    """
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(AttnCnn, self).__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.attn = Self_Attn(observation_space.shape[0])
        self.features_dim = features_dim
        n_input_channels = 1
        self.cnn = nn.Sequential(
            #nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.Conv2d(n_input_channels, 8, kernel_size=32, stride=2, padding=0),
            nn.ReLU(),
            #nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.Conv2d(8, 16, kernel_size=16, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(self.attn(th.as_tensor(observation_space.sample()[None])).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(self.attn(observations)))


class CnnAttn(nn.Module):
    """
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CnnAttn, self).__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.features_dim = features_dim
        n_input_channels=observation_space.shape[0]
        self.cnn = nn.Sequential(
            #nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=1, padding=0),
            nn.ReLU(),
            #nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            in_attn = (self.cnn(th.as_tensor(observation_space.sample()[None])).float()).shape[1]
            
        self.attn = Self_Attn(in_attn)
        self.flatten = (nn.Flatten())

       # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = (self.flatten(self.attn(self.cnn(th.as_tensor(observation_space.sample()[None])).float()))).shape[1]
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(n_flatten, features_dim), nn.ReLU())

        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.attn(self.cnn(observations)))


class AttnCnnMin(nn.Module):
    """ Self attention Layer"""
    #def __init__(self,in_dim,activation):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(AttnCnnMin,self).__init__()
        self.chanel_in = observation_space.shape[0]
        #self.activation = activation
        self.features_dim = features_dim

        self.query_conv = nn.Conv2d(in_channels = self.chanel_in , out_channels = self.chanel_in//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = self.chanel_in , out_channels = self.chanel_in//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = self.chanel_in , out_channels = self.chanel_in , kernel_size= 1)
        self.gamma = nn.Parameter(th.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #

        self.flatten = nn.Flatten()
        n_flatten = 65536
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X features_dim)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  th.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = th.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x

        return self.linear(self.flatten(attention))

class NewCNN1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(NewCNN1, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            #nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.Conv2d(n_input_channels, 32, kernel_size=6, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            #nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
            
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class DropoutCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(DropoutCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            #nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout2d(.3),
            #nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout2d(.3),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout2d(.3),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data)
    bias_init(module.bias.data)
    return module


class NNBase(nn.Module):

    def __init__(self, hidden_size):
        super(NNBase, self).__init__()
        recurrent = False
        recurrent_input_size = -1
        self._hidden_size = hidden_size
        self._recurrent = recurrent
        # for multi-headed agents
        self.active_agent = 0

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size

        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []

            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = th.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs


class FractalNet(NNBase):
    def __init__(self,num_inputs, recurrent=False, hidden_size=64,
                 map_width=16, n_conv_recs=2, n_recs=3,
                 intra_shr=False, inter_shr=False,
                 num_actions=19,
                 n_player_actions=4,
                 rule='extend',
                 in_w=1, in_h=1, out_w=1, out_h=1, n_chan=512, prebuild=None,
                 val_kern=3):
        super(FractalNet, self).__init__(hidden_size)
        num_inputs = num_inputs.shape[0]
        self.features_dim = num_inputs
        self.map_width = map_width
       #self.bn = nn.BatchNorm2d(num_inputs)
        # We can stack multiple Fractal Blocks
       #self.block_chans = block_chans = [32, 32, 16]
        self.block_chans = block_chans = [n_chan]
        self.num_blocks = num_blocks = len(block_chans)
        self.conv_init_ = init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0.1),
            nn.init.calculate_gain('relu'))

        for i in range(num_blocks):
            setattr(self, 'block_{}'.format(i),
                    FractalBlock(n_chan_in=block_chans[i-1], n_chan=block_chans[i],
                                 num_inputs=num_inputs, intra_shr=intra_shr,
                                 inter_shr=inter_shr, recurrent=recurrent,
                                 n_recs=n_recs,
                                 num_actions=num_actions, rule=rule, base=self))
        # An assumption. Run drop path globally on all blocks of stack if applicable
        self.n_cols = self.block_0.n_cols

        n_out_chan = block_chans[-1]
        self.n_out_chan = n_out_chan
        self.critic_dwn = init_(nn.Conv2d(n_out_chan, n_out_chan, val_kern, 2, 1))
        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0))
        self.linit_ = lambda m: init(m,
           nn.init.orthogonal_,
           lambda x: nn.init.constant_(x, 0))
        self.critic_out = init_(nn.Conv2d(n_out_chan, 1, 3, 1, 1))
        if out_w == 1:
            self.actor_out = init_(nn.Conv2d(n_out_chan, hidden_size, 1, 1, 0))
        else:
            self.actor_out = init_(nn.Conv2d(n_out_chan, num_actions, 1, 1, 0))
        actor_head = []
        # FIXME: hack for rct
        self.n_act_shrinks = int(math.log(self.map_width // out_w, 2))#+ 1
        if self.n_act_shrinks > 0:
            self.act_0 = init_(nn.Conv2d(self.n_out_chan, self.n_out_chan, 3, 2, 1))
        self.n_val_shrinks = int(math.log(self.map_width, 2))#+ 1
        print('Fractal Net dimensions debug:')
        print('out_w', out_w)
        print('num_actions', num_actions)
        print('n act shrinks: {}'.format(self.n_act_shrinks))
        print('n val shrinks: {}'.format(self.n_val_shrinks))
#       for i in range(n_shrinks):
#           actor_head.append(self.act_0)
#           actor_head.append(nn.ReLU())
#       self.actor_head = nn.Sequential(*actor_head)
        self.active_column = None
    # TODO: should be part of a subclass

    def auto_expand(self):
        self.block_0.auto_expand() # assumption
        self.n_cols += 1

    def forward(self, x, rnn_hxs=None, masks=None):
       #x = self.bn(x)
        for i in range(self.num_blocks):
            block = getattr(self, 'block_{}'.format(i))
            x = F.relu(block(x, rnn_hxs, masks))

        actions = x
        for i in range(self.n_act_shrinks):
            actions = self.act_0(actions)
            actions = F.relu(actions)
        actions = F.relu(self.actor_out(actions))
        actions = th.flatten(actions, start_dim=1)
        values = x

        for i in range(self.n_val_shrinks):
#       for i in range(int(math.log(self.map_width, 2))):
            values = F.relu(self.critic_dwn(values))
        values = self.critic_out(values)
        values = values.view(values.size(0), -1)

        return values, actions, rnn_hxs # no recurrent states
        

    def set_drop_path(self):
        for i in range(self.num_blocks):
            getattr(self, 'block_{}'.format(i)).set_drop_path()

    def set_active_column(self, a):
        self.active_column = a

        for i in range(self.num_blocks):
            getattr(self, 'block_{}'.format(i)).set_active_column(a)

class FractalBlock(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512,
                 map_width=16, n_recs=3, intra_shr=False,
                 inter_shr=False, num_actions=19, rule='extend', n_chan=32,
                 n_chan_in=32, base=None):

        super(FractalBlock, self).__init__(hidden_size)
        
        self.map_width = map_width
        self.n_chan = n_chan
        self.intracol_share = intra_shr # share weights between layers in a col.
        self.intercol_share = inter_shr # share weights between columns
        self.rule = rule # which fractal expansion rule to use
        # each rec is a call to a subfractal constructor, 1 rec = single-layered body
        self.n_recs = n_recs
        print("Fractal Block: expansion type: {}, {} recursions".format(
            self.rule, self.n_recs))

        self.SKIPSQUEEZE = rule == 'wide1' # actually we mean a fractal rule that grows linearly in max depth but exponentially in number of columns, rather than vice versa, with number of recursions #TODO: combine the two rules

        if self.rule == 'wide1':
            self.n_cols = 2 ** (self.n_recs - 1)
            print('{} cols'.format(self.n_cols))
        else:
            self.n_cols = self.n_recs
        self.COLUMNS = False # if true, we do not construct the network recursively, but as a row of concurrent columns
        # if true, weights are shared between recursions
        self.local_drop = False
        # at each join, which columns are taken as input (local drop as described in Fractal Net paper)
        self.global_drop = False
        self.active_column = None
        self.batch_norm = False
        self.c_init_ = init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0.1),
            nn.init.calculate_gain('relu'))
        self.embed_chan = nn.Conv2d(num_inputs, n_chan, 1, 1, 0)
        # TODO: right now, we initialize these only as placeholders to successfully load older models, get rid of these ASAP

        if False and self.intracol_share:
            # how many columns with distinct sets of layers?

            if self.intercol_share:
                n_unique_cols = 1
            else:
                n_unique_cols = self.n_recs

            for i in range(n_unique_cols):
                if self.intracol_share:
                    n_unique_layers = 1
                else:
                    n_unique_layers = 3
                setattr(self, 'fixed_{}'.format(i), init_(nn.Conv2d(
                    self.n_chan, self.n_chan, 3, 1, 1)))

                if n_unique_cols == 1 or i > 0:

                    setattr(self, 'join_{}'.format(i), init_(nn.Conv2d(
                        self.n_chan * 2, self.n_chan, 3, 1, 1)))

                    if self.rule == 'wide1' or self.rule == 'extend_sqz':
                        setattr(self, 'dwn_{}'.format(i), init_(nn.Conv2d(
                            self.n_chan, self.n_chan, 2, 2, 0)))
                        setattr(self, 'up_{}'.format(i), init_(nn.ConvTranspose2d(
                            self.n_chan, self.n_chan, 2, 2, 0)))
        f_c = None

        if self.rule == 'wide1':
            subfractal = SkipFractal
        elif self.rule == 'extend':
            if self.rule == 'extend_sqz':
                subfractal = SubFractal_squeeze
            else:
                subfractal = SubFractal
        n_recs = self.n_recs

        for i in range(n_recs):
            f_c = subfractal(self, f_c, n_rec=i, n_chan=self.n_chan)
        self.f_c = f_c
        self.subfractal = subfractal
        self.join_masks = self.f_c.join_masks


    def auto_expand(self):
        ''' Apply a fractal expansion without introducing new weight layers.

        For neuroevolution or inference.'''
        self.intracol_share = False
        self.f_c = self.subfractal(self, self.f_c, n_rec=self.n_recs, n_chan=self.n_chan)
        setattr(self, 'fixed_{}'.format(self.n_recs), None)
        self.f_c.copy_child_weights()
        self.f_c.fixed = copy.deepcopy(self.f_c.fixed)
        self.n_recs += 1
        self.n_cols += 1
        self.f_c.auto_expand()


    def forward(self, x, rnn_hxs=None, masks=None):
        x = self.embed_chan(x)
        depth = pow(2, self.n_recs - 1)
            # (column, join depth)

        if self.rule == 'wide1':
            net_coords = (0, self.n_recs - 1)
        else:
            net_coords = (self.n_recs - 1, depth - 1 )
        x = F.relu(self.f_c(x))

        return x

    def clear_join_masks(self):
        ''' Returns a set of join masks that will result in activation flowing
        through the entire fractal network.'''

        if self.rule == 'wide1':
            self.join_masks.fill(1)

            return
        i = 0

        for mask in self.join_masks:
            n_ins = len(mask)
            mask = [1]*n_ins
            self.join_masks[i] = mask
            i += 1

    def set_active_column(self, a):
        ''' Returns a set of join masks that will result in activation flowing
        through a (set of) sequential 'column(s)' of the network.
        - a: an integer, or list of integers, in which case multiple sequential
            columns are activated.'''
        self.global_drop = True
        self.local_drop = False

        if a == -1:
            self.f_c.reset_join_masks(True)
        else:
            self.f_c.reset_join_masks(False)
            self.f_c.set_active_column(a)
       #print('set active col to {}\n{}'.format(a, self.f_c.get_join_masks()))


    def set_local_drop(self):
        self.global_drop = False
        self.active_column = None
        reach = False # whether or not there is a path thru
        reach = self.f_c.set_local_drop(force=True)
       #print('local_drop\n {}'.format(self.get_join_masks()))
        assert reach


    def set_global_drop(self):
        a = np.random.randint(0, self.n_recs)
        self.set_active_column(a)

    def set_drop_path(self):
        if np.random.randint(0, 2) == 1:
            self.local_drop = self.set_local_drop()
        else:
            self.global_drop = self.set_global_drop()

    def get_join_masks(self):
        return self.f_c.get_join_masks()



class SubFractal(nn.Module):
    '''
    The recursive part of the network.
    '''
    def __init__(self, root, f_c, n_rec, n_chan):
        super(SubFractal, self).__init__()
        self.n_recs = root.n_recs
        self.n_rec = n_rec
        self.n_chan = n_chan
        self.join_layer = False
        init_ = root.c_init_

        if f_c is not None:
            self.f_c_A = f_c

            if root.intercol_share:
                self.copy_child_weights()
            self.f_c_B = f_c.mutate_copy(root)
            self.join_masks = {'body': True, 'skip': True}
        else:
            self.join_masks = {'body': False, 'skip': True}
        self.active_column = root.active_column

        if (not root.intercol_share) or self.n_rec == 0:
            self.fixed = init_(nn.Conv2d(self.n_chan, self.n_chan, 3, 1, 1))

            if self.join_layer and n_rec > 0:
                self.join = init_(nn.Conv2d(self.n_chan * 2, self.n_chan, 3, 1, 1))

                #if self.join_layer and n_rec > 0:
               #    self.join = getattr(root, 'join_{}'.format(j))

    def auto_expand(self):
        '''just increment n_recs'''
        self.n_recs += 1

    def mutate_copy(self, root):
        ''' Return a copy of myself to be used as my twin.'''

        if self.n_rec > 0:
            f_c = self.f_c_A.mutate_copy(root)
            twin = SubFractal(root, f_c, self.n_rec, n_chan=self.n_chan)
        else:
            twin = SubFractal(root, None, 0, n_chan=self.n_chan)

        if root.intracol_share:
            twin.fixed = self.fixed

        return twin


    def copy_child_weights(self):
        ''' Steal our child's weights to use as our own. Not deep (just refers to existing weights).'''

        if self.n_rec > 0:
            self.fixed = self.f_c_A.fixed

            if self.join_layer:
                self.join = self.f_c_A.join



    def reset_join_masks(self, val=True):
        self.join_masks['skip'] = val

        if self.n_rec > 0:
            self.join_masks['body'] = val
            self.f_c_A.reset_join_masks(val)
            self.f_c_B.reset_join_masks(val)
        else:
            self.join_masks['body'] = False # not needed


    def set_local_drop(self, force):
        ''' Returns True if path from source to target is yielded to self.join_masks.
                - force: a boolean, whether or not to force one path through.'''
        reach = False

        if self.n_rec == 0:
            self.set_child_drops(False, [0, 1])
            reach = True
        else:
            # try for natural path to target
            prob_body = 1 - (1/2) ** self.n_rec
            prob_skip = 1/2
            mask = (np.random.random_sample(2) > [prob_body, prob_skip]).astype(int)
            reach = self.set_child_drops(False, mask)

            if not reach and force: # then force one path down
                mask[1] = np.random.randint(0, 1) <= 1 / (self.n_recs - self.n_rec)
                mask[0] = (mask[1] + 1) % 2
                assert self.set_child_drops(True, mask) == True
                reach = True

        return reach


    def set_child_drops(self, force, mask):
        reach = False

        if force:
            assert 1 in mask

        if mask[1] == 1:
            self.join_masks['skip'] = True
            reach = True
        else:
            self.join_masks['skip'] = False
        self.join_masks['body'] = False

        if mask[0] == 1:
            reach_a = self.f_c_A.set_local_drop(force)

            if reach_a:
                reach_b = self.f_c_B.set_local_drop(force)

                if reach_b:
                    self.join_masks['body'] = True
                    reach = True
            else:
                assert not force

        if force:
            assert reach

        return reach


    def set_active_column(self, col_n):
        if col_n == self.n_rec:
            self.join_masks['skip'] = True
            self.join_masks['body'] = False
        else:
            self.join_masks['skip'] = False
            self.join_masks['body'] = True


    def get_join_masks(self):
        ''' for printing! '''
        stri = ''
        indent = ''

        for i in range(self.n_recs - self.n_rec):
            indent += '    '
        stri = stri + indent + str(self.join_masks)

        if self.n_rec != 0:
            stri = stri + '\n' + str(self.f_c_A.get_join_masks()) + '\n' + str(self.f_c_B.get_join_masks())

        return stri



    def forward(self, x):
        if x is None: return None
        x_c, x_c1 = x, x

        if self.join_masks['skip']:
            for i in range(1):
                x_c1 = F.relu(
                        #self.dropout_fixed
                        (self.fixed(x_c1)))

        if self.n_rec == 0:
            return x_c1

        if self.join_masks['body']:
            x_c = self.f_c_A(x_c)
            x_c =self.f_c_B(x_c)

        if x_c1 is None:
            return x_c

        if x_c is None:
            return x_c1

        if self.join_layer:
            x = F.relu(
                    #self.dropout_join
                    (self.join(th.cat((x_c, x_c1), dim=1))))
        else:
            x = (x_c1 + x_c * (self.n_rec)) / (self.n_rec + 1)

        return x


