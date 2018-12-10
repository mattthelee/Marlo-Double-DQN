from chainerrl.agents.dqn import DQN
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl import q_functions
from chainerrl import replay_buffer
from chainer import optimizers
import chainerrl
import logging
import sys
import os
import argparse

import chainer

import gym

# gym.undo_logger_setup()  # NOQA
from gym import spaces
import gym.wrappers

import numpy as np
import marlo
from marlo import experiments

import time

games = ['MarLo-Obstacles-v0',   'MarLo-TrickyArena-v0',      'MarLo-Vertical-v0',
         'MarLo-Attic-v0',       'MarLo-DefaultFlatWorld-v0', 'MarLo-DefaultWorld-v0',
         'MarLo-Eating-v0',      'MarLo-CatchTheMob-v0',       'MarLo-CliffWalking-v0',
         'MarLo-FindTheGoal-v0']



# For more details on DQN Parameters see https://chainerrl.readthedocs.io/en/latest/agents.html#agent-interfaces

# PARAMETER VARIABLES

GAME = games[9]     # Game to run in
VIDEO_RES = 84     # Sets the resolution in pixels of the MARLO screen (at VIDEO_RES x VIDEO_RES)
DEBUG_ON = True    # Limits output to console from this file
GPU = -1             # GPU to use (index: 0 to N). If no GPU available, set to -1.



# PARAMETER FUNCTIONS

# Sets the experimental profile
def set_experiment_profile():

    # Number of total time steps (across all episodes) for training. When this number of played steps is reached, training is over
    steps = 10 ** 6

    # After how many episodes an evaluation is performed. Its results are dumped to "scores.txt"
    eval_interval = 20  # 10 ** 4  # Commented values are default for large tasks.

    # Number of times the game is played for each evaluation
    eval_n_runs = 10  # 100

    # Maximum duration for the episode during evaluation
    max_eval_episode_len = 100

    return steps, eval_n_runs, eval_interval, max_eval_episode_len


# Sets the discount factor for the network.
def set_discount_factor():

    # Discount factor
    gamma = 0.99

    return gamma

# Sets the exploration dynamics.
def set_explorer(env):

    # Possible parameters:

    # Initial (and max) value of epsilon at the start of the experimentation.
    start_epsilon = 1.0

    # Minimum value of epsilon
    end_epsilon = 0.1

    # Constant epsilon
    cons_epsilon = 0.001

    # how many steps it takes for epsilon to decay
    final_exploration_steps = 10 ** 5


    # Options for exploration (more explorers at site-packages/chainerrl/explorers/)

    # Option 1: Constant
    constant_epsilon_explorer = explorers.ConstantEpsilonGreedy(
        epsilon=cons_epsilon,
        random_action_func=env.action_space.sample
    )

    # Option 2: Linear decay
    decay_epsilon_explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon,
        end_epsilon,
        final_exploration_steps,
        random_action_func=str(env.action_space.sample)
    )

    return constant_epsilon_explorer
    # return decay_epsilon_explorer


def set_SDG_optimizer(q_function):

    # Possible parameters:

    alpha_learning_rate = 0.01


    # Options for SDG (more optimizers available at site-packages/chainer/optimizers/ and https://docs.chainer.org/en/stable/reference/optimizers.html)

    # Option 1: Vanilla Stochastic Gradient Descent
    opt_sgd = optimizers.SGD(lr=alpha_learning_rate)

    # Option 2: Adam (A Method for Stochastic Optimization) optimizer
    opt_adam = optimizers.Adam()
    opt_adam.setup(q_function)

    return opt_sgd
    #return opt_adam



# Sets the Replay Buffer to use for DQN.
def set_replay_buffer():

    # Experience replay is disabled if the number of transitions in the replay buffer is lower than this value
    replay_start_size = 1000

    # After how many steps the batch of experiences are sampled again for SGD
    update_interval = 1

    # Capacity of the Reply Buffer for DQN
    rbuf_capacity = 5 * 10 ** 5

    return replay_start_size, update_interval, rbuf_capacity


# Sets the dynamics for updating the target network.
def set_network_target_update():
    # Number of steps to update the target network
    target_update_interval = 10 ** 2

    # Method for updating the target weights: 'hard' or 'soft'
    target_update_method = 'hard'

    # Tau of soft target update
    soft_update_tau = 1e-2

    return target_update_interval, target_update_method, soft_update_tau

# ------------------------------------------------


# SUPPORT FUNCTIONS

# Sets the directory for output files.
def dirs(args):
    out_dir = args.results_dir
    save_dir = args.save_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir_logs = out_dir + '/logging'
    if not os.path.exists(out_dir_logs):
        os.makedirs(out_dir_logs)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return out_dir, save_dir


def phi(observation):
    return observation.astype(np.float32)



# START!


# ARGUMENTS FOR THE DQN

parser = argparse.ArgumentParser(description='chainerrl dqn')
parser.add_argument('--n_hidden_channels', type=int, default=50, help='the number of hidden channels')
parser.add_argument('--n_hidden_layers', type=int, default=1, help='the number of hidden layers')
parser.add_argument('--results_dir', type=str, default="results", help='the results output dir')
parser.add_argument('--save_dir', type=str, default=None, help='the dir to save to or none')
parser.add_argument('--load_dir', type=str, default=None, help='the dir to save to or none.')
args = parser.parse_args()

out_dir, save_dir = dirs(args)

n_hidden_channels = args.n_hidden_channels
n_hidden_layers = args.n_hidden_layers

if DEBUG_ON:
    print("n_hidden_channels " + str(n_hidden_channels) + " n_hidden_layers " + str(n_hidden_layers))


# GAME SELECTION AND CONNECTION TO THE CLIENT

# Ensure that you have a minecraft-client running with : marlo-server --port 10000
client_pool = [('127.0.0.1', 10020)]
if DEBUG_ON:
    print("Game:", GAME)
join_tokens = marlo.make(GAME,
                         params=dict(
                             videoResolution=[VIDEO_RES, VIDEO_RES],
                             kill_clients_after_num_rounds=500
                         ))
env = marlo.init(join_tokens[0])

# ------------------------------------------

obs = env.reset()
env.render(mode="rgb_array")
if DEBUG_ON:
    print('initial observation:', obs)

action = env.action_space.sample()
obs, r, done, info = env.step(action)

if DEBUG_ON:
    print('next observation:', obs)
    print('reward:', r)
    print('done:', done)
    print('info:', info)

    print('actions:', str(env.action_space))
    print('sample action:', str(env.action_space.sample))

timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
obs_space = env.observation_space
obs_size = obs_space.low.size
action_space = env.action_space

n_actions = action_space.n
q_func = q_functions.FCStateQFunctionWithDiscreteAction(
    obs_size, n_actions,
    n_hidden_channels=n_hidden_channels,
    n_hidden_layers=n_hidden_layers
)

# Set up explorer
explorer = set_explorer(env)

# Set up replay buffer
replay_start_size, update_interval, rbuf_capacity = set_replay_buffer()

# Set up params for network target update
target_update_interval, target_update_method, soft_update_tau = set_network_target_update()

# Set up the Stochastic Gradient Descent Optimizer
optimizer = set_SDG_optimizer(q_func)

# Use GPU if any available
if GPU >= 0:
    chainer.cuda.get_device(GPU).use()
    q_func.to_gpu(GPU)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
rbuf = chainerrl.replay_buffer.ReplayBuffer(capacity=rbuf_capacity)

# Initialize the agent
agent = DQN(
    q_func, optimizer, rbuf,
    gpu=GPU,
    gamma=set_discount_factor(),
    explorer=explorer,
    replay_start_size=replay_start_size,
    target_update_interval=target_update_interval,
    update_interval=update_interval,
    phi=phi,
    target_update_method=target_update_method,
    soft_update_tau=soft_update_tau,
    episodic_update_len=16
)

if args.load_dir:
    if DEBUG_ON:
        print("Loading model")
    agent.load(args.load_dir)

# Sets the experiment profile
steps, eval_n_runs, eval_interval, max_eval_episode_len = set_experiment_profile()

# Trains an agent while regularly evaluating it.
experiments.train_agent_with_evaluation(
    agent=agent,
    env=env,
    eval_env=env,
    steps=steps,
    eval_n_runs=eval_n_runs,
    eval_interval=eval_interval,
    outdir=out_dir,
    max_episode_len=max_eval_episode_len,  #timestep_limit
    save_best_so_far_agent=False
)

if save_dir:
    if DEBUG_ON:
        print("Saving model")
    agent.save(save_dir)

# Draw the computational graph and save it in the output directory.
chainerrl.misc.draw_computational_graph(
    [q_func(np.zeros_like(obs_space.low, dtype=np.float32)[None])],
    os.path.join(out_dir, 'model')
)
