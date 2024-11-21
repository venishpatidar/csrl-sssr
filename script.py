import os
import cv2
import gym
import math
import json
import utils
import wandb
import argparse
import datetime
import itertools
import numpy as np
import taichi as ti
import torch
from torch.nn import Upsample

torch.set_num_threads(16)
torch.cuda.empty_cache()

from replay_memory import ReplayMemory
import dittogym
from model import GaussianPolicy, QNetwork

# Agents
from sac import SAC
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Parsing Arguments
parser = argparse.ArgumentParser(description='DittoGym Project')
parser.add_argument('--env_name', default="shapematch-coarse-v0",
                    help='name of the environment to run')
parser.add_argument('--config_file_path', type=str, default=None, metavar='G',
                    help='path of the config file')
parser.add_argument('--wandb', type=bool, default=False, 
                    help='if use wandb (default: False)')
parser.add_argument('--gui', type=bool, default=False, metavar='G',
                    help='if use gui (default: False)')
parser.add_argument('--visualize_interval', type=int, default=10,
                    help='visualization interval (default: 10)')

args = parser.parse_args()

# Name Id
args.name = args.env_name + "_" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

# Wandb
if args.wandb:
    wandb.init(project=args.env_name, name=args.name)
    wandb.config.update(args)
    wandb.define_metric("total_num_steps")
    wandb.define_metric("episode_num")
    wandb.define_metric("train_entropy", step_metric="total_num_steps")
    wandb.define_metric("train_step_reward", step_metric="total_num_steps")
    wandb.define_metric("train_locomotion", step_metric="total_num_steps")
    wandb.define_metric("train_split", step_metric="total_num_steps")
    wandb.define_metric("train_robot_target_distance", step_metric="total_num_steps")
    wandb.define_metric("train_robot_ball_distance", step_metric="total_num_steps")
    wandb.define_metric("train_ball_target_distance", step_metric="total_num_steps")
    wandb.define_metric("train_aver_q_loss", step_metric="total_num_steps")
    wandb.define_metric("train_policy_loss", step_metric="total_num_steps")
    wandb.define_metric("alpha", step_metric="total_num_steps")
    wandb.define_metric("std_norm", step_metric="total_num_steps")
    wandb.define_metric("mask_regularize_loss", step_metric="total_num_steps")
    wandb.define_metric("train_episode_reward", step_metric="episode_num")
    wandb.define_metric("train_episode_normalize_reward", step_metric="episode_num")
    wandb.define_metric("train_episode_length", step_metric="episode_num")


if args.config_file_path is not None:
    args = utils.load_from_json(args, args.config_file_path)
else:
    raise FileNotFoundError("config file not found")

# Loading directory and file_path
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

if not os.path.exists("./results"):
    os.makedirs("./results")
file_path = os.path.join(current_directory, "./results/" + args.name)
args.save_file_name = file_path
upsampled_action_res = args.action_res * args.action_res_resize
if not os.path.exists(file_path):
    os.makedirs(file_path)
if args.save_model and not os.path.exists(file_path + "/models"):
    os.makedirs(file_path + "/models")
json.dump(args.__dict__, open(file_path + "/config.json", 'w'), indent=4)


# Taichi
ti.init(arch=ti.gpu, random_seed=args.seed)

# GUI
if args.visualize:
    gui = ti.GUI("Dittogym", res=512, show_gui=args.gui)

# Device
device = torch.device("cuda" if args.cuda else "cpu")

# Environment
env = gym.make(args.env_name, cfg_path=file_path + "/config.json", wandb_logger=wandb)

# Random
utils.set_random_seed(args.seed, args.cuda_deterministic)
env.action_space.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

# Memory
memory = ReplayMemory(args.replay_size, args.seed, args.batch_size)

total_numsteps = 0
updates = 0
visualize_gap = 0
# Training Loop
for i_episode in itertools.count(1):
    episode_steps = 0
    episode_reward = 0
    episode_normalize_reward = 0
    done = False

    state = env.reset()
    env.render(gui, record=False)

    if args.visualize and total_numsteps >= args.start_steps and visualize_gap == args.visualize_interval:
        env.render(gui, record=True, record_id=total_numsteps)
        generate_video = total_numsteps
        visualize_gap = 0
        render = True
    else:
        generate_video = None
        render = False
        if not total_numsteps >= args.start_steps:
            visualize_gap = 0
        else:
            visualize_gap += 1

    while not done:
        if args.start_steps > total_numsteps:
            final_action = env.action_space.sample()
        else:

            final_action, _ = agent.select_action(state)
            
            # log action image (only log x direction)
            if args.wandb and total_numsteps % 200 == 0:
                final_action_ = ((final_action.reshape(2, args.action_res, args.action_res)[0] + 1) / 2 * 255)
                final_action_ = np.clip(cv2.resize(final_action_, (upsampled_action_res, upsampled_action_res),\
                    interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)
                wandb.log({"final_action_x": wandb.Image(final_action_)})
        
        # RL training
        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, entropy, alpha, std_norm, mask_regularize_loss \
                    = agent.update_parameters(memory, updates)
                if args.wandb:
                    wandb.log({'train_aver_q_loss': (critic_1_loss + critic_2_loss) / 2})
                    wandb.log({'train_policy_loss': policy_loss})
                    wandb.log({'train_entropy': entropy})
                    wandb.log({'alpha': alpha})
                    wandb.log({'std_norm': std_norm})
                    wandb.log({'mask_regularize_loss': mask_regularize_loss})
                updates += 1


        # Take final action and tranition to next state
        next_state, reward, terminated, truncated, _ = env.step(final_action)
        
        # render
        if args.visualize and render:
            env.render(gui, record=True)

        done = truncated or terminated
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        episode_normalize_reward += (reward - agent.mean) / (agent.var**0.5 + 1e-8)


        # Ignore the "done" signal if it comes from hitting the time horizon.
        mask = 1 if truncated else float(not terminated)
        memory.push(state, final_action, reward, next_state, mask) # Append transition to memory
        state = next_state

        if total_numsteps % 1000 == 0:
            if args.save_model:
                agent.save_model(filename=file_path + "/models/" + str(total_numsteps))

        if args.wandb:
            wandb.log({'total_num_steps': total_numsteps})
            wandb.log({'train_step_reward': reward})

    # End of Episode
    if args.wandb:
        wandb.log({'episode_num': i_episode})
        wandb.log({'train_episode_reward': episode_reward})
        wandb.log({'train_episode_normalize_reward': episode_normalize_reward})
        wandb.log({'train_episode_length': episode_steps})

    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}"\
    .format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    utils.generate_video(file_path, generate_video)

    if total_numsteps > args.max_num_steps:
        break

env.close()
