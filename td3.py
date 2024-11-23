import os
import math
import torch
from torch.optim import Adam
from torch.nn import Upsample
import torch.nn.functional as F
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork
from agent import Agent


class TD3(Agent):
    def __init__(self, num_inputs, action_space, args):
        super().__init__(num_inputs, action_space, args)
        
        self.action_res = args.action_res
        self.upsampled_action_res = args.action_res * args.action_res_resize
        self.target_update_interval = args.target_update_interval

        # Q-Network
        # ϕ_1_target ← ϕ_1, ϕ_2_target ← ϕ_2
        self.q_network = QNetwork(num_inputs, 
                                  self.action_res, 
                                  self.upsampled_action_res, 
                                  args.hidden_size).to(device=self.device)
        self.q_network_target = QNetwork(num_inputs, 
                                  self.action_res, 
                                  self.upsampled_action_res, 
                                  args.hidden_size).to(device=self.device)
        self.q_network_optimizer = Adam(self.q_network.parameters(), lr=args.lr)
        hard_update(self.q_network_target,self.q_network)

        # Policy Network 
        # θ_target ← θ
        self.policy = GaussianPolicy(num_inputs,
                                     self.action_res,
                                     self.upsampled_action_res,
                                     args.residual, 
                                     args.coarse2fine_bias).to(device=self.device)
        self.policy_target = GaussianPolicy(num_inputs,
                                     self.action_res,
                                     self.upsampled_action_res,
                                     args.residual, 
                                     args.coarse2fine_bias).to(device=self.device)

        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        hard_update(self.policy_target,self.policy)


    def select_action(self, state,  task=None):
        state = (torch.FloatTensor(state) / 255.0 * 2.0 - 1.0).to(self.device).unsqueeze(0)

        if task is None or "shapematch" not in task:
            action, _, _, _, mask = self.policy.sample(state)
        else:
            _, _, action, _, mask = self.policy.sample(state)
            action = torch.tanh(action)
        action = action.detach().cpu().numpy()[0]

        return action, None

    def update_parameters(self, memory, updates):
        
        # Sample a batch of transitions (state, action, reward, next_state, done) from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(self.args.batch_size)

        # Convert batches to PyTorch tensors
        state_batch = (torch.FloatTensor(state_batch) / 255.0 * 2.0 - 1.0).to(self.device)
        next_state_batch = (torch.FloatTensor(next_state_batch) / 255.0 * 2.0 - 1.0).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        # normalizing reward
        reward_batch = self.reward_normalization(reward_batch)

        # # DDPG: Deep Deterministic Policy Gradient

        # Q Function update
        with torch.no_grad():
            next_state_pi, _ , _, _, _ = self.policy_target.sample(next_state_batch)    

            # next_action_batch = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.q_network_target(next_state_batch, next_state_pi)
            next_q_value = reward_batch + (mask_batch * self.gamma * torch.min(qf1_next_target, qf2_next_target))

        # Compute current Q-value: Q(s, a) using q network
        qf1, qf2 = self.q_network(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # Optimize the Q-Funtion
        self.q_network_optimizer.zero_grad()
        qf_loss.backward()
        for params in self.q_network.parameters():
            torch.nn.utils.clip_grad_norm_(params, max_norm=10)
        self.q_network_optimizer.step()


        policy_loss=0.0
        if updates % 2 == 0:
            # Policy update
            pi, _, _, _, _ = self.policy.sample(state_batch)
            q1, _ = self.q_network(state_batch, pi)
            policy_loss = -q1.mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            for params in self.policy.parameters():
                torch.nn.utils.clip_grad_norm_(params, max_norm=10)
            self.policy_optim.step()


            # Update target networks using soft update
            soft_update(self.q_network_target, self.q_network, self.tau)
            soft_update(self.policy_target, self.policy, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss if not policy_loss else policy_loss.item(),_, self.alpha,\
            _, _
