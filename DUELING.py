import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from torch.nn.utils import clip_grad_norm_

class DuelingDQN(torch.nn.Module):
    """
    Dueling DQN网络
    """
    def __init__(self, state_size, action_size, hidden_size1, hidden_size2, hidden_size3):
        super(DuelingDQN, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden_size1)
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size2, hidden_size3)

        # Value stream
        self.value_stream = torch.nn.Linear(hidden_size3, 1)

        # Advantage stream
        self.advantage_stream = torch.nn.Linear(hidden_size3, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class Agent:
    """
    与环境交互并学习好的策略的代理
    """
    def __init__(self, state_size, action_size, hidden_size1, hidden_size2, hidden_size3, config, frame1):
        # ...
        # 使用Dueling DQN网络
        self.q_net = DuelingDQN(state_size, action_size, hidden_size1, hidden_size2, hidden_size3).to(self.device)
        self.q_target = DuelingDQN(state_size, action_size, hidden_size1, hidden_size2, hidden_size3).to(self.device)
        # ...

    # 其他方法保持不变

    def learn(self, experiences):
        # ...
        # Dueling Munchausen DQN的学习过程
        # ...

        # Q值的计算需要按照Dueling架构调整
        # 获取状态值和优势值
        state_value, advantage = self.q_target.forward(next_states).detach().split(1, dim=1)
        q_targets_next = state_value + (advantage - advantage.mean(dim=1, keepdim=True))

        # 同样的修改也应用于当前状态的Q值计算
        state_value, advantage = self.q_target.forward(states).detach().split(1, dim=1)
        q_k_targets = state_value + (advantage - advantage.mean(dim=1, keepdim=True))
        return loss




