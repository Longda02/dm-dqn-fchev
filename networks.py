# import torch
# import torch.nn as nn
#
#
def init_weight(layer):
    if type(layer) == nn.Linear:
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # 正态分布初始化（可以定义为初始化神经网络的权重）
        # torch.nn.init.kaiming_normal_使用正态分布对输入张量进行赋值，这里可以理解为对网络的权重进行初始化。
#
#
# class DQN(nn.Module):
#     def __init__(self, state_size, action_size, hidden_size1, hidden_size2, hidden_size3):
#         super(DQN, self).__init__()  # 继承自父类(nn.Module)的属性进行初始化  #一般用torch 构建网络的时候会存在
#         self.net = nn.Sequential(nn.Linear(state_size, hidden_size1),nn.Tanh(),
#                                  nn.Linear(hidden_size1, hidden_size2), nn.Tanh(),
#                                  nn.Linear(hidden_size2, hidden_size3), nn.Tanh(),
#                                  nn.Linear(hidden_size3, action_size), nn.Tanh(),
#                                  # nn.Linear(hidden_size4, hidden_size5), nn.Tanh(),
#                                  # nn.Linear(hidden_size5, action_size)
#                                  )
#         # torch.nn.Sequential是一个Sequential容器，
#         # 模块将按照构造函数中  传递的顺序  添加到模块中。另外，也可以传入一个有序模块。
#         # 可以加入激励函数,
#         self.net.apply(init_weight)
#
#     def forward(self, state):
#         return self.net(state)          # forward(self, state)设置输入变量state 输出的是Q值
##################################################################################
##################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size1, hidden_size2, hidden_size3):
        super(DuelingDQN, self).__init__()
        # 共享层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size1),
            nn.Tanh(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Tanh()
        )

        # 状态值流（Value Stream）
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size3),
            nn.Tanh(),
            nn.Linear(hidden_size3, 1)  # 输出单个值，代表状态值
        )

        # 优势值流（Advantage Stream）
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size3),
            nn.Tanh(),
            nn.Linear(hidden_size3, action_size)  # 输出每个动作的优势值
        )

        # 初始化权重
        self.shared_layers.apply(init_weight)
        self.value_stream.apply(init_weight)
        self.advantage_stream.apply(init_weight)

    def forward(self, state):
        # 共享层的输出
        shared_output = self.shared_layers(state)

        # 计算状态值和优势值
        value = self.value_stream(shared_output)
        advantage = self.advantage_stream(shared_output)

        # 计算Q值：状态值 + 优势值 - 优势值的平均值
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
