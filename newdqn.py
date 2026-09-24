import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from torch.nn.utils import clip_grad_norm_
import torch
import numpy as np
import random
import torch.optim as optim
import torch.nn.functional as F
from networks import DQN
# from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from Replay import ReplayBuffer
import pandas as pd
state_list = pd.read_excel("Excel/state-2.xlsx", nrows=8225, sheet_name='Sheet1', index_col=False, header=None)
TPM_10 = pd.read_excel('Excel/TPMfinal10.xlsx', sheet_name='Sheet1', index_col=False, header=None)
TPM_20 = pd.read_excel('Excel/TPMfinal20.xlsx', sheet_name='Sheet1', index_col=False, header=None)
TPM_30 = pd.read_excel('Excel/TPMfinal30.xlsx', sheet_name='Sheet1', index_col=False, header=None)
TPM_40 = pd.read_excel('Excel/TPMfinal40.xlsx', sheet_name='Sheet1', index_col=False, header=None)
TPM_50 = pd.read_excel('Excel/TPMfinal50.xlsx', sheet_name='Sheet1', index_col=False, header=None)
Loss_Max = 0.5

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
class agent:
    """与环境交互并且学习好的策略"""
    # def __init__(self, state_size, action_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5,config):
    def __init__(self, state_size, action_size, hidden_size1, hidden_size2, hidden_size3,config,frame1):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.frame1=frame1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    # 如果存在gpu就将数据转给gpu，否则转给cpu
        #         # self.device = torch.device("cpu")    # 如果存在gpu就将数据转给gpu，否则转给cpu
        # 这个device的用处是作为Tensor或者Model被分配到的位置。因此，在构建device对象后，紧跟的代码往往是：model = Model(...).to(device)
        # Q-Network
        # self.q_net = DQN(state_size, action_size, hidden_size1, hidden_size2, hidden_size3,hidden_size4, hidden_size5).to(self.device)
        self.q_net = DQN(state_size, action_size, hidden_size1, hidden_size2, hidden_size3).to(self.device)
        self.q_target = DQN(state_size, action_size, hidden_size1, hidden_size2, hidden_size3).to(self.device)
        # self.q_target = DQN(state_size, action_size, hidden_size1, hidden_size2, hidden_size3,hidden_size4, hidden_size5).to(self.device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)

        # ReplayBuffer
        self.buffer = ReplayBuffer(action_size, buffer_size=config.buffer_size, batch_size=config.batch_size)
        self.last_action = None
    def epsilon_explore(self,frame1, frames1,loss):
        #     if frame1 < 1e6:
        #         epsilon = max(1 - (frame1 * (1 / 1e6)), 0.01)
        #     else:
        #         epsilon = max(0.01 - 0.01 * ((frame1 - 1e6) / (frames1 - 1e6)), 0.001)
        #     # print(f"Frame: {frame}, Epsilon: {epsilon}")
        #
        #     return epsilon
        Loss_Max = 0.5
        if Loss_Max <= loss:
            Loss_Max = loss
        frame1_tensor = torch.tensor(float(frame1))
        epsilon = loss / Loss_Max + abs(
            np.random.normal(loss, 1 / (torch.log(frame1_tensor / 800) + 1)))
        epsilon = np.clip(epsilon, 0, 1)
        # print('epsilon:', epsilon)

        return epsilon

    # total_frames = 400000000  # 你可以设置总的步数
    # final_epsilon = epsilon_explore(0, total_frames)
    # print("最终 Epsilon:", final_epsilon)
    def get_action(self, state_index):
        # 根据当前策略返回给定状态的操作，确定性策略，画面每更新4帧多一次动作
        total_frames = 40000000
        action_num = 21
        self.frame1 += 1
        # print("最终 Epsilon:", self.frame1)
        state = state_list.values[state_index, :]
        Puc=state_list.values[state_index, 4]
        CUT = int(Puc * 20)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # 增加一个维度给batch_size
        action_values = self.q_net(state).detach()     # .detach()返回一个新的tensor，从当前计算图中分离下来的，
        action_values1 = action_values[:,0:action_num-CUT]  # .detach()返回一个新的tensor，从当前计算图中分离下来的，
        # self.q_net.train()
        # loss = self.learn()
        # loss = loss.cpu().detach().numpy()
        # Epsilon-greedy action selection
        # if random.random() >self.epsilon_explore(self.frame1,total_frames):           # DQN中 动作的选取 用的是∈-greedy （即有一定的概率选择动作）
        if random.random() >self.config.probability:           # DQN中 动作的选取 用的是∈-greedy （即有一定的概率选择动作）
            action = np.argmax(action_values1.cpu().numpy())     # DQN中 动作的评估 选用 贪婪算法 求得Q最大的动作
            # action = np.argmax(action_values.numpy())     # DQN中 动作的评估 选用 贪婪算法 求得Q最大的动作
            # print("最终 Epsilon:", epsilon)
            # .cpu().numpy()是将CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。
            # self.last_action = action
            return action
        else:
            # action = random.choice(np.arange(self.action_size-CUT))
            action = np.random.randint(0, action_num - CUT)
            # self.last_action = action
            return action               # 这里不管是哪种条件输出的都是动作的索引

    def learn(self, experiences):
        gamma = self.config.gamma
        alpha = self.config.alpha
        entropy_tau = self.config.entropy_tau
        states, actions, rewards, next_states, dones = experiences

        # Dueling DQN: 计算下一个状态的状态值和优势值
        next_state_value, next_state_advantage = self.q_target.forward(next_states).detach().split(1, dim=1)
        q_targets_next = next_state_value + (next_state_advantage - next_state_advantage.mean(dim=1, keepdim=True))

        # Munchausen 添加项的计算
        state_value, state_advantage = self.q_target.forward(states).detach().split(1, dim=1)
        q_k_targets = state_value + (state_advantage - state_advantage.mean(dim=1, keepdim=True))

        # 计算 Munchausen 添加项
        v_k_targets_next = q_targets_next.max(1)[0].unsqueeze(-1)
        logSum = torch.logsumexp((q_k_targets - v_k_targets_next) / entropy_tau, 1).unsqueeze(-1)
        tau_log_pi_next = q_k_targets - v_k_targets_next - entropy_tau * logSum

        # 计算策略目标
        pi_target = F.softmax(q_targets_next / entropy_tau, dim=1)

        # 计算目标 Q 值
        q_targets = rewards + (
                    gamma * (1 - dones) * ((pi_target * (q_targets_next - tau_log_pi_next)).sum(1))).unsqueeze(-1)

        # Munchausen reward
        v_k_targets = q_k_targets.max(1)[0].unsqueeze(-1)
        logSum = torch.logsumexp((q_k_targets - v_k_targets) / entropy_tau, 1).unsqueeze(-1)
        tau_log_pi = q_k_targets - v_k_targets - entropy_tau * logSum
        munchausen_addon = tau_log_pi.gather(1, actions.long())
        munchausen_reward = rewards + alpha * torch.clamp(munchausen_addon, min=-1, max=0)
        q_targets = q_targets + munchausen_reward

        # 计算预测的当前状态 Q 值
        current_state_value, current_state_advantage = self.q_net.forward(states).split(1, dim=1)
        q_expected = current_state_value + (
                    current_state_advantage.gather(1, actions.long()) - current_state_advantage.mean(dim=1,
                                                                                                     keepdim=True))

        # 计算损失
        loss = F.mse_loss(q_expected, q_targets)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.q_net.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()

        # 软更新目标网络
        self.soft_update(self.q_net, self.q_target, self.config.soft_update_tau)

        return loss
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)   # tau应该是一个比较小的数


    def save(self):
        # torch.save(self.policy_net.state_dict(), './SAC_model/policy_net.pth')
        torch.save(self.q_net.state_dict(), './SAC_model/MDQN_q_net12.14.1(256).pth')
        torch.save(self.q_target.state_dict(), './SAC_model/MDQN_q_target12.14.1(256).pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        # torch.load(self.q_net.state_dict(), './SAC_model/q_net5.pth')
        # torch.load(self.q_target.state_dict(), './SAC_model/q_target5.pth')
        # # torch.load(self.Q_net.state_dict(), './SAC_model/Q_net.pth')
        # # print()
        self.q_target.load_state_dict(torch.load('SAC_model/MDQN_q_target12.14.1(256).pth'))
        # self.Critic_net1.load_state_dict(torch.load('./TD3_model/model1_1/Critic_net1.pth'))
        # self.Critic_target_net1.load_state_dict(torch.load('./TD3_model/model1_13/Critic_target_net1.pth'))
        # self.Critic_net2.load_state_dict(torch.load('./TD3_model/model1_1/Critic_net2.pth'))
        # self.Critic_target_net2.load_state_dict(torch.load('./TD3_model/model1_1/Critic_target_net2.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")