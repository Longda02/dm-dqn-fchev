import torch
import numpy as np
import random
import torch.optim as optim
import torch.nn.functional as F
# from networks import DQN
from networks import DuelingDQN
# from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from Replay import ReplayBuffer
import pandas as pd


#   读取状态表和TPM矩阵
state_list = pd.read_excel("Excel/state-2.xlsx", nrows=8225, sheet_name='Sheet1', index_col=False, header=None)
TPM_10 = pd.read_excel('Excel/TPMfinal10.xlsx', sheet_name='Sheet1', index_col=False, header=None)
TPM_20 = pd.read_excel('Excel/TPMfinal20.xlsx', sheet_name='Sheet1', index_col=False, header=None)
TPM_30 = pd.read_excel('Excel/TPMfinal30.xlsx', sheet_name='Sheet1', index_col=False, header=None)
TPM_40 = pd.read_excel('Excel/TPMfinal40.xlsx', sheet_name='Sheet1', index_col=False, header=None)
TPM_50 = pd.read_excel('Excel/TPMfinal50.xlsx', sheet_name='Sheet1', index_col=False, header=None)
Loss_Max = 0.01

class agent:
    """与环境交互并且学习好的策略"""
    # def __init__(self, state_size, action_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5,config):
    def __init__(self, state_size, action_size, hidden_size1, hidden_size2, hidden_size3,config,frame1):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.frame1=frame1
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and not config.use_cpu) else "cpu")
        #         # self.device = torch.device("cpu")    # 如果存在gpu就将数据转给gpu，否则转给cpu
        # 这个device的用处是作为Tensor或者Model被分配到的位置。因此，在构建device对象后，紧跟的代码往往是：model = Model(...).to(device)
        # Q-Network
        # self.q_net = DQN(state_size, action_size, hidden_size1, hidden_size2, hidden_size3,hidden_size4, hidden_size5).to(self.device)
        self.q_net = DuelingDQN(state_size, action_size, hidden_size1, hidden_size2, hidden_size3).to(self.device)
        self.q_target = DuelingDQN(state_size, action_size, hidden_size1, hidden_size2, hidden_size3).to(self.device)
        # self.q_target = DQN(state_size, action_size, hidden_size1, hidden_size2, hidden_size3,hidden_size4, hidden_size5).to(self.device)

        # optimizer
        #
        # 4
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        # self.optimizer = Ranger(self.q_net.parameters(), lr=config.lr)

        # ReplayBuffer
        self.buffer = ReplayBuffer(action_size, buffer_size=config.buffer_size, batch_size=config.batch_size)

        # 训练时间步骤的初始化
        # self.t_step = 0
        # self.q_updates = 0
        # self.action_step = 4
        self.last_action = None
        # self.writer = SummaryWriter('result')
    def epsilon_explore(self,frame1, frames1,loss):
            if frame1 < 1e6:
                epsilon = max(1 - (frame1 * (1 / 1e6)), 0.01)
            else:
                epsilon = max(0.01 - 0.01 * ((frame1 - 1e6) / (frames1 - 1e6)), 0.001)
            # print(f"Frame: {frame}, Epsilon: {epsilon}")
            return epsilon
        # Loss_Max = 0.01
        # if Loss_Max <= loss:
        #     Loss_Max = loss
        # frame1_tensor = torch.tensor(float(frame1))
        # epsilon = loss / Loss_Max + abs(
        #     np.random.normal(loss, 1 / (torch.log(frame1_tensor / 800) + 1)))
        # epsilon = np.clip(epsilon, 0, 1)
        # # print('epsilon:', epsilon)
        #
        # return epsilon

    # total_frames = 400000000  # 你可以设置总的步数
    # final_epsilon = epsilon_explore(0, total_frames)
    # print("最终 Epsilon:", final_epsilon)
    def get_action(self, state):
        # 根据当前策略返回给定状态的操作
        # state: numpy array of shape (5,) = [SOCbat, SOCuc, velocity, Pdemand, Puc]
        total_frames = 40000000
        action_num = 21
        self.frame1 += 1
        Puc = state[4]   # Puc 是状态向量的第5个元素（索引4）
        CUT = int(Puc * 20)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # current_state = self.state_list.values[current_state_index, :]
        # Puc = self.state_list.values[current_state_index, 4]
        # CUT = int(Puc * 20)
        # current_state = current_state[np.newaxis, :]
        # torch.from_numpy(state)用来将数组array state 转换为张量Tensor
        #  例张量[1.2, -5.6, 9, 0.004]，调用.unsqueeze(0)之后 [[1.2, -5.6, 9, 0.004]] 增加 维度
        # self.q_net.eval()
        action_values = self.q_net(state_tensor).detach()
        action_values1 = action_values[:, 0:action_num - CUT]
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
        """
        使用一个批次的经验轨迹数据来更新值网络和策略网络
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state)) ：这个是基于真实值的标签
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """
        gamma = self.config.gamma
        alpha = self.config.alpha
        entropy_tau = self.config.entropy_tau
        states, actions, rewards, next_states = experiences
        # 从target网络模型里得到预测next的Q值(获得一堆Q值)
        # q_targets_next = self.q_target(next_states).detach()
        q_targets_next = self.q_target.forward(next_states).detach()  # self.q_target.forward 其中是调用self.q_target中的forward函数
        # 计算MDQN中的目标Q值（都是在目标网络中计算的！！！！）
        # 用LogSumExp计算entropy，目的是维持数值的稳定性，详情见博客
        # 如何维持数据的稳定性，（计算tau_log_pi_next）
        # 这里用了一个logsumexp 的trick （注意这里输入的是下个状态对应的Q值！！）
        q_k_targets_next = q_targets_next
        v_k_targets_next = q_targets_next.max(1)[0].unsqueeze(-1)   # unsqueeze(-1)对于一维的张量来增维的。选取qk中最大值作为vk。是每组（每行）里面的最大Q值。
        # print('v_k_targets_next', v_k_targets_next)
        # 为了在计算机中更好的计算，防止计算上溢或者下溢（exp的高次方或者很低次方计算机很难求解）让每个状态得到的各个动作输出的Q值减去当前状态下的最大Q值
        logSum = torch.logsumexp((q_k_targets_next - v_k_targets_next) / entropy_tau, 1).unsqueeze(-1)
        tau_log_pi_next = q_k_targets_next - v_k_targets_next - entropy_tau * logSum
        # 可以理解为tau_log_pi_ 中在分母求和的时候加入 v_k_targets_next后进行的一系列数据处理，
        # 最后求得为 q_k_targets_next - v_k_targets_next - entropy_tau * logSum 其中logSum 又如上个代码所示。
        # 目标策略
        pi_target = F.softmax(q_targets_next / entropy_tau, dim=1)  # 选用softmax将策略随机化，出现的概率
        # （计算下个状态对应的Q值）
        # 目标不仅使奖励最大化 也使策略的熵最大化 即为最后加入tau_log_pi_next
        # 初始的DQN中为 rewards+ gamma * (pi_target * (q_targets_next.sum(1))）
        # 这里加入了策略的熵项
        # q_targets = (gamma * (pi_target * (q_targets_next - tau_log_pi_next) * (1-dones)).sum(1)).unsqueeze(-1)
        q_targets = (gamma * (pi_target * (q_targets_next - tau_log_pi_next)).sum(1)).unsqueeze(-1)
        # q_targets = gamma *  v_k_targets_next
        #（计算当前策略对自身的评价-是当前状态在目标网络！！中求得的Q值）
        # 用logSum计算munchausen的addon（在MRL中处理环境对自身的评价即为奖励 还增加了策略自身对自己的评价，所以这里输出的当前状态对应的Q值！！）
        q_k_targets = self.q_target.forward(states).detach()
        v_k_targets = q_k_targets.max(1)[0].unsqueeze(-1)
        logSum = torch.logsumexp((q_k_targets - v_k_targets) / entropy_tau, 1).unsqueeze(-1)
        tau_log_pi = q_k_targets - v_k_targets - entropy_tau * logSum
        munchausen_addon = tau_log_pi.gather(1, actions.long())     # 选出对应动作的tua_log_pi（与计算熵最大化的处理方式一样，只是输入的是当前状态）
        # 计算munchausen reward
        # munchausen_reward = rewards + alpha * torch.clamp(munchausen_addon, min=-1, max=0)
        munchausen_reward = rewards
        # torch.clamp（）函数的功能将输入input张量每个元素的值压缩到区间[min,max]，并返回结果到一个新张量。
        # q_targets的计算（计算最终的目标Q值）
        q_targets = q_targets + munchausen_reward
        # q_targets = q_targets
        # print(' q_targets',  q_targets)
        # 用当前的状态去估计/预测Q值 （评估的Q值 是在当前网络中计算的！！！！）
        q_expected = self.q_net.forward(states).gather(1, actions.long())       # 选出对应动作的Q值
        # print('q_expected', q_expected)
        # 计算loss,target是我们想去接近的（相当于真实值）666666666666
        loss = F.mse_loss(q_expected.float(), q_targets.float())
        loss.backward()                                           # 反向传播来更新网络的参数。
        clip_grad_norm_(self.q_net.parameters(), max_norm=self.config.max_grad_norm)
        # clip_grad_norm_对所有的梯度乘以一个clip_coef=max_norm/total_norm ，而且乘的前提是clip_coef一定是小于1的，
        # 其中 max_norm越大，对于梯度爆炸的解决越柔和，max_norm越小，对梯度爆炸的解决越狠
        # 所以，按照这个情况：clip_grad_norm只解决梯度爆炸问题，不解决梯度消失问题
        self.optimizer.step()

        # 软更新target！！！
        self.soft_update(self.q_net, self.q_target, self.config.soft_update_tau)  # 调用下面设置的函数
        # return loss.detach().cpu().numpy()
        return loss

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)   # tau应该是一个比较小的数


    def save(self, save_dir='./UDDS_results/model'):
        import os
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.q_net.state_dict(), os.path.join(save_dir, 'UDDS_DMDQN_q_net.pth'))
        torch.save(self.q_target.state_dict(), os.path.join(save_dir, 'UDDS_DMDQN_q_target.pth'))
        print("====================================")
        print(f"Model saved to: {save_dir}")
        print("====================================")

    def load(self):
        # torch.load(self.q_net.state_dict(), './SAC_model/q_net5.pth')
        # torch.load(self.q_target.state_dict(), './SAC_model/q_target5.pth')
        # # torch.load(self.Q_net.state_dict(), './SAC_model/Q_net.pth')
        # # print()
        self.q_target.load_state_dict(torch.load('SAC_model/DMDQN_q_target01.05.66(256).pth'))
        # self.Critic_net1.load_state_dict(torch.load('./TD3_model/model1_1/Critic_net1.pth'))
        # self.Critic_target_net1.load_state_dict(torch.load('./TD3_model/model1_13/Critic_target_net1.pth'))
        # self.Critic_net2.load_state_dict(torch.load('./TD3_model/model1_1/Critic_net2.pth'))
        # self.Critic_target_net2.load_state_dict(torch.load('./TD3_model/model1_1/Critic_target_net2.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")