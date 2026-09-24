# initialization 中定义的是 env agent experence 参数的初始化
from UDDS_Environment import UDDS_Environment
from Agent import agent
from Replay import ReplayBuffer


class Init:
    def __init__(self, config):
        self.config = config                       # 将config中所有打包好的参数转换成self，换为定义的参数初始化，然后后面定义的初始化都在此的基础上
        self.state_size = self.config.state_size   # 可以看作下面每个定义的初始值（全局变量）
        self.action_size = self.config.action_size
        self.buffer_size = self.config.buffer_size
        self.batch_size = self.config.batch_size

    def init_env(self):
        """初始化 UDDS 环境 (使用 FCHEV_SOH 车辆模型)"""
        Env = UDDS_Environment(self.config)
        return Env

    def init_agent(self):                         # 一般设定的都是py文件中函数中定义的输入
        Robot = agent(self.state_size, self.action_size,
                      hidden_size1=self.config.hidden_size1,
                      hidden_size2=self.config.hidden_size2,
                      hidden_size3=self.config.hidden_size3,
                      # hidden_size4=self.config.hidden_size4,
                      # hidden_size5=self.config.hidden_size5,
                      config=self.config,
                      frame1=self.config.frame1
                      )
        return Robot

    def init_experience(self):
        import torch
        device = torch.device("cuda:0" if (torch.cuda.is_available() and not self.config.use_cpu) else "cpu")
        experience = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, device)
        return experience




