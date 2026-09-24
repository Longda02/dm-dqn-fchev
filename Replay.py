import copy
import random
from collections import namedtuple, deque

import numpy as np
import torch
# class SumTree(object):
#     data_pointer = 0
#     def __init__(self,capacity):
#         self.capacity = capacity
#         self.tree = np.zeros(2 * capacity - 1)
#         self.data = np.zeros(capacity, dtype=object)
#     def add(self, p, data):
#         tree_idx = self.data_pointer + self.capacity - 1
#         self.data[self.data_pointer] = data  # update data_frame，放入数据？？
#         self.update(tree_idx, p)
#         self.data_pointer += 1
#         if self.data_pointer >= self.capacity:  # replace when exceed the capacity
#             self.data_pointer = 0
#     def update(self, tree_idx, p):              # 更新父节点和子节点
#         change = p - self.tree[tree_idx]
#         self.tree[tree_idx] = p
#         # then propagate the change through tree
#         while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
#             tree_idx = (tree_idx - 1) // 2
#             self.tree[tree_idx] += change
#     def get_leaf(self, v):
#         """
#         Tree structure and array storage:
#         Tree index:
#              0         -> storing priority sum
#             / \
#           1     2
#          / \   / \
#         3   4 5   6    -> storing priority for transitions
#         Array type for storing:
#         [0,1,2,3,4,5,6]
#         """
#         parent_idx = 0
#         while True:     # the while loop is faster than the method in the reference code
#             cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
#             cr_idx = cl_idx + 1                 # 定义叶子的 左右两个子节点 的索引
#             if cl_idx >= len(self.tree):        # reach bottom, end search
#                 leaf_idx = parent_idx
#                 break
#             else:       # downward search, always search for a higher priority node
#                 if v <= self.tree[cl_idx]:      # 谁大 走哪边
#                     parent_idx = cl_idx
#                 else:
#                     v -= self.tree[cl_idx]      # 将值进行修改，走右边，则要减去左边的数
#                     parent_idx = cr_idx
#
#         data_idx = leaf_idx - self.capacity + 1
#         # 最下面一层的leaf_idx （整个树的索引数0-6）与  data_idx（最下面一层叶子索引0-3）的关系；
#         return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
#
#     @property
#     # property 是属性的意思，调用的时候更简单。
#     # 被property修饰的方法只有一个参数，self；它必须要有返回值
#     def total_p(self):
#         return self.tree[0]     # the root 根是所有子节点之和
class ReplayBuffer:
    """存储轨迹转移数组"""
    # def __init__(self, capacity):
    #     self.tree = SumTree(capacity)
    #
    # def store(self, transition):
    #     max_p = np.max(self.tree.tree[-self.tree.capacity:])
    #     if max_p == 0:
    #         max_p = self.abs_err_upper
    #     self.tree.add(max_p, transition)   # set the max p for new p
    def __init__(self, action_size, buffer_size, batch_size, device=None):
        self.action_size = action_size
        self.buffer = deque(maxlen=buffer_size)  # 一个buffer里能存多少条经验轨迹.根据buffer_size的尺寸来定义buffer，其中元素个数为buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])     # 经验元组
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def store(self, state, action, reward, next_state):
        """往buffer里添加新的经验"""
        e = self.experience(state, action, reward, next_state)
        self.buffer.append(e)

    def sample(self):
        """从buffer里随机采样一个批次的轨迹样本"""
        experiences = random.sample(self.buffer, k=self.batch_size)  # 随机抽取batch_size个样本

        # 将变量类型从np转为tensor，并从CPU挪到GPU中进行加速计算
        states = torch.as_tensor(np.vstack([e.state for e in experiences if e is not None]),
                                 dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.vstack([e.action for e in experiences if e is not None]),
                                  dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(np.vstack([e.reward for e in experiences if e is not None]),
                                  dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.vstack([e.next_state for e in experiences if e is not None]),
                                      dtype=torch.float32, device=self.device)
        # dones = torch.as_tensor(np.vstack([e.done for e in experiences if e is not None]),
        #                         dtype=torch.float32, device=self.device)
        # 拼接数组的方法：np.vstack():在竖直方向上堆叠   np.hstack():在水平方向上平铺

        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)




