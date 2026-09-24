# DM-DQN 训练主程序 —— UDDS 工况版本
# 每回合生成 epXX/ 文件夹，输出结果保存为 .mat 文件
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import scipy.io as sio
from config import get_config
from Initialization import Init


def run_udds(config):
    """UDDS 工况下的 DM-DQN 训练"""
    t_start = time.time()

    # ---- 创建输出根目录 ----
    results_dir = config.results_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'model'), exist_ok=True)

    # ---- 初始化 ----
    initial = Init(config)
    env = initial.init_env()
    agent = initial.init_agent()
    experience = initial.init_experience()

    # ---- 训练参数 ----
    udds_len = config.udds_episode_length  # 1370
    num_episodes = config.episodes         # 500
    buffer_size = config.buffer_size

    # ---- 全局记录（用于最终汇总） ----
    all_ep_loss = []
    all_ep_reward = []
    all_ep_h2 = []
    all_ep_soc_bat_end = []
    all_ep_soc_uc_end = []
    all_ep_fc_de = []

    # ---- 初始化状态 ----
    state = env.reset()
    global_step = 0
    ep_step = 0

    print("=" * 60)
    print("DM-DQN UDDS Training")
    print(f"  Episodes: {num_episodes}  |  UDDS length: {udds_len}s  |  Results: {results_dir}")
    print("=" * 60)

    for ep in range(num_episodes):
        # ---- 本回合记录 ----
        ep_loss = []
        ep_reward = []
        ep_soc_bat = []
        ep_soc_uc = []
        ep_h2_rate = []
        ep_p_fc = []
        ep_p_bat = []
        ep_p_uc = []
        ep_p_demand = []
        ep_velocity = []
        ep_fc_de = []

        ep_reward_sum = 0.0
        ep_h2_sum = 0.0
        ep_loss_sum = 0.0
        ep_step = 0

        # ---- 一个 UDDS 循环 ----
        for t in range(udds_len):
            global_step += 1
            ep_step += 1

            # 动作选择
            action_index = agent.get_action(state)

            # 环境交互
            reward, next_state, done = env.feedback(state, action_index)

            # 存储经验
            experience.store(state, action_index, reward, next_state)

            # 记录
            info = env.get_last_info()
            ep_reward.append(float(reward))
            ep_soc_bat.append(float(env.SOCbat))
            ep_soc_uc.append(float(env.SOCuc))
            ep_h2_rate.append(float(info.get('C_total', 0)))
            ep_p_fc.append(float(info.get('P_fc_kW', 0)))
            ep_p_bat.append(float(info.get('P_bat_kW', 0)))
            ep_p_uc.append(float(info.get('P_uc_kW', 0)))
            ep_p_demand.append(float(state[3]))
            ep_velocity.append(float(state[2]))
            ep_fc_de.append(float(env.fc_degradation))

            ep_reward_sum += reward
            ep_h2_sum += info.get('C_total', 0)

            # 经验池满后学习
            if global_step >= buffer_size:
                loss = agent.learn(experience.sample())
                loss_val = float(loss.cpu().detach().numpy())
                ep_loss.append(loss_val)
                ep_loss_sum += loss_val

            # 状态转移
            state = next_state

            if done:
                break

        # ---- 保存本回合 .mat 文件到 epXX/ ----
        ep_dir = os.path.join(results_dir, f'ep{ep:04d}')
        os.makedirs(ep_dir, exist_ok=True)

        ep_data = {
            'episode': ep,
            'steps': ep_step,
            'global_step_start': global_step - ep_step + 1,
            'global_step_end': global_step,
            # 每步数据 (1 × ep_step)
            'reward': np.array(ep_reward, dtype=np.float64).reshape(1, -1),
            'soc_bat': np.array(ep_soc_bat, dtype=np.float64).reshape(1, -1),
            'soc_uc': np.array(ep_soc_uc, dtype=np.float64).reshape(1, -1),
            'h2_eq_rate_g_per_s': np.array(ep_h2_rate, dtype=np.float64).reshape(1, -1),
            'h2_cumulative_g': np.cumsum(ep_h2_rate, dtype=np.float64).reshape(1, -1),
            'p_fc_kW': np.array(ep_p_fc, dtype=np.float64).reshape(1, -1),
            'p_bat_kW': np.array(ep_p_bat, dtype=np.float64).reshape(1, -1),
            'p_uc_kW': np.array(ep_p_uc, dtype=np.float64).reshape(1, -1),
            'p_demand_kW': np.array(ep_p_demand, dtype=np.float64).reshape(1, -1),
            'velocity_m_s': np.array(ep_velocity, dtype=np.float64).reshape(1, -1),
            'fc_degradation_cum': np.array(ep_fc_de, dtype=np.float64).reshape(1, -1),
            # 如果本回合有 loss
            'loss': np.array(ep_loss, dtype=np.float64).reshape(1, -1) if ep_loss else np.array([]),
            # 汇总标量
            'reward_sum': float(ep_reward_sum),
            'reward_mean': float(ep_reward_sum / ep_step) if ep_step > 0 else 0.0,
            'h2_eq_total_g': float(ep_h2_sum),
            'h2_eq_mean_g_per_s': float(ep_h2_sum / ep_step) if ep_step > 0 else 0.0,
            'loss_mean': float(ep_loss_sum / len(ep_loss)) if ep_loss else float('nan'),
            'soc_bat_final': float(ep_soc_bat[-1]) if ep_soc_bat else 0.0,
            'soc_uc_final': float(ep_soc_uc[-1]) if ep_soc_uc else 0.0,
            'fc_degradation_total': float(ep_fc_de[-1]) if ep_fc_de else 0.0,
        }
        sio.savemat(os.path.join(ep_dir, f'ep{ep:04d}_results.mat'), ep_data)

        # ---- 汇总记录 ----
        all_ep_loss.append(float(ep_loss_sum / len(ep_loss)) if ep_loss else float('nan'))
        all_ep_reward.append(float(ep_reward_sum))
        all_ep_h2.append(float(ep_h2_sum))
        all_ep_soc_bat_end.append(float(ep_soc_bat[-1]) if ep_soc_bat else 0.0)
        all_ep_soc_uc_end.append(float(ep_soc_uc[-1]) if ep_soc_uc else 0.0)
        all_ep_fc_de.append(float(ep_fc_de[-1]) if ep_fc_de else 0.0)

        # ---- 打印回合摘要 ----
        loss_str = f"Loss_avg={all_ep_loss[-1]:.4f}" if ep_loss else "Loss=---(buf)"
        print(f'[Ep {ep:4d}] Steps={ep_step:4d}  '
              f'Reward={all_ep_reward[-1]:.2f}  '
              f'H2_eq={all_ep_h2[-1]:.2f}g  '
              f'{loss_str}  '
              f'SOCbat={all_ep_soc_bat_end[-1]:.3f}  '
              f'SOCuc={all_ep_soc_uc_end[-1]:.3f}  '
              f'FC_De={all_ep_fc_de[-1]:.6f}')

        # 每 50 回合输出一次训练状态
        if (ep + 1) % 50 == 0 and ep_loss:
            recent_loss = np.nanmean(all_ep_loss[-50:])
            recent_h2 = np.mean(all_ep_h2[-50:])
            print(f'  >>> [Ep {ep-49}-{ep}]  AvgLoss={recent_loss:.4f}  '
                  f'AvgH2={recent_h2:.2f}g  '
                  f'Time={time.time()-t_start:.0f}s')

        # 收敛判断
        if ep_loss and len(ep_loss) >= 100:
            recent_100 = np.mean(ep_loss[-100:])
            if recent_100 <= 0.0001:
                print(f'\nConverged! Loss(100avg)={recent_100:.6f} at Episode {ep}')
                agent.save(os.path.join(results_dir, 'model'))
                break

        # 重置环境
        state = env.reset()

    # ============================================================
    # 训练结束，保存汇总 .mat
    # ============================================================
    elapsed = time.time() - t_start
    summary = {
        'num_episodes': len(all_ep_reward),
        'total_reward': np.array(all_ep_reward, dtype=np.float64).reshape(1, -1),
        'total_h2_eq_g': np.array(all_ep_h2, dtype=np.float64).reshape(1, -1),
        'loss_mean_per_ep': np.array(all_ep_loss, dtype=np.float64).reshape(1, -1),
        'soc_bat_final_per_ep': np.array(all_ep_soc_bat_end, dtype=np.float64).reshape(1, -1),
        'soc_uc_final_per_ep': np.array(all_ep_soc_uc_end, dtype=np.float64).reshape(1, -1),
        'fc_degradation_per_ep': np.array(all_ep_fc_de, dtype=np.float64).reshape(1, -1),
        'training_time_s': float(elapsed),
        'final_avg_loss_last_10': float(np.nanmean(all_ep_loss[-10:])) if len(all_ep_loss) >= 10 else float('nan'),
        'final_avg_h2_last_10': float(np.mean(all_ep_h2[-10:])) if len(all_ep_h2) >= 10 else float('nan'),
    }
    sio.savemat(os.path.join(results_dir, 'training_summary.mat'), summary)

    agent.save(os.path.join(results_dir, 'model'))

    print("\n" + "=" * 60)
    print(f"Training Done.  Time: {elapsed:.0f}s  ({elapsed/60:.1f}min)")
    print(f"Results: {results_dir}")
    print(f"  - training_summary.mat  (全局汇总)")
    print(f"  - ep0000~ep{len(all_ep_reward)-1:04d}/  (每回合详情)")
    print("=" * 60)

    # ---- 画汇总图 ----
    _plot_summary(all_ep_loss, all_ep_reward, all_ep_h2,
                  all_ep_soc_bat_end, all_ep_soc_uc_end)


def _plot_summary(losses, rewards, h2, soc_bat, soc_uc):
    """绘制训练汇总图"""
    episodes = range(len(rewards))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    ax = axes[0, 0]
    ax.plot(episodes, losses, 'b-', linewidth=0.8)
    ax.set_ylabel('Mean Loss')
    ax.set_xlabel('Episode')
    ax.set_title('Loss per Episode')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(episodes, rewards, 'g-', linewidth=0.8)
    ax.set_ylabel('Total Reward')
    ax.set_xlabel('Episode')
    ax.set_title('Reward per Episode')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(episodes, h2, 'r-', linewidth=0.8)
    ax.set_ylabel('H2 eq. (g)')
    ax.set_xlabel('Episode')
    ax.set_title('Equivalent H2 per Episode')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(episodes, soc_bat, 'b-', linewidth=0.8)
    ax.set_ylabel('SOCbat end')
    ax.set_xlabel('Episode')
    ax.set_title('Battery SOC')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(episodes, soc_uc, 'orange', linewidth=0.8)
    ax.set_ylabel('SOCuc end')
    ax.set_xlabel('Episode')
    ax.set_title('UC SOC')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(episodes, np.cumsum(h2) / 1000, 'purple', linewidth=0.8)
    ax.set_ylabel('Cum. H2 (kg)')
    ax.set_xlabel('Episode')
    ax.set_title('Cumulative H2 (kg)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = get_config()
    config = parser.parse_args()
    run_udds(config)
