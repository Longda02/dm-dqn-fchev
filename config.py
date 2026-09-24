# config 中定义的是参数的值
import argparse

def get_config():
    parser = argparse.ArgumentParser(description='M-RL(MDQN)')
    parser.add_argument('--denominator', default=50, type=int)
    # precondition
    parser.add_argument("--state_size", type=int, default=5)
    parser.add_argument("--state_num", type=int, default=8225)
    parser.add_argument("--action_size", type=int, default=21)
    parser.add_argument("--frame1", type=int, default=0)
    parser.add_argument("--episode",  type=int, default=400000000)
    parser.add_argument("--probability", type=float, default=0.35)   # 选用动作的概率
    parser.add_argument("--replay_state", type=float, default=5, help="Every five repetitions,the state is randomly selected")

    # prepare parameters
    parser.add_argument("--algorithm", type=str, default='masac')
    parser.add_argument("--run_num", type=int, default=1)
    parser.add_argument("--n_episodes", type=int, default=250)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--gamma",  type=int, default=0.99)
    parser.add_argument("--frames", type=int, default=int(45000), help="every now many frame action")
    parser.add_argument("--eps_frames", type=int, default=int(5000), help="every now many frame action")
    parser.add_argument("--min_eps", type=float, default=0.025, help="every now many frame action")
    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")

    # UDDS parameters
    parser.add_argument("--udds_data_path", type=str, default="D:/Drive_Cycle/Standard_UDDS.mat",
                        help="Path to UDDS speed data (.mat file)")
    parser.add_argument("--udds_episode_length", type=int, default=1370,
                        help="Length of one UDDS cycle in seconds")
    parser.add_argument("--initial_soc_bat", type=float, default=0.7,
                        help="Initial SOC of battery")
    parser.add_argument("--initial_soc_uc", type=float, default=0.8,
                        help="Initial SOC of ultracapacitor")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of UDDS episodes for training")
    parser.add_argument("--results_dir", type=str, default="./UDDS_results/",
                        help="Directory to save training results (.mat files)")
    # parser.add_argument("--seed", type=int, default=1, help="numpy/torch的随机种子，复现实验")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--use_cpu", action='store_true', default=True, help="Force use CPU even if GPU is available")
    parser.add_argument("--cuda_deterministic", action='store_false', default=True, help="by default, make sure random seed effective. if.cpu().numpy() set, bypass such function.")

    # env parameters

    # replay buffer parameters
    parser.add_argument("--buffer_size", type=int, default=int(800), help="Max length for buffer")
    parser.add_argument("--batch_size", type=int, default=int(32), help="batch_size大小")

    # network parameters
    parser.add_argument("--soft_update_tau", type=float, default=0.005, help="Max length for any episode")
    parser.add_argument("--hidden_size1", type=int, default=128,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--hidden_size2", type=int, default=256,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--hidden_size3", type=int, default=128,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--layer_N", type=int, default=1, help="Number of layers for actor/critic networks")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=1e-5, help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4, help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--update_every", type=int, default=1)

    # QR-DQN parameters
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help='max norm of gradients (default: 0.5)')
    parser.add_argument("--entropy_tau", type=float, default=0.03, help='entroy的系数')
    # parser.add_argument("--entropy_tau", type=float, default=1, help='entroy的系数')
    parser.add_argument("--alpha", type=float, default=0.9, help='munchansen系数')

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true', default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5, help="time duration between contiunous twice log printing.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")

    return parser




