# UDDS Environment for DM-DQN
# 替代 Enviroment.py，使用 UDDS 工况 + FCHEV_SOH 车辆模型
import numpy as np
import pandas as pd
import math
import scipy.io as sio
from FCHEV_SOH import FCHEV_SOH

# 读取动作空间
action_space = pd.read_excel('Excel/action space 21.xlsx', index_col=False, header=None)

# UDDS 速度数据
_udds_data = sio.loadmat('D:/Drive_Cycle/Standard_UDDS.mat')
UDDS_SPEED = _udds_data['speed_vector'].flatten()  # m/s, length 1370
UDDS_LENGTH = len(UDDS_SPEED)


class UDDS_Environment:
    """
    UDDS 工况下的混合动力系统环境
    状态向量: [SOCbat, SOCuc, velocity(m/s), Pdemand(kW), Puc]
    动作: 0~20 整数索引, 对应 21 种 FC/Battery 功率分配比例
    """

    def __init__(self, config):
        self.config = config

        # ---- 初始化 FCHEV_SOH 车辆模型 ----
        self.fchev = FCHEV_SOH()

        # ---- UC (超级电容) 参数 ----
        self.C_uc = 1500.0      # 电容 (F)，公交车级别大容量
        self.V_uc_max = 380.0   # 最大电压 (V)
        self.V_uc_min = 190.0   # 最低电压 (V)
        self.E_uc_max = 0.5 * self.C_uc * (self.V_uc_max ** 2 - self.V_uc_min ** 2) / 3600000.0  # kWh
        self.SOCuc_opt = 0.8
        self.SOCuc_max = 1.0
        self.SOCuc_min = 0.4

        # ---- 电池 SOC 参数 ----
        self.SOCbat_opt = 0.7
        self.SOCbat_max = 0.8
        self.SOCbat_min = 0.4
        self.beta = 1000.0      # SOC 偏移惩罚系数

        # ---- 效率常数 ----
        self.Nch_ave_bat = 0.95
        self.Ndisch_ave_bat = 0.95
        self.Nch_ave_uc = 0.95
        self.Ndis_ave_uc = 0.95

        # ---- 燃料电池参数 ----
        self.Pfc_rating = 60.0  # kW (from FCHEV_SOH)
        self.Nfc_opt = 0.7
        self.Nfc_max = 0.8
        self.Nfc_min = 0.6

        # ---- 预计算 UDDS 功率需求 ----
        self.Pdemand_list = self._precompute_pdemand()

        # ---- 运行时状态 ----
        self.time_step = 0
        self.SOCbat = config.initial_soc_bat
        self.SOCuc = config.initial_soc_uc
        self.Puc_values = [0.0, 0.2, 0.4, 0.6, 0.8]

        # ---- 记录 ----
        self.h2_total = 0.0          # 累计实际氢耗 (g)
        self.h2_eq_total = 0.0       # 累计等效氢耗 (g)
        self.fc_degradation = 0.0    # 累计 FC 衰退
        self.step_count = 0

    def _precompute_pdemand(self):
        """从 UDDS 速度曲线预计算每步的功率需求 (kW)"""
        pdemand_list = []
        v_prev = 0.0
        for t in range(UDDS_LENGTH):
            v = UDDS_SPEED[t]
            a = (v - v_prev) / 1.0  # dt = 1s
            v_prev = v

            try:
                T_axle, W_axle, P_axle = self.fchev.T_W_axle(v, a)
                T_mot, W_mot, mot_eff, P_mot = self.fchev.run_motor(T_axle, W_axle, P_axle)
                Pdemand_kW = P_mot / 1000.0  # W -> kW
            except Exception:
                Pdemand_kW = 0.0

            pdemand_list.append(Pdemand_kW)

        return np.array(pdemand_list)

    def reset(self):
        """重置环境到初始状态"""
        self.time_step = 0
        self.SOCbat = self.config.initial_soc_bat
        self.SOCuc = self.config.initial_soc_uc
        self.h2_total = 0.0
        self.h2_eq_total = 0.0
        self.fc_degradation = 0.0
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        """构建当前状态向量"""
        t = self.time_step % UDDS_LENGTH
        velocity = UDDS_SPEED[t]
        Pdemand = self.Pdemand_list[t]
        # Puc 从 5 个离散值随机选取
        Puc = self.Puc_values[np.random.randint(0, 5)]
        return np.array([self.SOCbat, self.SOCuc, velocity, Pdemand, Puc], dtype=np.float32)

    def _compute_Kfc(self, P_fc_kW):
        """燃料电池惩罚因子"""
        Nfc = P_fc_kW / self.Pfc_rating if self.Pfc_rating > 0 else 0
        Nfc = np.clip(Nfc, 0, 1)
        Kfc = (1 - 2 * (Nfc - self.Nfc_opt) / (self.Nfc_max - self.Nfc_min)) ** 4
        return Kfc

    def _compute_Kbat(self, SOCbat):
        """电池 SOC 惩罚因子"""
        if self.SOCbat_min <= SOCbat <= self.SOCbat_max:
            Kbat = (1 - 2 * (SOCbat - self.SOCbat_opt) / (self.SOCbat_max - self.SOCbat_min)) ** 8
        else:
            Kbat = (1 - 2 * (SOCbat - self.SOCbat_opt) / (self.SOCbat_max - self.SOCbat_min)) ** 16
        return Kbat

    def _compute_Kuc(self, SOCuc):
        """超级电容 SOC 惩罚因子"""
        if 0.5 < SOCuc <= 0.8:
            Kuc = (1 - 2 * (SOCuc - self.SOCuc_opt) / (self.SOCuc_max - self.SOCuc_min)) ** 2
        elif 0.4 <= SOCuc <= 0.5:
            Kuc = (1 - 2 * (SOCuc - self.SOCuc_opt) / (self.SOCuc_max - self.SOCuc_min)) ** 8
        else:
            Kuc = (1 - 2 * (SOCuc - self.SOCuc_opt) / (self.SOCuc_max - self.SOCuc_min)) ** 16
        return Kuc

    def feedback(self, state_vector, action_index):
        """
        执行动作，返回 (reward, next_state_vector, done)
        state_vector: [SOCbat, SOCuc, velocity, Pdemand, Puc]
        """
        SOCbat = state_vector[0]
        SOCuc = state_vector[1]
        Pdemand = state_vector[3]   # kW
        Puc_ratio = state_vector[4]

        # ---- 功率分配 ----
        # UC 功率
        P_uc_kW = Pdemand * Puc_ratio
        remaining_kW = Pdemand - P_uc_kW

        # FC 和电池分配比例 (来自动作空间)
        Pfc_ratio = action_space.values[0, action_index]
        Pbat_ratio = action_space.values[1, action_index]

        P_fc_kW = remaining_kW * Pfc_ratio
        P_bat_kW = remaining_kW * Pbat_ratio

        # 限制 FC 功率范围
        P_fc_kW = np.clip(P_fc_kW, 0.0, self.fchev.P_FC_max)

        # ---- 燃料电池模型 ----
        if P_fc_kW > 0.1:
            P_dcdc_kW, h2_fcs, fc_info = self.fchev.run_fuel_cell(P_fc_kW)
        else:
            h2_fcs = 0.0
            P_dcdc_kW = 0.0
            fc_info = {'fce_eff': 0.0}

        # ---- 电池模型 ----
        P_bat_W = P_bat_kW * 1000.0
        SOC_delta, SOCbat_new, bat_done, bat_info = self.fchev.run_power_battery(P_bat_W, SOCbat)

        # 电池等效氢耗 (参照 ref/agentEMS.py: h2_batt = P_batt/1000 * h2_conv_coef)
        try:
            h2_conv_coef = self.fchev.get_h2_conv_coef(max(P_fc_kW, 1.0))
        except Exception:
            h2_conv_coef = 60.0
        if P_bat_kW >= 0:
            Cbat = P_bat_kW * h2_conv_coef  # g/s (匹配 ref 公式, 单位已由插值保证)
        else:
            Cbat = 0.0

        # ---- UC 模型 ----
        P_uc_W = P_uc_kW * 1000.0
        E_uc_max_J = 0.5 * self.C_uc * (self.V_uc_max**2 - self.V_uc_min**2)
        SOCuc_delta = -P_uc_W * 1.0 / E_uc_max_J
        SOCuc_new = SOCuc + SOCuc_delta
        SOCuc_new = np.clip(SOCuc_new, 0.0, 1.0)

        # UC 等效氢耗 (与电池同公式)
        if P_uc_kW >= 0:
            Cuc = P_uc_kW * h2_conv_coef
        else:
            Cuc = 0.0

        # ---- 等效氢耗 (参照 ref: h2_equal = h2_fcs + h2_batt) ----
        Cfc = h2_fcs
        C_total = Cfc + Cbat + Cuc  # g/s, 无 K 因子膨胀

        # ---- 奖励 (参照 ref/agentEMS.py get_reward) ----
        h2_price = 55.0 / 1000.0       # ￥/g
        eq_h2_money = h2_price * (Cfc + Cbat + Cuc)  # 等效氢耗费用

        # FC 衰退费用
        try:
            FCS_De, _ = self.fchev.run_FC_SOH(max(P_fc_kW, 0.0))
            self.fc_degradation += FCS_De
        except Exception:
            FCS_De = 0.0
        FCS_price = 300000.0
        FCS_money = 10.0 * FCS_price * FCS_De

        # SOC 惩罚
        w_soc = 20.0
        soc_cost = w_soc * abs(SOCbat - self.SOCbat_opt)

        # 电池放电惩罚
        if P_bat_kW > 0:
            batt_discharge_penalty = 0.02 * P_bat_kW
        else:
            batt_discharge_penalty = 0.0

        # FC 功率平滑惩罚
        delta_P_fc = abs(P_fc_kW - getattr(self, '_prev_P_fc', P_fc_kW))
        self._prev_P_fc = P_fc_kW
        if delta_P_fc == 0:
            P_fc_smooth_penalty = 0.0
        else:
            P_fc_smooth_penalty = 0.02 * delta_P_fc

        # 效率奖励
        current_eff = fc_info.get('fce_eff', 0.4) if fc_info else 0.4
        eff_threshold = 0.4
        eff_penalty_weight = 10.0
        r_eff = -eff_penalty_weight * (eff_threshold - current_eff)

        # 总奖励
        r = -(eq_h2_money + FCS_money + soc_cost + P_fc_smooth_penalty + batt_discharge_penalty) + r_eff

        # ---- FC 衰退 (已在上方计算) ----
        # FCS_De 已计算

        # ---- 累计指标 ----
        self.h2_total += h2_fcs
        self.h2_eq_total += C_total

        # ---- 时间步推进 ----
        self.time_step += 1
        self.SOCbat = SOCbat_new
        self.SOCuc = SOCuc_new
        self.step_count += 1

        # ---- 检查终止（仅电池 SOC 极端时终止，UC 不终止仅惩罚） ----
        done = bat_done
        done = done or (self.SOCbat < 0.3) or (self.SOCbat > 0.9)

        # ---- 构建下一状态 ----
        t_next = self.time_step % UDDS_LENGTH
        velocity_next = UDDS_SPEED[t_next]
        Pdemand_next = self.Pdemand_list[t_next]
        Puc_next = self.Puc_values[np.random.randint(0, 5)]

        next_state = np.array([self.SOCbat, self.SOCuc, velocity_next, Pdemand_next, Puc_next],
                              dtype=np.float32)

        # ---- 存储详细信息供外部记录 ----
        self._last_info = {
            'P_fc_kW': P_fc_kW,
            'P_bat_kW': P_bat_kW,
            'P_uc_kW': P_uc_kW,
            'h2_fcs': h2_fcs,
            'C_total': C_total,
            'Cfc': Cfc,
            'Cbat': Cbat,
            'Cuc': Cuc,
            'SOCbat_delta': SOC_delta,
            'SOCuc_delta': SOCuc_delta,
            'FCS_De': FCS_De,
            'eq_h2_money': eq_h2_money,
            'soc_cost': soc_cost,
        }

        return r, next_state, done

    def get_last_info(self):
        return getattr(self, '_last_info', {})
