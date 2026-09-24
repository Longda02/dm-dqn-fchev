import numpy as np
import pandas as pd
import math

#   读取状态表和TPM矩阵
state_list = pd.read_excel("Excel/state-2.xlsx", nrows=8225, sheet_name='Sheet1', index_col=False, header=None)
TPM_10 = pd.read_excel('Excel/TPMfinal10.xlsx', sheet_name='Sheet1', index_col=False, header=None)
TPM_20 = pd.read_excel('Excel/TPMfinal20.xlsx', sheet_name='Sheet1', index_col=False, header=None)
TPM_30 = pd.read_excel('Excel/TPMfinal30.xlsx', sheet_name='Sheet1', index_col=False, header=None)
TPM_40 = pd.read_excel('Excel/TPMfinal40.xlsx', sheet_name='Sheet1', index_col=False, header=None)
TPM_50 = pd.read_excel('Excel/TPMfinal50.xlsx', sheet_name='Sheet1', index_col=False, header=None)
action_space = pd.read_excel('Excel/action space 21.xlsx', index_col=False, header=None)

# 保留小数点位数
num_bit_state = 2

class env(object):
    def __init__(self):
        self.stationary_counting = 0
        self.s__index = 0
        self.num = 1
        self.i = 0
        self.power_index_0 = 0
        self.Q_loss_bat = 0
        self.Q_loss_fc = 0
        self.Ah = 0
        self.alpha_bat = 0
        self.beta_bat = 0
        self.num_fc = 0
        self.Initial_current = 0

    def feedback(self, a_index, s_index):
        SOCbat = state_list.values[s_index, 0]
        SOCuc = state_list.values[s_index, 1]
        velocity = state_list.values[s_index, 2]
        Pdemand = state_list.values[s_index, 3]
        Puc = state_list.values[s_index, 4]
        power_index = 2 * Pdemand
        TPM_current_index = int(power_index)
        if velocity == 10:
            probability_cur2fut = TPM_10.iloc[TPM_current_index, :]
        elif velocity == 20:
            probability_cur2fut = TPM_20.iloc[TPM_current_index, :]
        elif velocity == 30:
            probability_cur2fut = TPM_30.iloc[TPM_current_index, :]
        elif velocity == 40:
            probability_cur2fut = TPM_40.iloc[TPM_current_index, :]
        elif velocity == 50:
            probability_cur2fut = TPM_50.iloc[TPM_current_index, :]
            probability_cur2fut = probability_cur2fut.values.reshape(47, 1)
        future_power_set = np.arange(0, 235, 5).reshape(1, 47)
        future_power_set = future_power_set / 10
        for value in future_power_set:
            future_power_expectation = np.dot(future_power_set, probability_cur2fut)
            n = int(future_power_expectation)
            m = future_power_expectation - n
        if m >= 0.25 and m < 0.75:
            m = 0.5
            future_power_expectation = n + m
        elif m < 0.25 and 0 <= m:
            m = 0
            future_power_expectation = n + m
        elif m >= 0.75:
            m = 0
            future_power_expectation = n + m + 1

        next_SOCbat = state_list.values[s_index, 0]
        next_SOCuc = state_list.values[s_index, 1]
        next_velocity = state_list.values[s_index, 2]
        next_Pdemand = future_power_expectation
        next_Puc = state_list.values[s_index, 4]
        s_ = [next_SOCbat, next_SOCuc, next_velocity, next_Pdemand, next_Puc]
        if (state_list.values[s_index, 0] == next_SOCbat) and (state_list.values[s_index, 1] == next_SOCuc) \
                and (state_list.values[s_index, 4] == next_Puc) and (state_list.values[s_index, 3] == next_Pdemand):
            self.num = 0
            self.s__index = s_index
        else:
            num_index = math.ceil((s_index + 1) / 235)
            for i in range((num_index - 1) * 235, num_index * 235):
                if state_list.values[i, 4] == next_Puc:
                    if state_list.values[i, 3] == next_Pdemand:
                        self.s__index = i
                        break
                    else:
                        continue
                else:
                    continue
        s_[2] = round((s_[2] - 10) / 40, num_bit_state)
        s_[3] = round(s_[3] / 23, num_bit_state)
        Puc_input = next_Puc
        a = action_space.values[0, a_index]
        Pfc_input = a
        Pbat_input = action_space.values[1, a_index]
        Mh2 = 2
        Ncell = 70
        F = 96487
        Ubus = 380
        Pfc_rating = 10000
        # Transition of units from watt to kilowatt
        Puc = Pdemand * Puc_input * 1000
        Pfc = (Pdemand - Pdemand * Puc_input) * Pfc_input * 1000
        Pbat = (Pdemand - Pdemand * Puc_input) * Pbat_input * 1000
        Ifc = Pfc / Ubus  # Current of fuel cell
        Ibat = Pbat / Ubus   # Current of battery
        Cfc = (1.2 * Mh2 * Ncell * Ifc) / (2 * F)  # Hydrogen Consumption of Fuel Cells
        # Cbat Computational Formula ,the inputs is Pbat,Ndisch_bat,Nch_bat
        Cfc_ave = Cfc  # Average (Instantaneous) hydrogen consumption of fuel cell
        Nch_ave_bat = 0.95  # Average charging efficiency of battery
        Pfc_ave = Pfc  # Average (Instantaneous) power of fuel cell
        Ndisch_ave_bat = 0.95  # Average discharging efficiency of battery

        if Pbat >= 0:
            Cbat = (Pbat * Cfc_ave) / (Ndisch_ave_bat * Nch_ave_bat * Pfc_ave)  # the battery discharge
        else:
            Cbat = (Pbat * Ndisch_ave_bat * Nch_ave_bat * Cfc_ave) / Pfc_ave  # the battery charge
        # Cuc Computational Formula ,the inputs is Puc,Ndisch_uc,Nch_uc
        Nch_ave_uc = 0.95  # Average charging efficiency of ultercapacitor
        Ndis_ave_uc = 0.95  # Average discharging efficiency of ultercapacitor

        if Puc >= 0:
            Cuc = (Puc * Cfc_ave) / (Ndis_ave_uc * Nch_ave_uc * Pfc_ave)  # the ultercapacitor discharge
        else:
            Cuc = (Puc * Ndis_ave_uc * Nch_ave_uc * Cfc_ave) / Pfc_ave  # the ultercapacitor charge
        # Kfc Computational Formula ,the input is Nfc
        Nfc_opt = 0.7  # Optimal efficiency value of fuel cell
        Nfc_max = 0.8  # Maximum efficiency value of fuel cell
        Nfc_min = 0.6  # Minimum efficiency value of fuel cell
        Nfc = Pfc / Pfc_rating  # Nfc
        Kfc = (1 - 2 * (Nfc - Nfc_opt) / (Nfc_max - Nfc_min)) ** 4  # Fuel cell penalty function
        # Kbat Computational Formula ,the input is SOCbat
        SOCbat_opt = 0.7  # Optimal SOC value of battery
        SOCbat_max = 0.8  # Maximum SOC value of battery
        SOCbat_min = 0.4  # Minimum SOC value of battery
        if SOCbat >= SOCbat_min and SOCbat <= SOCbat_max:
            Kbat = (1 - 2 * (SOCbat - SOCbat_opt) / (
                    SOCbat_max - SOCbat_min)) ** 8  # SOC penalty function for battery in safety range
        else:
            Kbat = (1 - 2 * (SOCbat - SOCbat_opt) / (
                    SOCbat_max - SOCbat_min)) ** 16  # SOC penalty function for battery not in safety range

        # Kuc Computational Formula, the input is SOCuc
        SOCuc_opt = 0.8  # Optimal SOC value of ultercapacitor
        SOCuc_max = 1  # Maximum SOC value of ultercapacitor
        SOCuc_min = 0.4  # Minimum SOC value of ultercapacitor
        if SOCuc > 0.5 and SOCuc <= 0.8:
            Kuc = (1 - 2 * (SOCuc - SOCuc_opt) / (
                    SOCuc_max - SOCuc_min)) ** 2  # SOC penalty function for ultercapacitor in safety range
        else:
            if SOCuc >= 0.4 and SOCuc <= 0.5:
                Kuc = (1 - 2 * (SOCuc - SOCuc_opt) / (
                        SOCuc_max - SOCuc_min)) ** 8  # SOC penalty function for ultercapacitor in subsafety range
            else:
                Kuc = (1 - 2 * (SOCuc - SOCuc_opt) / (
                        SOCuc_max - SOCuc_min)) ** 16  # SOC penalty function for ultercapacitor not in safety range

        # Compute the total reward with total consumption and variation of SOC
        beta = 1000  # Transition factor to make two items with the same magnitudes

        C_total = Kfc * Cfc + Kbat * Cbat + Kuc * Cuc
        delta_SOC = SOCbat - SOCbat_opt
        Reward = (C_total + beta * (delta_SOC) ** 2)
        r = -(Reward - 0.00228) / (1233.714 - 0.00228)
        # if (a >= 0.95) or (a <= 0.05):
        #     r = r - 1
        if (np.isnan(r)):
            r = -1
        # print('Reward', r)
        return r, s_, self.s__index