# DM-DQN for FCHEV Energy Management

本仓库整理自 `F:\徐鹏飞整理材料\DM-DQN代码\DM-DQN`，包含用于燃料电池混合动力汽车能量管理的 DM-DQN 研究代码及必要的 Excel 状态/动作数据。

## 主要内容

- `main.py`：训练入口
- `Agent.py`、`networks.py`、`Replay.py`：智能体、网络和经验回放
- `UDDS_Environment.py`、`Enviroment.py`：环境模型
- `FCHEV_SOH.py`：燃料电池混合动力系统模型
- `Excel/`：状态空间、动作空间和转移概率矩阵
- `项目代码分析报告.md`：原项目分析说明

## Python 依赖

建议使用 Python 3.8 创建独立环境：

```bash
python -m venv .venv
pip install -r requirements.txt
```

## 当前已知限制

本次整理未找到下列外部数据，因此仓库是“研究代码归档”，尚不能开箱即跑：

- `D:/Drive_Cycle/Standard_UDDS.mat`
- `P_fc.mat`、`P_fce.mat`、`h2_consumption.mat`、`fce_eff.mat`
- `conversion.mat`、`P_fc_conv.mat`
- `mot_eff.mat`、`W_mot.mat`、`T_mot.mat`
- `mot_trq_min.mat`、`mot_trq_max.mat`、`e_dcdc.mat`

`UDDS_Environment.py` 和 `FCHEV_SOH.py` 中还存在本机绝对路径。运行前应把这些路径改为项目内相对路径或命令行参数。

## 未纳入版本库的内容

- `venv/`、`__pycache__/`：本地环境和缓存
- `UDDS_results/`：训练中间结果、模型输出和日志
- `EXCEL_ALL/`、`Train_Excel/`、`DM-DQN_model/`：实验输出
- `ref/`：依赖缺失的参考代码，不属于当前可复现主流程

## 许可

当前未附加开源许可证。在版权所有者明确授权之前，请勿假定本仓库内容可被复制、修改或再分发。
