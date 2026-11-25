# DM-DQN for FCHEV Energy Management

本仓库包含用于燃料电池混合动力汽车能量管理的 DM-DQN 研究代码及必要的 Excel 状态/动作数据。

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

## 许可

当前未附加开源许可证。在版权所有者明确授权之前，请勿假定本仓库内容可被复制、修改或再分发。
