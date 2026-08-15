# K1 Parkour 训练诊断资源

## Knowledge

- [`params/env.yaml`](logs/instinct_rl/k1_parkour/20260815_022431_k1_hiking_amp_19clips_v1/params/env.yaml)
  本次训练保存的环境、奖励、终止和课程配置；用于确认每条 Episode 曲线的真实定义。
- [`params/agent.yaml`](logs/instinct_rl/k1_parkour/20260815_022431_k1_hiking_amp_19clips_v1/params/agent.yaml)
  本次训练保存的 PPO、AMP、MoE 和优化器配置；用于解释 Loss、Policy 和 Train 指标。
- [`reward_manager.py`](source/instinctlab/instinctlab/managers/reward_manager.py)
  奖励日志三种统计后缀的实现来源。
- [`wasabi.py`](source/instinct_rl/instinct_rl/algorithms/wasabi.py)
  AMP 判别器、判别器奖励和损失的实现来源。

## Gaps

- 当前日志没有逐地形成功率，也没有直接记录深度编码器是否真正影响动作；需要额外评估指标或消融实验。

## Wisdom (Communities)

- 当前优先依据项目代码、仿真视频和实机实验闭环，不额外引入社区建议。
