# Mission: Booster K1 Hiking in the Wild 训练诊断

## Why
能够从 TensorBoard、仿真回放和实机测试中判断 K1 是否真正学会了依靠深度感知通过复杂地形，而不是只看总奖励或单条课程曲线。

## Success looks like
- 能解释当前 K1 Parkour 训练日志中的每一组指标
- 能区分策略进步、课程推进、AMP 动作质量与训练数值稳定性
- 能根据曲线决定继续训练、调整奖励或排查失败模式

## Constraints
- 以当前 InstinctLab K1 配置和实际 TensorBoard tags 为准
- 优先服务 Hiking in the Wild 复现与后续实机部署

## Out of scope
- 泛化到与当前 K1 Parkour 任务无关的强化学习算法
