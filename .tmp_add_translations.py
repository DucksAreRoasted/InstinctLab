# -*- coding: utf-8 -*-
"""给论文阅读笔记中裸露的英文术语补充中文翻译（临时脚本，用完即删）。"""
import sys

P132 = "/home/ducks/humanoid-motion-intelligence/论文与项目/论文逐篇解读/P132_Hiking_in_the_Wild_论文阅读笔记.md"
P125 = "/home/ducks/humanoid-motion-intelligence/论文与项目/论文逐篇解读/P125_DreamWaQ_论文阅读笔记.md"

m132 = [
    ("> **关键词**：Humanoid Locomotion、Depth Image、PPO、Asymmetric Actor-Critic、MoE、Terrain Edge Detection、Flat Patch Sampling、AMP、Zero-shot Sim-to-Real",
     "> **关键词**：Humanoid Locomotion（人形机器人运动控制）、Depth Image（深度图像）、PPO（近端策略优化）、Asymmetric Actor-Critic（非对称 Actor-Critic）、MoE（专家混合）、Terrain Edge Detection（地形边缘检测）、Flat Patch Sampling（平坦区域采样）、AMP（对抗运动先验）、Zero-shot Sim-to-Real（零样本仿真到真机迁移）"),
    ("- IMU；", "- IMU（惯性测量单元）；"),
    ("- joint position；", "- joint position（关节位置）；"),
    ("- joint velocity；", "- joint velocity（关节速度）；"),
    ("- contact-induced dynamics；", "- contact-induced dynamics（接触诱导的动力学）；"),
    ("它属于 **reactive control**。", "它属于 **reactive control（反应式控制）**。"),
    ("必须从 **reactive stability** 转向 **proactive perception**，也就是“looking ahead”。",
     "必须从 **reactive stability（反应式稳定）** 转向 **proactive perception（主动感知）**，也就是“looking ahead（向前看）”。"),
    ("1. **依赖准确 state estimation**：", "1. **依赖准确 state estimation（状态估计）**："),
    ("LiDAR 容易出现 motion distortion；", "LiDAR 容易出现 motion distortion（运动畸变）；"),
    ("提高 value estimation 和 advantage estimation 的质量", "提高 value estimation（价值估计）和 advantage estimation（优势估计）的质量"),
    ("- $r_{\\mathrm{task}}$：velocity / heading / navigation tracking；",
     "- $r_{\\mathrm{task}}$：velocity / heading / navigation tracking（速度/航向/导航跟踪）；"),
    ("- $r_{\\mathrm{reg}}$：能耗、平滑性、姿态等 regularization；",
     "- $r_{\\mathrm{reg}}$：能耗、平滑性、姿态等 regularization（正则化）；"),
    ("- $r_{\\mathrm{safe}}$：危险接触与 terrain edge 安全；",
     "- $r_{\\mathrm{safe}}$：危险接触与 terrain edge（地形边缘）安全；"),
    ("- $r_{\\mathrm{amp}}$：motion prior 风格奖励。",
     "- $r_{\\mathrm{amp}}$：motion prior（运动先验）风格奖励。"),
    ("作者利用 **NVIDIA Warp + GPU parallel ray casting** 生成高频深度图。",
     "作者利用 **NVIDIA Warp + GPU parallel ray casting（GPU 并行光线投射）** 生成高频深度图。"),
    ("其中 $\\mathbf{n}_c$ 是 camera forward direction。",
     "其中 $\\mathbf{n}_c$ 是 camera forward direction（相机前向方向）。"),
    ("双方在中间 perception space 靠近。", "双方在中间 perception space（感知空间）靠近。"),
    ("- stereo matching failure；", "- stereo matching failure（立体匹配失败）；"),
    ("模拟 motion blur 和 sensor jitter。", "模拟 motion blur（运动模糊）和 sensor jitter（传感器抖动）。"),
    ("其中 inpainting 主要恢复真实深度相机中的 zero-valued / black invalid regions。",
     "其中 inpainting（图像修复）主要恢复真实深度相机中的 zero-valued / black invalid regions（零值/黑色无效区域）。"),
    ("- 什么时候调整 gait phase。", "- 什么时候调整 gait phase（步态相位）。"),
    ("- $m$：输入 depth frame 数；", "- $m$：输入 depth frame（深度帧）数；"),
    ("- $\\ell$：temporal stride。", "- $\\ell$：temporal stride（时间步长）。"),
    ("个 simulation steps。", "个 simulation steps（仿真步）。"),
    ("- relative motion；\n- obstacle approaching trend；\n- time-to-contact；\n- terrain profile evolution。",
     "- relative motion（相对运动）；\n- obstacle approaching trend（障碍接近趋势）；\n- time-to-contact（接触时间）；\n- terrain profile evolution（地形轮廓演化）。"),
    ("- unpredictable contact dynamics；", "- unpredictable contact dynamics（不可预测的接触动力学）；"),
    ("因此提出 edge-aware safety constraint。", "因此提出 edge-aware safety constraint（边缘感知安全约束）。"),
    ("agent 可能学会原地转圈等 reward hacking 行为", "agent 可能学会原地转圈等 reward hacking（奖励欺骗）行为"),
    ("作者改为先在 terrain mesh 上找“真实可站立”的 flat patch。",
     "作者改为先在 terrain mesh（地形网格）上找“真实可站立”的 flat patch（平坦区域）。"),
    ("在半径 $r$ 内 ray cast 得到高度集合：", "在半径 $r$ 内 ray cast（光线投射）得到高度集合："),
    ("这一步不是 foothold planning，而是：", "这一步不是 foothold planning（落脚点规划），而是："),
    ("> **给 RL 生成合理、可达、确实需要跨越地形才能到达的 navigation target。**",
     "> **给 RL 生成合理、可达、确实需要跨越地形才能到达的 navigation target（导航目标）。**"),
    ("将目标转换到机器人 base frame：", "将目标转换到机器人 base frame（基座坐标系）："),
    ("以学习 in-place turning。", "以学习 in-place turning（原地转向）。"),
    ("为了减轻 mode collapse，walking 与 running 分开训练。",
     "为了减轻 mode collapse（模式坍缩），walking（行走）与 running（跑步）分开训练。"),
    ("论文采用 Least-Squares 形式：", "论文采用 Least-Squares（最小二乘）形式："),
    ("作者认为这种 MSE + quadratic reward 的梯度更平滑、更稳定。",
     "作者认为这种 MSE + quadratic reward（二次奖励）的梯度更平滑、更稳定。"),
    ("而更像一个 policy capacity 设计。", "而更像一个 policy capacity（策略容量）设计。"),
    ("先 downsample：", "先 downsample（下采样）："),
    ("以降低 latency。", "以降低 latency（延迟）。"),
    ("| Terrain | Ours SR | No Edge SR | Ours Landing Area | No Edge |",
     "| Terrain（地形） | Ours SR（成功率） | No Edge SR（无边缘成功率） | Ours Landing Area（落地面积） | No Edge（无边缘） |"),
    ("说明 edge penalty 对狭窄、离散 foothold 的意义尤其大。",
     "说明 edge penalty（边缘惩罚）对狭窄、离散 foothold（落脚点）的意义尤其大。"),
    ("| Method | Success Rate |", "| Method（方法） | Success Rate（成功率） |"),
    ("并正确决定动作 timing。", "并正确决定动作 timing（时机）。"),
    ("说明 vanilla MLP 在复杂、多模态行为上容量不足。",
     "说明 vanilla MLP（普通多层感知机）在复杂、多模态行为上容量不足。"),
    ("| Pure Proprioception | Depth Perception |", "| Pure Proprioception（纯本体感知） | Depth Perception（深度感知） |"),
    ("| Reflex | Anticipation |", "| Reflex（反射） | Anticipation（预测） |"),
    ("**Proprioception 提供 robust reflex，Vision 提供 anticipatory adjustment。**",
     "**Proprioception 提供 robust reflex（鲁棒反射），Vision 提供 anticipatory adjustment（提前调整）。**"),
    ("- backward locomotion；\n- lateral locomotion；\n- omnidirectional perception。",
     "- backward locomotion（后退运动）；\n- lateral locomotion（侧向运动）；\n- omnidirectional perception（全向感知）。"),
    ("因此 walking 与 running 仍存在 specialized policy / post-training。",
     "因此 walking 与 running 仍存在 specialized policy / post-training（专用策略/后训练）。"),
    ("- previous action；", "- previous action（上一动作）；"),
    ("- command。", "- command（速度指令）。"),
]

m125 = [
    ("> **核心关键词**：Proprioception、Asymmetric Actor-Critic、PPO、CENet、β-VAE、AdaBoot、Sim-to-Real",
     "> **核心关键词**：Proprioception（本体感知）、Asymmetric Actor-Critic（非对称 Actor-Critic）、PPO（近端策略优化）、CENet（上下文辅助估计网络）、β-VAE（β 变分自编码器）、AdaBoot（自适应引导）、Sim-to-Real（仿真到真机迁移）"),
    ("仅利用 IMU、关节编码器等本体感知信息", "仅利用 IMU（惯性测量单元）、关节编码器等本体感知信息"),
    ("PPO 再通过 advantage / policy gradient，把这种更高质量的评价反馈给 Actor。",
     "PPO 再通过 advantage / policy gradient（优势/策略梯度），把这种更高质量的评价反馈给 Actor。"),
    ("然后通过 Behavior Cloning：", "然后通过 Behavior Cloning（行为克隆）："),
    ("这也是论文所谓 **implicit terrain imagination** 的来源。",
     "这也是论文所谓 **implicit terrain imagination（隐式地形想象）** 的来源。"),
    ("而是从身体动态变化中形成一个与地形相关的 latent representation。",
     "而是从身体动态变化中形成一个与地形相关的 latent representation（隐式表征）。"),
    ("CENet = **Context-Aided Estimator Network**", "CENet = **Context-Aided Estimator Network（上下文辅助估计网络）**"),
    ("> **速度估计和环境表征共用一个 Encoder。**", "> **速度估计和环境表征共用一个 Encoder（编码器）。**"),
    ("通过 auto-encoder / β-VAE 学习系统动态", "通过 auto-encoder（自编码器）/ β-VAE 学习系统动态"),
    ("作者还提出了 **Adaptive Bootstrapping，AdaBoot**。",
     "作者还提出了 **Adaptive Bootstrapping（自适应引导），AdaBoot**。"),
    ("这里的 bootstrapping 可以理解为：", "这里的 bootstrapping（引导）可以理解为："),
    ("训练时有时故意使用 estimator 输出，而不是始终使用 simulator ground truth。",
     "训练时有时故意使用 estimator（估计器）输出，而不是始终使用 simulator ground truth（仿真器真值）。"),
    ("作者根据多个 domain-randomized 环境中 episode reward 的变异系数：",
     "作者根据多个 domain-randomized（域随机化）环境中 episode reward（回合奖励）的变异系数："),
    ("作者还使用 grid-adaptive curriculum 改善低速转弯时的稳定性",
     "作者还使用 grid-adaptive curriculum（网格自适应课程）改善低速转弯时的稳定性"),
    ("1. **Baseline**\n   - 无 adaptation", "1. **Baseline（基线）**\n   - 无 adaptation（自适应）"),
    ("2. **AdaptationNet**\n   - Teacher-Student\n   - 隐式环境编码",
     "2. **AdaptationNet（自适应网络）**\n   - Teacher-Student（教师-学生）\n   - 隐式环境编码"),
    ("3. **EstimatorNet**\n   - 显式状态估计\n   - 没有 context estimation",
     "3. **EstimatorNet（估计网络）**\n   - 显式状态估计\n   - 没有 context estimation（上下文估计）"),
    ("4. **DreamWaQ w/o AdaBoot**", "4. **DreamWaQ w/o AdaBoot（无 AdaBoot）**"),
    ("5. **DreamWaQ w/ AdaBoot**", "5. **DreamWaQ w/ AdaBoot（含 AdaBoot）**"),
    ("- **Oracle Policy**\n- 可以直接访问 terrain height map",
     "- **Oracle Policy（预言机策略）**\n- 可以直接访问 terrain height map（地形高度图）"),
    ("- motor heating\n- power distribution\n- robustness",
     "- motor heating（电机发热）\n- power distribution（功率分布）\n- robustness（鲁棒性）"),
    ("> **self-supervised dynamics representation learning 思想。**",
     "> **self-supervised dynamics representation learning（自监督动力学表征学习）思想。**"),
    ("不是固定地把 estimator noise 输入 Actor，而是根据训练稳定程度逐渐增加。",
     "不是固定地把 estimator noise（估计器噪声）输入 Actor，而是根据训练稳定程度逐渐增加。"),
    ("> estimator 在早期不准确，会反过来破坏 policy learning",
     "> estimator（估计器）在早期不准确，会反过来破坏 policy learning（策略学习）"),
    ("DreamWaQ 非常适合作为一个 **pure proprioceptive locomotion baseline**。",
     "DreamWaQ 非常适合作为一个 **pure proprioceptive locomotion baseline（纯本体感知运动基线）**。"),
    ("> **robust reflexive locomotion backbone**", "> **robust reflexive locomotion backbone（鲁棒的反射式运动主干）**"),
    ("而不是完整的 perceptive locomotion 终点方案。", "而不是完整的 perceptive locomotion（感知运动）终点方案。"),
    ("- body velocity\n- environment context", "- body velocity（身体线速度）\n- environment context（环境上下文）"),
    ("通过预测下一时刻 observation 学习隐式 dynamics/context。",
     "通过预测下一时刻 observation（观测）学习隐式 dynamics/context（动力学/上下文）。"),
    ("根据训练稳定程度动态提高 estimator bootstrapping 概率。",
     "根据训练稳定程度动态提高 estimator bootstrapping（估计器引导）概率。"),
    ("在一个已经训练好的 proprioceptive locomotion policy 上，再加入 depth vision 专门负责",
     "在一个已经训练好的 proprioceptive locomotion policy（本体感知运动策略）上，再加入 depth vision（深度视觉）专门负责"),
]


def apply(path, mappings):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    n_miss = 0
    for old, new in mappings:
        c = text.count(old)
        if c == 0:
            n_miss += 1
            print(f"[MISS] {old[:60]!r}")
            continue
        text = text.replace(old, new)
        print(f"[{c}x] {old[:60]!r}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"=== {path.split('/')[-1]} done, miss={n_miss} ===")
    return n_miss


if __name__ == "__main__":
    a = apply(P132, m132)
    b = apply(P125, m125)
    sys.exit(1 if (a + b) else 0)
