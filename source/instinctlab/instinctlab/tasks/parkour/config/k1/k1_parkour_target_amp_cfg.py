# Copyright (c) 2025, BoosterRobotics
# SPDX-License-Identifier: BSD-3-Clause

"""Booster K1 对官方 Hiking in the Wild（越野跑）parkour 任务的适配层。

本文件是 K1 训练代码中机器人专属部分的核心：继承 G1 版本的越野跑
任务配置 G1ParkourRoughEnvCfg，并在 K1ParkourConfigMixin.apply_k1_config()
中把所有与机器人形态学相关的部分替换为 K1 的：机器人资产、AMP 动作
参考集、动作空间缩放、深度相机（按真机 ROS 标定参数模拟）、足部高度
扫描仪、足部体积点云，以及各奖励项/终止项中的身体与关节名称映射。

任务注册见同目录 __init__.py，训练算法与网络配置见
agents/instinct_rl_amp_cfg.py。

本文件用到的 Isaac Lab 语法规则速查：
1. @configclass：本质是 dataclass（支持继承、序列化），配置对象可变，
   既可 copy.deepcopy 也可实例化后直接改字段。
2. 配置继承 + Mixin：任务配置层层继承，G1 定义任务主体（奖励/观测/
   地形），K1 用 Mixin 在 __post_init__ 里打补丁。__post_init__ 是
   dataclass 钩子，实例化完成后自动调用。
3. {ENV_REGEX_NS}：prim 路径模板字符串，代表所有环境实例的命名空间
   /World/envs/env_.*，正则形式可一次性匹配所有 env。
4. 名字正则：引号内的 .*、(?!...)、(a|b) 都是正则表达式，在 USD
   刚体/关节名上做全匹配，是 Isaac Lab 批量指定身体的惯用法。
5. SceneEntityCfg：奖励/观测/终止函数通过它引用场景实体，asset_cfg
   解析机器人资产上的 body/joint，sensor_cfg 解析 scene 里注册的
   传感器，解析时同样支持正则。
6. 六大管理器：ManagerBasedRLEnvCfg 的环境由 scene、observations
   （分 policy/critic 组）、actions、rewards、terminations、commands、
   events（随机化）、curriculum（课程）组成，self.xxx 访问的都是它们。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

from isaaclab.utils import configclass

from instinctlab.assets.booster_k1 import (
    BOOSTER_K1_CFG,
    K1_ACTION_SCALE,
    K1_JOINT_NAMES,
    K1_LINK_NAMES,
    K1_SYMMETRY_JOINT_MAPPING,
    K1_SYMMETRY_JOINT_SIGNS,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.sensors import get_link_prim_targets
from instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import (
    G1ParkourRoughEnvCfg,
    G1ParkourRoughEnvCfg_PLAY,
)
from instinctlab.utils.noise import RangeBasedGaussianNoiseCfg

# AMP 动作参考所用的关键刚体（K1 的 Trunk 对应 G1 的 pelvis）。
# 判别器只比较这些 link 的位姿（位置+朝向），而不是全部 23 个 link，
# 既能抓住动作风格，又不会让参考状态维度过高。
K1_PARKOUR_LINKS = [
    "Trunk",
    "Head_2",
    "Left_Arm_3",
    "Right_Arm_3",
    "left_hand_link",
    "right_hand_link",
    "Left_Hip_Yaw",
    "Right_Hip_Yaw",
    "Left_Shank",
    "Right_Shank",
    "left_foot_link",
    "right_foot_link",
]
# 对称增广映射：index 为 i 的 link 的镜像 partner 是 index 为
# K1_PARKOUR_LINK_SYMMETRY[i] 的 link。例如 0 -> 0（Trunk 自身对称），
# 2(Left_Arm_3) -> 3(Right_Arm_3)，10(left_foot_link) -> 11(right_foot_link)。
K1_PARKOUR_LINK_SYMMETRY = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]

# 真机 /boostercamera/head/depth/camera_info 话题的实时标定参数（内参）。
# 仿真中的射线图像保持较小分辨率（64x36）以保证训练吞吐，而其视场角
# （aperture）仍与标定后的完整传感器视场一致，即"图像变小、视角不变"。
K1_DEPTH_WIDTH_PX = 544          # 真机深度传感器全分辨率（像素）
K1_DEPTH_HEIGHT_PX = 448
K1_DEPTH_FOCAL_LENGTH_PX = 210.77337743177728     # 焦距（像素）
K1_DEPTH_PRINCIPAL_X_PX = 241.36559391021729      # 主点 x（像素，不居中有偏移）
K1_DEPTH_PRINCIPAL_Y_PX = 217.1773950913373       # 主点 y
K1_DEPTH_UPDATE_PERIOD_S = 0.05   # 深度相机刷新周期：0.05s = 20 Hz
# 深度观测保留的历史帧数与降采样间隔：
# 历史缓冲保存最近 16 帧，观测时每隔 2 帧取 1 帧（见下方 obs 相关注释）。
K1_DEPTH_HISTORY_LENGTH = 16
K1_DEPTH_HISTORY_SKIP_FRAMES = 2
# K1 动作参考数据包目录（GMR 重定向后的越野跑动作），可用环境变量覆盖。
# parents[7]：__file__ 位于 InstinctLab/source/instinctlab/instinctlab/
# tasks/parkour/config/k1/ 下，向上 7 层回到仓库根目录 InstinctLab/。
K1_MOTION_PACKAGE_DIR = Path(__file__).resolve().parents[7] / "parkour_motion_reference" / "booster_k1"

# 复制一份 K1 资产配置作为本任务专用。
# copy.deepcopy 是必须的：BOOSTER_K1_CFG 是 assets/booster_k1.py 中的
# 模块级常量，被多个任务共享；配置对象可变，直接赋值只复制引用，
# 改它就会污染其他任务（比如平地行走任务）。Isaac Lab 中共享配置
# 一律先 deepcopy 再改。
K1_CFG = copy.deepcopy(BOOSTER_K1_CFG)
# merge_fixed_joints=True：把 URDF 中的固定关节合并进父 link，
# 减少刚体/自由度数量，加快仿真，代价是这些关节不能再被独立控制。
K1_CFG.spawn.merge_fixed_joints = True
# 机器人根刚体（Trunk）的初始位姿：z=0.57 为 K1 站立时 Trunk 的标称高度。
# 复位事件会在其基础上随机化，故这里只是出生位置。
K1_CFG.init_state.pos = (0.0, 0.0, 0.57)


@configclass
class K1AmassMotionCfg(AmassMotionCfgBase):
    """K1 动作参考数据配置：由 GMR 的 smplx_to_k1.json 约束重定向而来。

    AMP（Adversarial Motion Priors）判别器需要参考动作数据集作为专家示范，
    本类描述这份数据集从哪里读、怎么采样。
    """

    # 数据目录，可用环境变量 INSTINCTLAB_K1_MOTION_DIR 覆盖
    path = os.path.expanduser(os.environ.get("INSTINCTLAB_K1_MOTION_DIR", str(K1_MOTION_PACKAGE_DIR)))
    # 动作已离线重定向到 K1，运行时不再做 SMPL 重定向（置 None 跳过该步骤）
    retargetting_func = None
    # 动作片段清单 yaml（G1 版本用它过滤掉不适合 parkour 的动作），
    # 可用环境变量 INSTINCTLAB_K1_MOTION_SELECTION 覆盖
    filtered_motion_selection_filepath = os.path.expanduser(
        os.environ.get("INSTINCTLAB_K1_MOTION_SELECTION", str(Path(path) / "motions.yaml"))
    )
    # 从动作片段长度的 [0, 0.9] 区间随机选起始帧（比例），
    # 避免每个 episode 都从同一姿势开始播
    motion_start_from_middle_range = [0.0, 0.9]
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    # 双线性插值把参考动作重采样到仿真时间步上，保证帧与帧之间平滑
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontward"


# AMP 动作参考管理器：判别器据此区分参考动作与策略生成动作。
# 它在场景里生成一个幽灵机器人（RobotReference），按数据集播放动作，
# 训练时策略动作与参考动作的 link 位姿序列一起喂给判别器。
K1_MOTION_REFERENCE_CFG = MotionReferenceManagerCfg(
    # 机器人上作为参考根节点的 link（K1 用 Trunk，对应 G1 的 torso_link）。
    # 参考动作的根位姿会锚定到这个 link 上。
    prim_path="{ENV_REGEX_NS}/Robot/Trunk",
    # 机器人 URDF 路径：用于构建运动学链，把动作数据映射到 K1 的骨架
    robot_model_path=K1_CFG.spawn.asset_path,
    # 幽灵机器人所在 prim 路径（正则匹配所有 env 的 RobotReference）
    reference_prim_path="/World/envs/env_.*/RobotReference/Trunk",
    # 左右对称增广：一份参考动作可镜像出两份，扩大有效数据量
    symmetric_augmentation_link_mapping=K1_PARKOUR_LINK_SYMMETRY,
    symmetric_augmentation_joint_mapping=K1_SYMMETRY_JOINT_MAPPING,
    # 镜像时哪些关节需要取反（例如左髋 roll 镜像到右髋 roll，符号相反）
    symmetric_augmentation_joint_reverse_buf=K1_SYMMETRY_JOINT_SIGNS,
    # 参考动作帧间隔 0.02s（50 Hz），与策略控制频率一致
    frame_interval_s=0.02,
    # 参考动作时间索引的更新周期（秒）
    update_period=0.02,
    # 为判别器缓存最近 10 帧参考状态（判别器输入是一个短序列，不是单帧）
    num_frames=10,
    # 命名的动作数据集："run_walk" 键对应上述 K1AmassMotionCfg
    motion_buffers={"run_walk": K1AmassMotionCfg()},
    # 判别器比较哪些 link 的位姿（与 K1_PARKOUR_LINKS 一致）
    link_of_interests=K1_PARKOUR_LINKS,
    # 动作序列的切分播放方式："Even" = 均匀切分
    mp_split_method="Even",
)


class K1ParkourConfigMixin:
    """把 K1 形态学应用到 parkour 任务中机器人专属的接缝处。

    Isaac Lab 惯用的配置 Mixin 模式：任务主体逻辑（奖励函数、观测结构、
    地形、命令）复用 G1 版本，机器人相关的差异全部集中在这个 mixin 里。
    子类的 __post_init__ 先执行 G1 的初始化，再调用 apply_k1_config()
    逐个覆盖为 K1 的设置，相当于给继承来的配置打补丁，
    避免复制整个 parkour_env_cfg.py。
    """

    def apply_k1_config(self):
        # 1. 机器人资产与动作参考
        # cfg.replace(prim_path=...)：isaaclab 配置对象自带的方法，
        # 返回替换了指定字段的副本（类似 dataclasses.replace）。
        # {ENV_REGEX_NS} 在环境实例化时解析为每个 env 的命名空间
        # （/World/envs/env_0, env_1, ...）。
        self.scene.robot = K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.motion_reference = copy.deepcopy(K1_MOTION_REFERENCE_CFG)

        # G1 的躯干初始高度为 0.9 m，K1 只有 0.57 m。楼梯尺寸不能直接沿用
        # G1 的 23/45 cm 上限。深拷贝地形模板后，同时缩小上楼梯和下楼梯，
        # 确保 K1 训练配置不会污染共享的 G1 地形配置。
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator = copy.deepcopy(self.scene.terrain.terrain_generator)
            k1_terrain = self.scene.terrain.terrain_generator
            for terrain_name in ("pyramid_stairs", "pyramid_stairs_inv"):
                k1_terrain.sub_terrains[terrain_name].step_height_range = (0.04, 0.14)
            for terrain_name in ("pyramid_stairs_high", "pyramid_stairs_inv_high"):
                k1_terrain.sub_terrains[terrain_name].step_height_range = (0.05, 0.22)

        # 2. 动作空间
        # self.actions 是 ActionManagerCfg；joint_pos 是 G1 配置里定义的
        # JointPositionActionCfg（位置控制动作项）。
        # Isaac Lab 位置动作的语义：目标关节角 = 当前关节角 + scale x 动作值。
        #   joint_names=[".*"]：正则匹配所有关节（K1 的 22 个关节）
        #   scale：每个关节的缩放系数（K1_ACTION_SCALE 是 dict[str, float]，
        #     键为正则；在 assets/booster_k1.py 中按 0.25 x 力矩上限/刚度
        #     算出，表示网络输出 1.0 对应多大的角度增量）
        #   clip={".*": (-1.0, 1.0)}：动作值裁剪范围（策略输出通常已过
        #     tanh 天然落在 [-1,1]，此处是双保险）
        self.actions.joint_pos.joint_names = [".*"]
        self.actions.joint_pos.scale = K1_ACTION_SCALE
        self.actions.joint_pos.clip = {".*": (-1.0, 1.0)}

        # 3. 深度相机（模拟真机头部 RGB-D 相机）
        # self.scene.camera 是 G1 配置里定义的 NoisyGroupedRayCasterCameraCfg
        # （instinctlab 扩展的分组射线相机+噪声管线，继承 isaaclab 的
        # GroupedRayCasterCameraCfg）。相机是 scene 中注册的传感器之一，
        # 配置字段与 isaaclab 传感器规范一致：
        #   prim_path  = 挂载的 prim（相机随该 link 运动）
        #   offset     = 相机系相对该 prim 的位姿
        #   update_period = 传感器最小刷新间隔（秒）

        # 相机挂载点取自在产 K1 上实测的 ROS TF 变换
        # head_pitch_link -> head_color_optical_frame。
        # Head_2 是仿真资产中对应的 link；把相机挂在可动的头部上，
        # 可以保留头部运动对图像的真实物理影响。
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/Head_2"
        # 射线求交的目标网格：地面 + 机器人自身所有 link 的 /visuals 网格。
        # get_link_prim_targets() 把 link 名列表转成
        # "/World/envs/env_.*/Robot/<link>/visuals" 形式的 prim 表达式。
        # 把自己加进求交目标，机器人的手脚就会出现在深度图里（自遮挡），
        # 与真机相机看到的一致，这是 sim2real 深度训练的关键。
        self.scene.camera.mesh_prim_paths = ["/World/ground", *get_link_prim_targets(K1_LINK_NAMES)]
        # 相机相对 Head_2 的位姿（真机 TF 实测值）：
        # pos 单位米；rot 是四元数 (x, y, z, w)。
        self.scene.camera.offset.pos = (0.05663342989, 0.0462427773, 0.0962657193)
        self.scene.camera.offset.rot = (
            0.5132977331550982,
            -0.5083061254903114,
            0.4877471740434324,
            -0.49015611200872644,
        )
        # 位姿约定："ros" = 相机坐标系遵循 ROS 约定（z 朝前、x 朝右、
        # y 朝下，四元数 x,y,z,w）。G1 版本用 "world"（Isaac 世界系），
        # K1 因为数据来自 ROS TF 所以必须用 "ros" 解释。
        self.scene.camera.offset.convention = "ros"
        # 用真机标定内参换算仿真相机的视场角与主点偏移。
        # pattern_cfg 是 PinholeCameraPatternCfg（针孔相机射线模式）。
        # 焦距归一化为 1.0 时，aperture 就是单位焦距下焦平面上的孔径尺寸
        # （与 2 x tan(fov/2) 等价）。这里用像素尺寸/焦距(px) 得到与真机
        # 相同的水平/垂直视场角。
        self.scene.camera.pattern_cfg.horizontal_aperture = K1_DEPTH_WIDTH_PX / K1_DEPTH_FOCAL_LENGTH_PX
        self.scene.camera.pattern_cfg.vertical_aperture = K1_DEPTH_HEIGHT_PX / K1_DEPTH_FOCAL_LENGTH_PX
        # 主点偏移：真机主点不居中，有 (主点 - 图像中心)/焦距(px) 的偏移，
        # 在归一化焦平面上同样平移射线模式，模拟这种不对称。
        self.scene.camera.pattern_cfg.horizontal_aperture_offset = (
            K1_DEPTH_PRINCIPAL_X_PX - K1_DEPTH_WIDTH_PX / 2
        ) / K1_DEPTH_FOCAL_LENGTH_PX
        self.scene.camera.pattern_cfg.vertical_aperture_offset = (
            K1_DEPTH_PRINCIPAL_Y_PX - K1_DEPTH_HEIGHT_PX / 2
        ) / K1_DEPTH_FOCAL_LENGTH_PX
        self.scene.camera.update_period = K1_DEPTH_UPDATE_PERIOD_S
        # history_length 是 isaaclab SensorBaseCfg 字段：保存的历史帧数。
        # 其副作用同样重要：isaaclab 的 SensorBase.update() 中，只要
        # history_length > 0，每个物理步都会执行过期检查与缓冲刷新，
        # 而不是等有人读数据时才惰性刷新。否则 50 Hz 策略在 0.02s 间隔
        # 读数据，0.05s 的惰性刷新会实际落在 0.06s 上（相位漂移）。
        self.scene.camera.history_length = 1
        # 噪声管线 noise_pipeline 是 instinctlab NoisyCameraCfgMixin 的
        # 字段：dict[操作名, NoiseCfg]。Python 3.8+ 的 dict 保持插入顺序，
        # 噪声操作按顺序依次施加。先取出 G1 配置里已有的管线，
        # 改其中两项后整体放回。
        # 64x36 的射线图像是对完整传感器的降采样。取其下半部分中央区域，
        # 得到策略使用的 32x18、聚焦地形的观测图。
        noise_pipeline = self.scene.camera.noise_pipeline
        # crop_region 是 (上, 下, 左, 右) 四边各裁多少像素：
        # 上裁 18 行（去掉上半部天空），左右各裁 16 列（保留中央 32 列），
        # 64x36 变成 (64-32)x(36-18) = 32x18。
        noise_pipeline["crop_and_resize"].crop_region = (18, 0, 16, 16)
        self.scene.camera.noise_pipeline = {
            "crop_and_resize": noise_pipeline["crop_and_resize"],
            "gaussian_blur": noise_pipeline["gaussian_blur"],
            # RangeBasedGaussianNoiseCfg：在 [min_value, max_value] 深度
            # 区间内叠加标准差为 noise_std 的高斯噪声（模拟真实深度传感器
            # 的测距噪声）。对真机 20 帧静态采样显示：在策略使用的 2.5 m
            # 量程内，帧间差异中位数为 12 mm，p90 为 62 mm，据此设定噪声。
            "sensor_noise": RangeBasedGaussianNoiseCfg(
                min_value=0.1,
                max_value=2.5,
                noise_std=0.02,
            ),
            "depth_normalization": noise_pipeline["depth_normalization"],
        }
        # data_histories：对指定数据类型叠加历史帧缓冲，数据存放在
        # sensor.data["<data_type>_history"]，形状 (N, T, H, W)。
        # 键名带 _noised 后缀表示对加噪后的输出存历史。
        # 这里保存最近 16 帧加噪深度图供观测项读取。
        self.scene.camera.data_histories["distance_to_image_plane_noised"] = K1_DEPTH_HISTORY_LENGTH
        # self.observations 是 ObservationManagerCfg；policy/critic 是两个
        # ObservationGroupCfg（观测组，可分别配置）。depth_image 是组里的
        # 一个 ObsTerm（观测项：func + params + noise/scale 等）。
        # history_skip_frames 传给观测函数：从历史序列中每隔 2 帧取 1 帧
        # （images[:, ::2]），16 帧历史经降采样后输出 8 帧，
        # 覆盖更长时间窗口且不增加网络输入维度。
        for observation_group in (self.observations.policy, self.observations.critic):
            observation_group.depth_image.params["history_skip_frames"] = K1_DEPTH_HISTORY_SKIP_FRAMES

        # 4. 足部高度扫描仪
        # RayCasterCfg 传感器：从脚上方 z=20m 处向下打射线测地面高度，
        # 测量脚下地形有多高/多深（越障感知的关键信息）。
        # G1 版本挂在 ankle_roll link 上，K1 的脚结构不同，改挂到
        # *_foot_link（脚板）并微调前后偏移。
        self.scene.left_height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/left_foot_link"
        self.scene.left_height_scanner.offset.pos = (0.014, 0.0, 20.0)
        self.scene.right_height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/right_foot_link"
        self.scene.right_height_scanner.offset.pos = (0.014, 0.0, 20.0)

        # 5. 足部体积点云
        # VolumePointsCfg（instinctlab 扩展传感器）：在指定 prim 坐标系内
        # 生成一个 3D 网格采样点集，用于检测脚底下哪些位置有地形支撑
        # （奖励/观测会用到这些点与地面的关系）。
        # K1 按实际脚掌尺寸重设采样盒：长约 16cm、宽 7cm、高 2.4cm。
        self.scene.leg_volume_points.prim_path = "{ENV_REGEX_NS}/Robot/.*_foot_link"
        foot_points = self.scene.leg_volume_points.points_generator
        foot_points.x_min = -0.066
        foot_points.x_max = 0.094
        foot_points.y_min = -0.035
        foot_points.y_max = 0.035
        foot_points.z_min = -0.024
        foot_points.z_max = 0.0

        # 6. 奖励项：把 G1 的 body/joint 名称映射到 K1
        # self.rewards 是 RewardManagerCfg；rewards 字段是
        # dict[名字, RewardTermCfg]，每项的 params 是传给奖励函数 func 的
        # 参数字典。params 里常见的 sensor_cfg/asset_cfg 是 SceneEntityCfg，
        # 按名字解析场景实体：
        #   asset_cfg  -> 机器人资产（Articulation）上的 body/joint
        #   sensor_cfg -> scene 里注册的传感器（如接触传感器 contact_forces）
        # 解析时名字支持正则，所以改 body_names 只需换成 K1 的命名模式。
        rewards = self.rewards.rewards
        # feet_air_time：腾空时间奖励，接触传感器上的脚部 body
        for term_name in ("feet_air_time",):
            getattr(rewards, term_name).params["sensor_cfg"].body_names = ".*_foot_link"
        for term_name in ("feet_slide", "feet_flat_ori"):
            term = getattr(rewards, term_name)
            term.params["sensor_cfg"].body_names = ".*_foot_link"
            term.params["asset_cfg"].body_names = ".*_foot_link"
        rewards.feet_at_plane.params["contact_sensor_cfg"].body_names = ".*_foot_link"
        rewards.feet_at_plane.params["asset_cfg"].body_names = ".*_foot_link"
        rewards.feet_at_plane.params["height_offset"] = 0.024
        rewards.feet_close_xy.params["asset_cfg"].body_names = ".*_foot_link"
        rewards.joint_deviation_hip.params["asset_cfg"].joint_names = [
            ".*_Hip_Yaw",
            ".*_Hip_Roll",
        ]
        # 正则交替：(Hip|Knee|Ankle) 匹配三类关节名中的任一种
        for term_name in ("dof_torques_l2", "energy"):
            getattr(rewards, term_name).params["asset_cfg"].joint_names = [".*_(Hip|Knee|Ankle)_.*"]
        rewards.freeze_upper_body.params["asset_cfg"].joint_names = [
            ".*Head.*",
            ".*_Shoulder_.*",
            ".*_Elbow_.*",
        ]
        rewards.pelvis_orientation_l2.params["asset_cfg"].body_names = "Trunk"
        # 负向前瞻 (?!.*_foot_link)：匹配除脚板以外的所有 body，
        # 除脚之外任何部位接触地面都算 undesired contact
        rewards.undesired_contacts.params["sensor_cfg"].body_names = "(?!.*_foot_link).*"

        # 7. 终止条件
        # self.terminations 是 TerminationManagerCfg：
        # base_contact 项 = Trunk（躯干）触地即终止（摔倒了）
        self.terminations.base_contact.params["sensor_cfg"].body_names = "Trunk"
        # root_height 项 = Trunk 高度低于 0.32m 即终止（K1 躯干极限高度，
        # 比 G1 更矮，所以从 G1 的值改小）。该项可能为 None（未启用），
        # 先判空是防御性写法。
        if self.terminations.root_height is not None:
            self.terminations.root_height.params["minimum_height"] = 0.32


@configclass
class K1ParkourEnvCfg(G1ParkourRoughEnvCfg, K1ParkourConfigMixin):
    """K1 越野跑训练环境配置（训练用）。

    多继承的 MRO 决定初始化顺序：G1ParkourRoughEnvCfg 的 __post_init__
    先执行（搭建完整 G1 任务），再执行 apply_k1_config() 覆盖为 K1，
    这就是配置继承+补丁模式的完整闭环。
    """

    def __post_init__(self):
        super().__post_init__()
        self.apply_k1_config()


@configclass
class K1ParkourEnvCfg_PLAY(G1ParkourRoughEnvCfg_PLAY, K1ParkourConfigMixin):
    """K1 越野跑演示/回放环境配置（播放 checkpoint 用）。"""

    def __post_init__(self):
        super().__post_init__()
        self.apply_k1_config()

        # 检查点演示只需要一行有代表性的地形，而不是 G1 较大的调试场景。
        # 保留全部 10 列地形，同时去掉未使用的重复网格和调试可视化几何体，
        # 使视频回放更干净。
        # num_envs：并行环境数量，演示只开 1 个
        self.scene.num_envs = 1
        # TerrainGeneratorCfg：地形生成器，num_rows/num_cols 是地形格子的行列数
        if self.scene.terrain.terrain_generator is not None:
            # G1 Play 配置持有模块级地形模板；先深拷贝，避免 K1 演示参数
            # 污染 G1 回放或后续实例。
            self.scene.terrain.terrain_generator = copy.deepcopy(self.scene.terrain.terrain_generator)
            play_terrain = self.scene.terrain.terrain_generator
            play_terrain.num_rows = 1
            play_terrain.num_cols = 10

            # 演示环境使用训练范围的较低子集，便于直观检查当前 checkpoint。
            for terrain_name in ("pyramid_stairs", "pyramid_stairs_inv"):
                play_terrain.sub_terrains[terrain_name].step_height_range = (0.04, 0.10)
            for terrain_name in ("pyramid_stairs_high", "pyramid_stairs_inv_high"):
                play_terrain.sub_terrains[terrain_name].step_height_range = (0.05, 0.14)
        # debug_vis：传感器/命令的调试可视化开关，回放时关掉避免画面杂乱
        self.scene.leg_volume_points.debug_vis = False
        self.commands.base_velocity.debug_vis = False
