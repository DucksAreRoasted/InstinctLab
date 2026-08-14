import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as Event
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from instinctlab.assets.booster_k1 import BOOSTER_K1_LOCOMOTION_CFG, K1_LOCOMOTION_ACTION_SCALE
from instinctlab.tasks.locomotion.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEventsCfg,
    G1FlatRewardsCfg,
    G1FlatSceneCfg,
    G1FlatTerminationsCfg,
)

_BASE_REWARDS = G1FlatRewardsCfg()
_BASE_TERMINATIONS = G1FlatTerminationsCfg()
_BASE_EVENTS = G1FlatEventsCfg()


@configclass
class K1FlatSceneCfg(G1FlatSceneCfg):
    robot = BOOSTER_K1_LOCOMOTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class K1FlatRewardsCfg(G1FlatRewardsCfg):
    feet_air_time = _BASE_REWARDS.feet_air_time.replace(
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "threshold": 0.4,
        }
    )
    feet_slide = _BASE_REWARDS.feet_slide.replace(
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        }
    )
    dof_pos_limits = _BASE_REWARDS.dof_pos_limits.replace(
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Pitch", ".*_Ankle_Roll"])}
    )
    joint_deviation_hip = _BASE_REWARDS.joint_deviation_hip.replace(
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw", ".*_Hip_Roll"])}
    )
    joint_deviation_arms: RewTerm | None = None
    joint_deviation_torso: RewTerm | None = None
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-1.0, params={"target_height": 0.57})
    joint_deviation_knee = _BASE_REWARDS.joint_deviation_knee.replace(
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Knee_Pitch"])}
    )
    dof_acc_l2 = _BASE_REWARDS.dof_acc_l2.replace(
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_Pitch"])}
    )
    dof_torques_l2 = _BASE_REWARDS.dof_torques_l2.replace(
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_Pitch"])}
    )


@configclass
class K1FlatTerminationsCfg(G1FlatTerminationsCfg):
    base_contact = DoneTerm(
        func=_BASE_TERMINATIONS.base_contact.func,
        time_out=False,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=r"^(?!left_foot_link$)(?!right_foot_link$).+$",
            ),
            "threshold": 10.0,
        },
    )


@configclass
class K1FlatEventsCfg(G1FlatEventsCfg):
    add_base_mass: Event | None = None
    randomize_body_mass = Event(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    base_external_force_torque = _BASE_EVENTS.base_external_force_torque.replace(
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        }
    )
    reset_base = _BASE_EVENTS.reset_base.replace(
        params={
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
    )
    reset_robot_joints = Event(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (0.0, 0.0)},
    )


@configclass
class K1FlatEnvCfg(G1FlatEnvCfg):
    scene: K1FlatSceneCfg = K1FlatSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: K1FlatRewardsCfg = K1FlatRewardsCfg()
    terminations: K1FlatTerminationsCfg = K1FlatTerminationsCfg()
    events: K1FlatEventsCfg = K1FlatEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = K1_LOCOMOTION_ACTION_SCALE
        self.actions.joint_pos.clip = {".*": (-1.0, 1.0)}
        self.run_name = self.run_name.replace("G1Flat", "K1Flat", 1)
        self.viewer.eye = (1.5, 1.5, 0.4)


@configclass
class K1FlatEnvCfg_PLAY(K1FlatEnvCfg):
    scene: K1FlatSceneCfg = K1FlatSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (2.0, 2.0)
        self.events.base_external_force_torque = None
        self.events.push_robot = None
