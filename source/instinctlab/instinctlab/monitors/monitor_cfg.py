from isaaclab.managers import ManagerTermBaseCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils.configclass import configclass


@configclass
class MonitorSensorCfg(SensorBaseCfg):
    update_period: float = 0.005  # update every decimation

    prim_path: str = ""  # path to the sensor


@configclass
class MonitorTermCfg(ManagerTermBaseCfg):
    pass


@configclass
class TorqueMonitorSensorCfg(MonitorSensorCfg):
    """NOTE: Due to the update of joint_acc every decimation, it significantly decreases the performance (about 0.25x slower)."""

    class_type: type | str = "{DIR}.monitors:TorqueMonitorSensor"

    prim_path: str = "/World/envs/env_*/Robot"

    history_length: int = 4  # assuming is the number of decimation
