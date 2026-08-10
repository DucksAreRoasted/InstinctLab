from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass


@configclass
class HierarchicalContactSensorCfg(ContactSensorCfg):
    """Configuration for contact-reporting bodies nested below the sensor root."""

    class_type: type | str = "{DIR}.contact_sensor:HierarchicalContactSensor"
