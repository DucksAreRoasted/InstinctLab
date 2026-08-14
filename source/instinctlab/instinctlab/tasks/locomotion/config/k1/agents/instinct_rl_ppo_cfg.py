from isaaclab.utils import configclass

from instinctlab.tasks.locomotion.config.g1.agents.instinct_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class K1FlatPPORunnerCfg(G1FlatPPORunnerCfg):
    max_iterations = 50000
    experiment_name = "k1_locomotion_flat"

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.8
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.entropy_coef = 0.0
