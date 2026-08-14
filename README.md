# Project Instinct

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## 概述

本仓库是 [Project-Instinct](https://project-instinct.github.io/) 的环境（environment）侧实现。

我们的目标是将人形机器人（足式机器人）全身控制的强化学习（Reinforcement Learning）工业化。

**关键特性：**

- `Isolation`（隔离）：在 Isaac Lab 核心仓库之外工作，确保你的开发成果保持自包含。
- `Flexibility`（灵活）：本模板已配置为允许你的代码以扩展（extension）的形式在 Omniverse 中运行。
- `Unified Ecosystem`（统一生态）：本仓库是 Project-Instinct 生态的一部分，该生态还包括 [instinct_rl](https://github.com/project-instinct/instinct_rl) 和 [instinct_onboard](https://github.com/project-instinct/instinct_onboard) 仓库。
    - 该生态的核心设计是将每个实验视为一个独立的、结构化的文件夹，文件夹以时间戳作为唯一标识开头。
    - 给 `play.py` 脚本添加 `--exportonnx` 标志即可将策略导出为 ONNX 模型。之后，你应直接将 logdir 复制到机器人电脑上，并使用 `instinct_onboard` 工作流在真实机器人上运行该策略。

**关键词：** extension, template, isaaclab

## 警告

本代码库遵循 [CC BY-NC 4.0 license](LICENSE) 许可，并继承了 IsaacLab 中的许可。你不得将该材料用于商业目的，例如制作演示来宣传你的商业产品，或将代码包装用于你自己的商业目的。

## 贡献

请参阅我们的 [Contributor Agreement](CONTRIBUTOR_AGREEMENT.md) 了解贡献指南。通过贡献或提交 pull request，你同意将你的贡献的版权所有权转让给项目维护者。

请参阅 [CONTRIBUTORS.md](CONTRIBUTORS.md) 获取已确认贡献者的名单。

## 安装

- 按照 [安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab，并**切换到 5.1.0 版本**。我们推荐使用 conda 安装，因为它简化了从终端调用 Python 脚本的过程。我们使用的 IsaacLab commit 是 origin/main 上的 `f73c331738`（post-v2.3.2）。

- 按照 [安装指南](https://github.com/project-instinct/instinct_rl/blob/main/README.md) 安装 Instinct-RL。
    太长不看（TL; DR）：
    ```bash
    git clone https://github.com/project-instinct/instinct_rl.git
    python -m pip install -e instinct_rl
    ```

- 将本仓库与 Isaac Lab 安装目录分开克隆（即放在 `IsaacLab` 目录之外）：

    ```bash
    # Option 1: HTTPS
    git clone https://github.com/project-instinct/instinctlab.git

    # Option 2: SSH
    git clone git@github.com:project-instinct/instinctlab.git
    ```

- 使用已安装 Isaac Lab 的 Python 解释器安装该库：

    ```bash
    python -m pip install -e source/instinctlab
    ```

- 安装 [instinct-rl](https://github.com/project-instinct/instinct_rl) 后，要配合 `instinct-rl` 运行，可使用以下命令：

    ```bash
    python scripts/instinct_rl/train.py --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --headless
    ```

### Booster K1 locomotion

Booster K1 集成包含一个完整的 22 自由度（DoF）资产和一个 12 自由度（DoF）locomotion 资产，并带有已标定的电机、指令延迟和力矩-速度参数。使用以下命令训练或运行平地任务：

```bash
python scripts/instinct_rl/train.py --task=Instinct-Locomotion-Flat-K1-v0 --headless
python scripts/instinct_rl/play.py --task=Instinct-Locomotion-Flat-K1-Play-v0
```

K1 机器人的描述与网格（mesh）按照 BSD 3-Clause 许可在
`source/instinctlab/instinctlab/assets/resources/booster_k1/LICENSE` 中再分发。

## 关键组件文档

- [Instinct-RL 文档](https://github.com/project-instinct/instinct_rl/blob/main/README.md)
- [InstinctLab 文档](https://github.com/project-instinct/instinctlab/blob/main/DOCS.md)

### 设置 IDE（可选）

要设置 IDE，请按照以下说明操作：

- 运行 VSCode Tasks：按下 `Ctrl+Shift+P`，选择 `Tasks: Run Task`，然后在下拉菜单中运行 `setup_python_env`。运行该任务时，系统会提示你添加 Isaac Sim 安装目录的绝对路径。

如果一切执行正确，它会在 `.vscode` 目录中创建一个文件 `.python.env`。该文件包含 Isaac Sim 和 Omniverse 提供的所有扩展的 python 路径。这有助于在编写代码时对所有 python 模块建立索引，从而提供智能补全提示。

## 代码格式化

我们提供了一个 pre-commit 模板，用于自动格式化你的代码。
安装 pre-commit：

```bash
pip install pre-commit
```

然后你可以用以下命令运行 pre-commit：

```bash
pre-commit run --all-files
```

要让 `pre-commit` 在每次提交时自动运行，可在你的仓库中使用以下命令：

```bash
pre-commit install
```

## 训练你自己的项目

***为了保留你的代码开发成果和进度，请参考 https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/index.html 将你自己的项目创建为独立仓库。***

并将 `scripts/instinct_rl` 复制到你自己的仓库中。

### 或者你就是固执地想 fork 并直接修改本仓库中的代码。

- 请在 `source/instinctlab/instinctlab/tasks` 目录中新建一个文件夹。该文件夹的名称应为你的项目名称。在文件夹内部，务必（DO）在每一层子文件夹中添加 `__init__.py`。（很多人往往会忘记这一步，结果找不到本应注册的任务。）

- 我们继承 IsaacLab 中基于 manager 的 RL env 来添加新特性。务必（DO）在 `gym.register` 调用中使用 `instinctlab.envs:InstinctRlEnv` 作为 entry_point。例如，如果你想添加一个新任务，可以使用以下代码：

```python
import gymnasium as gym
from . import agents
task_entry = "instinctlab.tasks.shadowing.perceptive.config.g1"
gym.register(
    id="Instinct-Perceptive-Shadowing-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.perceptive_shadowing_cfg:G1PerceptiveShadowingEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
    },
)
```

## 故障排查

### Pylance 缺少对扩展的索引

在某些 VsCode 版本中，部分扩展的索引会缺失。此时，请在 `.vscode/settings.json` 的 `"python.analysis.extraPaths"` 键下添加你的扩展路径。

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/instinctlab"
    ]
}
```
