# Parkour Task

## Basic Usage Guidelines

### Parkour Task

**Task ID:** `Instinct-Parkour-Target-Amp-G1-v0`

1. Go to `config/g1/g1_parkour_target_amp_cfg.py` and set the `path` and `filtered_motion_selection_filepath` in `AmassMotionCfg` to the reference motion you want to use.

2. Train the policy:
```bash
python scripts/instinct_rl/train.py --headless --task=Instinct-Parkour-Target-Amp-G1-v0
```

3. Play trained policy (load_run must be provided, absolute path is recommended, or use `--no_resume` to visualize untrained policy):

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=<run_name>
```

4. Export trained policy (load_run must be provided, absolute path is recommended):

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=<run_name> --exportonnx --useonnx
```

## Common Options

- `--num_envs`: Number of parallel environments (default varies by task)
- `--keyboard_control`: Enable keyboard control during playing
- `--load_run`: Run name to load checkpoint from for playing
- `--video`: Record training/playback videos
- `--exportonnx`: Export the trained model to ONNX format for onboard deployment during playing
- `--useonnx`: Use the ONNX model for inference during playing (requires `--exportonnx`)

## Booster K1 Hiking in the Wild workflow

The K1 task keeps the official Parkour terrain, depth perception, virtual-edge and
foot-volume safety observations, AMP motion reference, and MoE policy. It replaces
the G1 robot-specific asset, joint/link mappings, actuator limits, and contact
geometry with the 22-DoF Booster K1 definitions.

**Task IDs:**

- Training: `Instinct-Parkour-Target-Amp-K1-v0`
- Evaluation: `Instinct-Parkour-Target-Amp-K1-Play-v0`

Retarget an SMPL-X motion with GMR, then convert the GMR pickle to the motion format
consumed by InstinctLab:

```bash
python /home/ducks/GMR/scripts/smplx_to_robot.py \
  --robot booster_k1 \
  --smplx_file <motion_stageii.npz> \
  --save_path <motion.pkl>

python scripts/gmr/convert_k1_motion.py \
  <motion.pkl> <dataset_dir>/<motion>.retargeted.npz
```

The repository includes seven curated AMASS-CMU trajectories retargeted
directly for K1 under `parkour_motion_reference/booster_k1_v2`: normal and
fast walking, running, a start-to-run transition, two stair ascents, and one
stair descent. Its `motions.yaml` loads the separate clips into one AMP
training set while preserving their reset boundaries. After copying the
complete project to a server, start training directly:

```bash
python scripts/instinct_rl/train.py \
  --headless --task=Instinct-Parkour-Target-Amp-K1-v0
```

For a custom dataset, create a motion-selection file such as
`<dataset_dir>/motions.yaml`:

```yaml
selected_files:
  - <motion>.retargeted.npz
```

Select the custom dataset and start training:

```bash
export INSTINCTLAB_K1_MOTION_DIR=<dataset_dir>
export INSTINCTLAB_K1_MOTION_SELECTION=<dataset_dir>/motions.yaml

python scripts/instinct_rl/train.py \
  --headless --task=Instinct-Parkour-Target-Amp-K1-v0
```

Evaluate a checkpoint with the K1 play environment:

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py \
  --task=Instinct-Parkour-Target-Amp-K1-Play-v0 \
  --load_run=<run_name>
```

The simulated depth camera uses a nominal K1 trunk mount. Measure the physical
camera transform and update `camera.offset` in
`config/k1/k1_parkour_target_amp_cfg.py` before real-robot deployment.
