# Booster K1 parkour motion package

This directory contains the 19 trajectories recovered from
`parkour_motion_without_run_retargetted.npz` and retargeted from Unitree G1 to
the 22-DoF Booster K1 at 50 Hz:

- `stairs_descent_001` through `stairs_descent_009`
- `stairs_ascent_010` through `stairs_ascent_017`
- `grounded_locomotion_018` and `grounded_locomotion_019`

`motions.yaml` presents all 19 trajectories to the AMP motion buffer as one
training set. The files intentionally remain separate: concatenating their
arrays would create non-physical position and velocity jumps at clip
boundaries, while the motion buffer already samples across files and resets at
the correct trajectory boundary.

The K1 parkour task resolves this directory relative to the project root, so a
complete project copy can train without dataset environment variables:

```bash
python scripts/instinct_rl/train.py \
  --task=Instinct-Parkour-Target-Amp-K1-v0 \
  --headless
```

`INSTINCTLAB_K1_MOTION_DIR` and `INSTINCTLAB_K1_MOTION_SELECTION` may still be
set to override the packaged dataset. Run `sha256sum -c SHA256SUMS` from this
directory after copying the project to verify the binary motion files.
