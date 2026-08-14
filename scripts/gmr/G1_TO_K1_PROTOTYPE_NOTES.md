# G1 to Booster K1 retargeting prototype

## Question

Can the released 29-DoF G1 Hiking motion be converted into a usable 22-DoF
Booster K1 motion by reconstructing semantic link targets with G1 forward
kinematics and solving K1 inverse kinematics?

## Verdict

Promising. Three independently segmented clips were finite, stayed inside K1
joint limits, respected the configured motor velocity limits, and loaded into
the official K1 Parkour motion buffer. The Isaac environment completed reset
and step with a 22-dimensional action and depth observations.

| Source frames | Duration | Median speed | Max velocity ratio | Result |
|---|---:|---:|---:|---|
| 1012:1133 | 2.42 s | 0.413 m/s | 0.520 | promising |
| 1133:1237 | 2.08 s | 0.452 m/s | 0.575 | promising |
| 1927:2104 | 3.54 s | 0.368 m/s | 0.870 | promising |

The original archive contains concatenation discontinuities. A loose split
allowed one G1 knee jump to produce a K1 knee speed 2.59 times its limit. A
0.25-radian single-frame joint threshold removed that invalid transition.

## Before production use

- infer foot contacts and anchor stance feet;
- solve K1 root height from foot contact instead of scaling G1 root height;
- reduce ankle-pitch saturation;
- add per-link position/orientation residuals and self-collision checks;
- replay references under K1 actuator dynamics before admitting them to AMP;
- replace the throwaway terminal shell with a tested batch converter.

## Prototype command

```bash
/home/ducks/miniconda3/envs/gmr/bin/python \
  scripts/gmr/prototype_g1_to_k1.py \
  --input <parkour_motion_without_run_retargetted.npz>
```

The prototype files are deliberately temporary. Promote the validated semantic
adapter into the production GMR converter, then delete the terminal shell.

The command accepts either the released ZIP directly or the extracted NPZ. To
list automatically detected clips and visualize one of them:

```bash
/home/ducks/miniconda3/envs/gmr/bin/python scripts/gmr/prototype_g1_to_k1.py \
  --input hiking-in-the-wild_Data\&Model.zip --list

/home/ducks/miniconda3/envs/gmr/bin/python scripts/gmr/prototype_g1_to_k1.py \
  --input hiking-in-the-wild_Data\&Model.zip --segment 2 --visualize
```

The MuJoCo window shows the K1 model with colored semantic target axes. Close
the window to stop playback; add `--show-labels` when target names are useful.
The released ZIP path, segment listing, automatic retargeting, and one-pass
viewer playback were exercised successfully on segment 2.

Segments 1--17 are airborne, takeoff, or landing references in the released
archive; segments 18--19 are grounded locomotion. The early recommendation to
use segment 2 as an ordinary walk was incorrect. The CLI now labels motion
kind, reports K1 foot clearance, warns before airborne playback, and defaults
to a short grounded locomotion segment.

The initial 0.25-radian joint-step split was also too aggressive: dynamic
takeoff/landing joints were mistaken for clip boundaries. Boundaries now
require a global root translation or orientation discontinuity. On the
released archive this reduces the visible continuous sequences from 26 to 19.
