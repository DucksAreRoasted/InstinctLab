# Booster K1 locomotion and stairs motion pack v2

This package replaces the short G1-derived hiking fragments with motions
retargeted directly from AMASS-CMU SMPL-X data to the 22-DoF Booster K1.

| Output | CMU source | Purpose | Source interval |
| --- | --- | --- | --- |
| `walk_normal_35_01` | 35_01, Walk | normal walk | 0.10--2.90 s; played at 24 Hz |
| `walk_fast_82_12` | 82_12, happy or fast walk forward | fast walk | 0.20--3.25 s |
| `run_steady_111_24` | 111_24, Run | steady run gait | 0.35--6.70 s |
| `start_to_run_143_03` | 143_03, Start to Run | transition | 1.50--3.90 s; played at 22.5 Hz |
| `stairs_ascent_114_07` | 114_07, Walking up and down stairs | ascent | 2.70--7.30 s |
| `stairs_descent_114_07` | 114_07, Walking up and down stairs | descent | 7.30--11.80 s |
| `stairs_ascent_83_31` | 83_31, walk forward stepping up stairs | second ascent style | 1.30--7.50 s |

All other clips retain GMR's approximately 30 Hz output rate. The normal walk
and transition are deliberately slowed to match the K1 parkour command range,
which is centered around 0.45--1.0 m/s.

The source AMASS files remain in the user's separately licensed dataset. AMASS
permits non-commercial research use and prohibits redistribution; do not
publish this generated binary directory without checking the dataset license.
The underlying CMU motion descriptions are indexed at
<https://mocap.cs.cmu.edu/>.

Build after producing the six GMR pickle files:

```bash
python scripts/gmr/build_k1_locomotion_pack.py \
  --gmr-dir /tmp/k1_motion_v2_gmr \
  --output-dir parkour_motion_reference/booster_k1_v2
```

Run `sha256sum -c SHA256SUMS` in this directory to verify the generated files.

Preview one generated trajectory with the GMR environment:

```bash
/home/ducks/miniconda3/envs/gmr/bin/python scripts/gmr/play_k1_motion.py \
  parkour_motion_reference/booster_k1_v2/stairs_ascent_114_07.retargeted.npz \
  --loop --follow-camera
```
