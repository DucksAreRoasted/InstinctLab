# Migrating an Isaac Lab 2.3.2 project to Isaac Lab 3.0.0-beta2 with PhysX

Updated: 2026-08-12

## Purpose and scope

This document is a practical guide for agents and downstream projects migrating from Isaac Lab
2.3.2 / Isaac Sim 5.1 to the pinned Isaac Lab 3.0.0-beta2 PhysX stack used by InstinctLab. It is
based on the completed InstinctLab migration, including numerical comparisons, sensor stress tests,
training and play canaries, video inspection, export checks, and rollback drills.

The target in this guide is deliberately exact:

| Component | Verified target |
|---|---|
| Python | `3.12.13` |
| Isaac Sim | `6.0.1` |
| Isaac Lab branch | `release/3.0.0-beta2` |
| Isaac Lab commit | `6a7acb0320a0bdc15b13e44e83b575e00797faf4` |
| `isaaclab` package | `6.1.17` |
| `isaaclab-assets` | `0.3.5` |
| `isaaclab-physx` | `1.1.3` |
| `isaaclab-tasks` | `1.10.9` |
| `isaaclab-visualizers` | `0.1.0` |
| PyTorch family | Torch `2.10.0`, TorchAudio `2.10.0`, TorchVision `0.25.0` |
| Warp | `1.13.0` |
| Gymnasium | `1.2.1` |
| NumPy | `>=2` |

Do not replace the full commit with a floating branch name. The maintained beta2 branch contains
post-tag fixes, so a different branch head or the pristine beta2 tag is a different migration
target and must be assessed again.

This guide covers the **PhysX milestone only**. Isaac Lab 3.0 is multi-backend, but Newton has
different solver behavior and requires separate task, sensor, asset, and learning validation. A
successful PhysX migration is a baseline for later Newton work, not proof of Newton compatibility.

Projects using deformables, surface grippers, XR/teleoperation, or a custom Kit experience must
also follow the corresponding feature section in the pinned upstream migration guide. In
particular, surface-gripper implementations moved to `isaaclab_physx`, deformable schemas changed,
and the old OpenXR device stack is superseded by Isaac Teleop. Treat each enabled feature as an
additional migration and verification surface; do not assume the core task gates cover it.

## The migration in one page

The upgrade is not primarily an import rename. Five changes interact:

1. The runtime changes to Python 3.12 and Isaac Sim 6.0.1.
2. Isaac Lab separates backend-neutral APIs from `isaaclab_physx` implementations.
3. in-memory quaternions change globally from WXYZ to XYZW.
4. asset and sensor data becomes Warp-backed `ProxyArray` data.
5. the URDF importer and simulator lifecycle change, affecting USD hierarchy, views, sensors,
   launch ordering, rendering, and generated assets.

Use this order:

| Phase | Objective | Gate before continuing |
|---|---|---|
| 0 | Freeze the 2.3.2 reference | Reproducible baseline and exact repository/runtime manifest |
| 1 | Build a separate 3.0 environment | Fresh-process imports at pinned versions |
| 2 | Port imports, configs, and launch lifecycle | Every config resolves without constructing a simulation |
| 3 | Convert quaternion semantics | Focused math/file-boundary tests pass |
| 4 | Port `ProxyArray` and write APIs | Torch/Warp ownership and partial writes are verified |
| 5 | Port custom sensors and PhysX views | Partial-update, reset/recreate, and numerical sensor gates pass |
| 6 | Rebuild and validate assets | Names, hierarchy, physics properties, and collision geometry pass |
| 7 | Integrate tasks and RL | Reset, rollout, train, resume, video, and export pass |
| 8 | Compare and release | Numerical, learning, performance, and rollback evidence is accepted |

Commit coherent phases separately. Do not combine quaternion, asset, sensor, and launcher changes
into one unreviewable patch.

## Rules that prevent false success

- Keep the 2.3.2 checkout and environment read-only. Create a new environment and worktree for 3.0.
- Record the commit and dirty state of every editable repository. A branch name is not provenance.
- Preserve user-owned paths, datasets, assets, logs, and checkpoints. Never silently repair a
  missing dataset path.
- Do not rely on deprecated compatibility aliases merely because imports succeed. Port to the
  target-native API and audit remaining aliases explicitly.
- Do not run broad search-and-replace over quaternion literals or `.data.*` expressions. Both need
  semantic, owner-aware review.
- Do not require bitwise physics equality across Isaac Sim generations. Define tolerances before
  examining target results and document discontinuous threshold behavior separately.
- Run simulator verification on the real Linux/CUDA host. Static documentation or source checks on
  another platform do not replace runtime tests.
- Treat registration, construction, reset, rollout, training, video, and export as separate gates.
  Passing one does not imply the others.

## Phase 0: freeze a reproducible 2.3.2 baseline

Before changing code, capture each repository independently:

```bash
git status --short --branch
git rev-parse HEAD
git diff --stat
git diff --name-status
```

Record at least:

- Python, Isaac Sim, Isaac Lab, CUDA, driver, Torch, Warp, NumPy, and Gymnasium versions;
- GPU model and `nvidia-smi` output;
- full SHAs and worktree status for the project, Isaac Lab, and the RL library;
- the resolved environment and agent configs;
- dataset and generated-asset paths plus checksums where practical;
- task IDs, observation/action spaces, manager term names and order;
- seeds, environment count, simulator timestep, decimation, and commands;
- reset state and a fixed action sequence;
- observations, rewards, termination flags, contact data, and sensor outputs;
- environment creation time, steady-state throughput, host memory, and device memory.

Use versioned output directories outside the read-only baseline checkout. A good baseline contains
both a small dataset-free task and the most complex dataset/sensor task the project supports.

Define comparison tolerances now. At minimum, include root pose, quaternion angle error, joint
position and velocity, reward error, contact rate/force, ray hits or depth, and termination timing.

## Phase 1: create and pin the target runtime

Create a new checkout at the exact target commit instead of merging the target into the old Isaac
Lab branch:

```bash
git fetch origin release/3.0.0-beta2
git switch --detach 6a7acb0320a0bdc15b13e44e83b575e00797faf4
git rev-parse HEAD
```

Create a separate Python 3.12 environment and install Isaac Sim 6.0.1 and Isaac Lab by following
the installation procedure in that checkout. Install the downstream project and RL library as
editable packages only after their revisions and dependency metadata have been selected.

For a PhysX project, declare both Python and extension dependencies. A downstream package based on
the verified InstinctLab runtime used requirements equivalent to:

```python
INSTALL_REQUIRES = [
    "isaaclab==6.1.17",
    "isaaclab-assets==0.3.5",
    "isaaclab-physx==1.1.3",
    "isaaclab-tasks==1.10.9",
    "isaaclab-visualizers==0.1.0",
    "numpy>=2",
    "torch==2.10.0",
    "torchaudio==2.10.0",
    "torchvision==0.25.0",
    "gymnasium==1.2.1",
    "warp-lang==1.13.0",
]
```

Adapt optional dependencies to the downstream project, but do not loosen conflicting pins merely
to make installation complete. Solve the runtime as one set and save the resolved package list.

For an Omniverse extension, add the PhysX backend dependency:

```toml
[dependencies]
"isaaclab" = {}
"isaaclab_physx" = {}
```

`isaaclab_visualizers` is a Python package, not an Omniverse extension dependency. Do not add it to
`extension.toml` unless the pinned target actually provides extension metadata for it.

Update project metadata to Python 3.12, including `python_requires`, Pyright, isort, Docker images,
CI, and IDE configuration.

The phase gate is a fresh process that prints the expected revisions and imports the core package,
PhysX backend, project package, RL package, and task-registration package without missing-extension
or dependency errors.

## Phase 2: port imports, configuration, and lifecycle

### Backend-neutral versus PhysX-specific imports

Keep factory-backed public assets and sensors on the `isaaclab` surface:

```python
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.sensors import ContactSensor, FrameTransformer, RayCaster
```

Import from `isaaclab_physx` only when selecting PhysX configuration, using the PhysX manager, or
subclassing a concrete PhysX implementation:

```python
from isaaclab_physx.physics import PhysxCfg, PhysxManager
from isaaclab_physx.sensors.ray_caster import MultiMeshRayCaster
```

Direct concrete imports intentionally make that custom class PhysX-only. Do not describe such a
class as backend-neutral.

### Common import changes

| 2.3.2 usage | 3.0.0-beta2 PhysX usage |
|---|---|
| `omni.physics.tensors.impl.api` | `omni.physics.tensors.api` |
| `isaacsim.core.simulation_manager.SimulationManager` | `isaaclab_physx.physics.PhysxManager` |
| `XformPrimView` | `isaaclab.sim.views.FrameView` |
| `root_physx_view` | `root_view` or public asset data |
| direct `isaacsim.core.utils.*` | matching `isaaclab.sim.utils.*` when available |
| old configclass locations | `isaaclab.utils.configclass` |

Prefer Isaac Lab utilities over direct Isaac Sim APIs. If an Isaac Sim experimental API remains
necessary, declare or enable its extension explicitly instead of relying on import order.

### Explicit PhysX configuration

`SimulationCfg.physx` is gone. Assign a `PhysxCfg` to `SimulationCfg.physics`:

```python
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg

sim = SimulationCfg(
    dt=1.0 / 120.0,
    physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
)
```

Convert code such as `self.sim.physx.gpu_max_rigid_patch_count = ...` to either construction of a
complete `PhysxCfg` or assignment through `self.sim.physics`. Do not attach undeclared attributes.

### Schema configuration split

Isaac Lab 3.0 splits common schema fields from backend-specific fields. Use common bases only when
all configured fields are truly backend-neutral; otherwise select the PhysX class explicitly.

| Legacy alias | Common base | PhysX configuration |
|---|---|---|
| `RigidBodyPropertiesCfg` | `RigidBodyBaseCfg` | `PhysxRigidBodyPropertiesCfg` |
| `JointDrivePropertiesCfg` | `JointDriveBaseCfg` | `PhysxJointDrivePropertiesCfg` |
| `CollisionPropertiesCfg` | `CollisionBaseCfg` | `PhysxCollisionPropertiesCfg` |
| `ArticulationRootPropertiesCfg` | `ArticulationRootBaseCfg` | `PhysxArticulationRootPropertiesCfg` |
| `RigidBodyMaterialCfg` | `RigidBodyMaterialBaseCfg` | `PhysxRigidBodyMaterialCfg` |

Also migrate joint-drive fields `max_velocity` to `max_joint_velocity` and `max_effort` to
`max_force`. Compatibility aliases exist in the pinned target, but leaving them hides unfinished
work.

### Simulator lifecycle

Custom train and play scripts should use the target lifecycle:

```python
from isaaclab_tasks.utils import add_launcher_args, launch_simulation

add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

with launch_simulation(env_cfg, args_cli):
    # Import simulator-dependent modules here.
    env = gym.make(task_id, cfg=env_cfg)
    try:
        ...
    finally:
        env.close()
```

Do not recreate the old direct `AppLauncher` / global `simulation_app` ownership model. Resolve
pure configuration, checkpoint paths, and CLI values before launch; import USD, PhysX, simulator
implementations, and vector wrappers only after the context starts when possible.

Task registration and config inspection should not initialize Kit. For namespaces that re-export
many implementations, use the pinned target's `.pyi` plus `lazy_export()` pattern. Test this with a
fresh process and verify that registration/config import loads no `pxr`, `omni`, or `carb` modules
before `launch_simulation`.

## Phase 3: migrate quaternion semantics as a data-schema change

Isaac Lab 3.0 uses XYZW throughout memory:

| Meaning | 2.3.2 | 3.0.0-beta2 |
|---|---|---|
| Order | `(w, x, y, z)` | `(x, y, z, w)` |
| Identity | `(1, 0, 0, 0)` | `(0, 0, 0, 1)` |

This is the highest silent-correctness risk: code often continues running with wrong orientations.

Audit all of the following:

- asset and sensor offsets;
- initial and reset poses;
- goals, commands, and marker poses;
- direct physics-view reads;
- quaternion buffers and hard-coded identities;
- custom Torch and Warp kernels;
- observations, rewards, and headings;
- interpolation and symmetry augmentation;
- datasets, caches, checkpoints, ONNX inputs, and exported metadata.

The suffix `_w` means **world frame**, not WXYZ. Do not rename fields solely because they end in
`_w`.

Run the pinned Isaac Lab quaternion finder in report mode first:

```bash
python /path/to/IsaacLab/scripts/tools/find_quaternions.py \
  --path source --base <pre-migration-sha>
python /path/to/IsaacLab/scripts/tools/find_quaternions.py \
  --path scripts --base <pre-migration-sha>
```

Use any automatic identity fix interactively and review every nonidentity four-vector manually.
RGBA values, plane equations, and unrelated four-vectors are easy false positives.

Adopt one internal rule:

> Every in-memory quaternion is XYZW. A non-XYZW format may exist only at a named, documented file
> or protocol boundary and is converted exactly once.

For example, if an existing dataset contract stores WXYZ:

```python
root_quat_wxyz = motion_file["root_quat"]
root_quat_xyzw = math_utils.convert_quat(root_quat_wxyz, to="xyzw")
```

`convert_quat` exists in this pinned target and is appropriate at proven serialization boundaries.
Do not use it internally to compensate for mixed conventions. Give raw variables convention-bearing
names and attach convention metadata to newly generated formats.

If policy observations contain quaternion components, old checkpoints may be schema-incompatible
even when their tensor shape is unchanged. Retrain unless there is an explicitly designed and
verified compatibility policy; do not silently reorder checkpoint inputs.

After the source audit, run representative tasks with:

```bash
WARN_ON_TORCH_QUATF_ACCESS=1 python <launcher> ...
```

Inspect every warning produced by a `.torch` read of a Warp quaternion property. This detector does
not cover direct Warp-side access or arbitrary project tensors, so it complements rather than
replaces the source and file-boundary audit.

## Phase 4: migrate `ProxyArray` and write semantics

### Treat every `data.*` access according to its owner

Isaac Lab asset and sensor properties now commonly return `ProxyArray`, which exposes explicit
Torch and Warp views:

```python
joint_pos_torch = robot.data.joint_pos.torch
joint_pos_warp = robot.data.joint_pos.warp
```

Do not append `.torch` globally. A downstream project's own `data` object may already hold Torch
tensors.

Use these rules:

- use `.torch` for indexing, slicing, cloning, concatenation, `torch.*`, and third-party Torch code;
- pass a `ProxyArray` directly to `wp.launch()` when its CUDA-array interface is sufficient;
- use `.warp` only when a concrete `warp.array`, pointer, stride, or exact Warp type is required;
- replace `wp.to_torch(proxy_array)` compatibility calls with `proxy_array.torch`;
- do not depend on the temporary `__torch_function__` compatibility bridge;
- never rebind immutable simulator-owned `ProxyArray` fields; write into their backing buffers or
  maintain a separate project-owned output buffer.

Keep `TimestampedBuffer` for genuinely Torch-owned project data. Use `TimestampedBufferWarp` for a
custom asset or sensor following the simulator's Warp-backed data contract.

### Select `_index` or `_mask` writes explicitly

Unsuffixed asset write methods were split:

- `_index` accepts compact partial data corresponding to selected environment/body IDs;
- `_mask` accepts full-size data and boolean Warp masks.

Example:

```python
env_ids = torch.tensor([1, 4], device=device)
partial_pose = desired_pose[env_ids]  # two rows
robot.write_root_pose_to_sim_index(partial_pose, env_ids=env_ids)
```

Do not pass a full `num_envs` buffer to an `_index` call or compact rows to a `_mask` call. Add
focused tests that prove unselected environments remain unchanged.

For `RigidObjectCollection`, migrate `object_*` names to `body_*` first, then choose `_index` or
`_mask`. Examples include `find_bodies`, `body_pose_w`, and
`write_body_pose_to_sim_index`.

### Inventory other changed sensor APIs

Projects using these APIs need an explicit decision:

- the 2.3.2 full-state `Imu` is named `Pva` in 3.0; the new `Imu` exposes only angular velocity and
  linear acceleration;
- `ArticulationData.body_incoming_joint_wrench_b` is removed; add a `JointWrenchSensor`;
- contact-sensor `pose_w`, `pos_w`, and `quat_w` are deprecated; use a `FrameTransformer` or another
  dedicated pose sensor;
- deformable-body schemas and materials changed substantially and require their own asset and
  physics comparison.

Do not let the unchanged import name `Imu` hide the semantic change. Compare the actual fields used
by observations and rewards.

## Phase 5: port custom sensors and low-level PhysX code

Custom sensors are where import-compatible migrations most often fail at runtime.

### PhysX tensor views

Update the module path and manager:

```python
import omni.physics.tensors.api as physx
from isaaclab_physx.physics import PhysxManager

physics_sim_view = PhysxManager.get_physics_sim_view()
```

Prefer public asset/sensor data over a raw view when it provides the needed value. If a raw view is
unavoidable, isolate it in one small project-local component responsible for path resolution,
device semantics, indexing, shape, and XYZW handling.

### Contact sensors and imported hierarchy

Importer 3.0 may produce nested rigid links. Any project code assuming that all rigid bodies are
direct children of an articulation root must be rewritten or removed.

Check:

- how contact-report APIs are applied to every intended rigid body;
- whether prim expressions match complete descendant paths;
- environment-major versus body-major ordering returned by a view;
- body-name ordering and filtered-contact matrices;
- contact thresholds near termination boundaries.

The pinned beta2 PhysX contact sensor had a descendant-path issue for InstinctLab's imported G1.
InstinctLab used a narrow hierarchical contact-sensor workaround that constructs complete paths and
validates the resulting view order. Do not copy that workaround blindly: first reproduce the issue
with the downstream asset and remove the workaround when the pinned upstream behavior is sufficient.

### Ray casters and ray cameras

The ray pipeline is Warp-native. Review all custom ray subclasses together:

- subclass the correct base or concrete PhysX implementation;
- use a child sensor prim and authored offset as required by the target;
- replace `attach_yaw_only` with `ray_alignment`;
- account for device-qualified mesh-cache keys such as `(prim_path, device)`;
- add the environment mask to direct kernel calls that require it;
- use Warp-owned ray, hit, depth, normal, and camera buffers correctly;
- verify mesh transforms for static and dynamic targets;
- validate partial updates and debug visualization.

For grouped dynamic meshes, never infer a ray's environment from a flattened launch row unless that
mapping is proven. A robust design stores an explicit world ID per ray and a checked, precomputed
world-to-entity membership table. Store a shared static mesh once and keep dynamic per-world bodies
as distinct entities. Validate every index before upload and range-check it in custom kernels.

### Custom camera output

Camera data is also `ProxyArray`-backed. Noise and history code should cross into Torch explicitly,
while native camera output stays in the target's ownership model. Preserve intended dimensions when
creating zero-copy views; image history often requires a channel-vector Warp dtype to retain
`(N, T, H, W, C)` rather than flattening a dimension.

### Sensor lifecycle gate

For every custom sensor, test:

1. full update;
2. selected-environment update;
3. reset of selected environments;
4. repeated create/step/close cycles;
5. recreation after close;
6. debug visualization;
7. cache isolation on every available CUDA device;
8. stable host and device memory after warm-up.

An empty outdated mask must be cheap and side-effect free. InstinctLab found a large apparent
performance regression caused by custom interpolation work running even when Isaac Lab dispatched
an empty sensor update. Return early without changing timestamps or buffers when no environment is
outdated, while preserving the base sensor's contract for real updates.

## Phase 6: rebuild and validate converted assets

The Isaac Sim 6 URDF and MJCF importers are rewrites. Do not reuse old generated USDs without
proving they are compatible, and do not assume a successful conversion preserved physics semantics.

Important changes include:

- structured output such as `{usd_dir}/{robot_name}/{robot_name}.usda`;
- nested rigid-body hierarchy;
- removal of the old `usd_file_name` workflow;
- `replace_cylinders_with_capsules` becoming a deprecated no-op;
- changed assumptions around instanceability and editable layers.

MJCF projects must separately audit removed converter options (`fix_base`, `link_density`,
`import_inertia_tensor`, and `import_sites`), the nested rigid-body hierarchy, and the source model's
`<freejoint>` semantics. Use the pinned upstream migration guide for the exact replacement fields.

Regenerate into a staging directory and compare against the 2.3.2 baseline:

- articulation root and fixed-base behavior;
- joint and body names, counts, and order;
- limits, axes, inertias, masses, and center-of-mass transforms;
- drive gains, effort/velocity limits, armature, and friction;
- collision shapes, materials, filters, self-collision, and contact reporting;
- default pose and a fixed drop or fixed-action rollout;
- prim paths used by custom sensors, markers, and mesh traversal.

Never let raw USD traversal order define policy semantics. Resolve policy-facing joint and body
indices by name and record the canonical order in the task schema. Importer ordering may change even
when the robot is otherwise correct.

If the old importer supplied a required behavior that the new importer dropped, compose the new
standard importer with a narrow offline postprocessor. Do not fork the importer and do not mutate
instantiated runtime prims.

InstinctLab's G1 collision cylinders illustrate the pattern:

1. inventory the exact 2.3.2 generated capsules;
2. keep a manifest of expected prim paths and geometry;
3. run the standard 3.0 conversion in disposable staging;
4. edit the defining USD layer offline;
5. preserve schemas, transforms, physics attributes, materials, and extents;
6. fail on path/count mismatch;
7. hash all inputs, tool revisions, config, and postprocessor version;
8. publish an immutable digest-scoped final asset;
9. load only that final artifact at runtime and record its digest in experiment metadata.

Do not guess capsule geometry. `UsdGeom.Capsule.height` is the cylindrical spine length, excluding
the hemispherical caps. Capture the old generated geometry before selecting a conversion rule.

## Phase 7: port viewer, video, task registration, and RL integration

### Viewer, visualizer, and renderer are different owners

- `ManagerBasedEnvCfg.viewer: ViewerCfg` owns camera pose, origin type, and asset tracking.
- `SimulationCfg.visualizer_cfgs` selects visualization backends.
- the renderer produces camera frames.
- Gymnasium `RecordVideo` owns trigger, length, encoding, naming, and output directory.

Keep task-relative camera tracking on `ViewerCfg` and the inherited `ViewportCameraController`.
Do not replace `origin_type="asset_root"` with an absolute `set_camera_view()` call. That can create
a valid but uniformly gray video because the task is outside the camera view.

`SimulationCfg.default_visualizer_cfg` is not a field in this pinned target. Do not add it.

Replace live 3.0 uses of `--headless` with:

```bash
--viz none  # no viewer; suitable for training and offscreen video
--viz kit   # interactive Kit window
```

The pinned beta2 launcher may leave the Kit GUI gate unset in some custom launchers. If the window
does not appear, verify the upstream launcher behavior before using the narrow Kit-only argument
`--/isaaclab/has_gui=true`. Do not apply it to Newton or treat it as a physics setting.

For video, ordering matters:

```python
if args_cli.video:
    args_cli.enable_cameras = True  # before launch_simulation

with launch_simulation(env_cfg, args_cli):
    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
```

Validate pixels, not just the MP4 container: inspect representative frames for the robot, terrain,
reference geometry, and debug markers.

### Task and manager behavior

For every registered task, compare:

- observation and action names, order, shapes, dtypes, and Gym spaces;
- reward, termination, event, curriculum, and command term order;
- history-buffer reset behavior;
- timeout and dataset-exhaustion semantics;
- object and reference synchronization;
- `extras` and logging keys.

If a dataset silently wraps without ending the episode, explicitly reset stateful observation terms
and history buffers for those environments. Otherwise, the new motion segment inherits history from
the preceding segment.

### RL, checkpoints, distributed execution, and export

Keep a custom RL runner when the library is not one of Isaac Lab's unified backends, but place it
inside `with launch_simulation(...)`. Preserve vector wrappers, checkpoint lookup, logging, resume,
and environment close semantics.

Log the commit, status, and diff for every editable source repository involved in a run, not only
the file containing the launcher.

Test resume as behavior, not merely loading: verify that the next checkpoint number advances and
that model parameters change. Treat `max_iterations` according to the RL library's documented
meaning after resume.

With Torch 2.10, export ONNX explicitly at opset 18. Otherwise, a requested older opset may emit a
failed downgrade traceback even though a valid opset-18 file remains. Run `onnx.checker`, test all
policy variants, include external weights/normalizers, and compare at least one exported policy's
multi-step outputs with PyTorch.

For distributed training, a world-size-one `torchrun` verifies initialization and code structure but
does not prove multi-GPU correctness. Run true multi-GPU when hardware is available.

## Phase 8: verification and release

Use a ladder so failures remain local:

1. source compiles;
2. package metadata resolves;
3. task registration imports without simulator modules;
4. every config instantiates without launching simulation;
5. asset conversion and hierarchy audits pass;
6. each task constructs and resets;
7. four finite zero/fixed-action steps pass;
8. partial update/reset leaves unselected environments unchanged;
9. longer reset and sensor lifecycle stress passes;
10. one optimizer update per task family passes with finite losses;
11. resume, play, video, and export pass;
12. distributed initialization/update/save passes;
13. controlled 2.3.2-versus-3.0 numerical comparisons pass;
14. steady-state performance and memory comparisons pass;
15. rollback to the read-only 2.3.2 stack succeeds from a clean process.

Use exact configured datasets. If a path is missing, stop and obtain the correct path rather than
substituting another dataset and claiming the task passed.

For physics comparisons, align joints and bodies by name. Use aggregate error metrics and declare
tolerances. Contact thresholds can create discontinuous termination differences even when continuous
state/reward trajectories are close; report those explicitly.

Warm up performance tests and measure long enough to avoid startup and synchronization noise. A
short InstinctLab Perceptive microbenchmark once reported a `13.30%` regression, while the later
matched 3,000-timestep end-to-end Play smoke measured a `0.05%` improvement. Retain short profiles
for diagnosis, but make release decisions from the representative end-to-end workload.

## Static migration audit

Run these searches after implementation and classify every result:

```bash
rg -n 'omni\.physics\.tensors\.impl\.api|obtain_world_pose_from_view|XformPrimView' source scripts
rg -n 'isaacsim\.core\.simulation_manager|root_physx_view' source scripts
rg -n 'SimulationCfg\.physx|\.sim\.physx\.' source scripts
rg -n '\.write_[A-Za-z_]+_to_sim\(' source scripts
rg -n 'convert_quat|quat_|rot=' source scripts
rg -n '\.data\.[A-Za-z_][A-Za-z_0-9]*' source scripts
rg -n 'default_visualizer_cfg|ViewerCfg|\.viewer\.|visualizer_cfgs' source scripts
rg -n 'RecordVideo|render_mode=.rgb_array.|enable_cameras' source scripts
rg -n -- '--headless' README.md source scripts
```

Interpretation matters:

- write-method matches are valid only when the chosen `_index`/`_mask` semantics and shapes are
  verified;
- `convert_quat` should remain only at proven non-XYZW file/protocol boundaries;
- `.data.*` matches include both Isaac Lab `ProxyArray` data and project-owned Torch data;
- `ViewerCfg` and `visualizer_cfgs` are not zero-match targets; audit their ownership;
- the 2.3.2 rollback command may remain the sole documented `--headless` exception.

Also run project formatters, type checks, and `pre-commit run --all-files` when configured. If hooks
would modify preserved user-owned changes, run them in a disposable copy, inspect the output, and
apply only migration-owned edits.

## Common failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Task runs but orientations are wrong | WXYZ literal/data crossed into XYZW memory | identities, camera offsets, file-boundary conversion |
| `torch.*` warning or type failure | implicit `ProxyArray` compatibility path | add `.torch` at the consumer boundary |
| Warp pointer/stride failure | `ProxyArray` passed where concrete Warp storage is required | use `.warp` at that exact call |
| Unselected environments change | `_index`/`_mask` data shape mismatch | compact rows versus full-size masked buffer |
| Missing `omni.physics.tensors.impl` | private module path removed | import `omni.physics.tensors.api` |
| No contact bodies or wrong ordering | nested importer hierarchy/view grouping | resolved full prim paths and env-major order |
| Duplicate/self ray hits | articulation descendants traversed more than once | stop traversal at rigid-body boundaries |
| Gray but valid video | camera is absolute or cameras enabled too late | `ViewerCfg` origin/tracking and launch order |
| Import abort before launch | task registration imports USD/Kit modules eagerly | `.pyi` plus `lazy_export()` and deferred imports |
| Recreate loop grows memory | global caches retain sensor/asset/FK objects | cache ownership and close/reset invalidation |
| ONNX downgrade traceback | Torch 2.10 cannot lower requested opset | export explicitly at opset 18 and check file |
| Good short rollout, bad learning | physics/contact or observation schema drift | fixed-seed curves, term meanings, longer canary |

## Completion checklist

A downstream PhysX migration is ready for release only when:

- [ ] runtime and all editable repositories are pinned by full SHA/version;
- [ ] the 2.3.2 reference remains available and read-only;
- [ ] no live code uses removed private imports or `SimulationCfg.physx`;
- [ ] schema configs are intentionally common or PhysX-specific;
- [ ] internal quaternions are XYZW and every other convention is a named boundary;
- [ ] checkpoint and observation compatibility has an explicit decision;
- [ ] Isaac Lab `ProxyArray` data crosses to Torch/Warp explicitly;
- [ ] all writes have verified `_index` or `_mask` semantics;
- [ ] policy-facing joint/body ordering is name-defined;
- [ ] generated assets have recorded inputs, configuration, tool revision, and digest;
- [ ] custom sensor full/partial/reset/recreate/debug gates pass;
- [ ] every task family passes reset, rollout, and a finite training canary;
- [ ] play, resume, video pixel content, and opset-18 export pass;
- [ ] numerical and representative performance comparisons are accepted;
- [ ] known deviations and untested gates are recorded rather than implied to pass;
- [ ] rollback has been exercised from clean processes.

## Evidence from the InstinctLab migration

The InstinctLab migration used this process for seven G1 task families. The resulting PhysX stack
passed all registered config instantiations; dataset-ready reset/rollout and one-update training
canaries; contact, ray, camera, volume-point, partial-update, and recreation tests; video inspection;
checkpoint resume; multiple ONNX variants; a world-size-one distributed canary; controlled
2.3.2/3.0 comparisons; a 20-update learning comparison; and representative performance tests.

The two remaining hardware/process caveats at the time of writing are true multi-GPU execution and
multi-device ray-cache isolation on a host with more than one GPU. They do not invalidate the
single-GPU PhysX migration, but they must not be reported as tested.

Project-specific evidence and decisions remain in:

- [`ISAACLAB_3_0_0_BETA2_UPGRADE_PLAN.md`](ISAACLAB_3_0_0_BETA2_UPGRADE_PLAN.md)
- [`PROGRESS.md`](PROGRESS.md)
- [`UPGRADE_REQUIREMENTS_and_INFO.md`](UPGRADE_REQUIREMENTS_and_INFO.md)

## Upstream references

- [Isaac Lab 3.0 migration guide](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/migration/migrating_to_isaaclab_3-0.html)
- [Working with ProxyArray](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/how-to/proxy_array.html)
- [Isaac Lab 3.0.0-beta2 PhysX backend](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/overview/core-concepts/physical-backends/physx/index.html)
- [Pinned Isaac Lab branch](https://github.com/isaac-sim/IsaacLab/tree/release/3.0.0-beta2)
- [Pinned target commit](https://github.com/isaac-sim/IsaacLab/commit/6a7acb0320a0bdc15b13e44e83b575e00797faf4)
