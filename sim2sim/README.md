# MuJoCo sim2sim

Runs checkpoints written by `train.sh` in MuJoCo, outside Isaac Lab, with the same metrics as
`eval.sh`.

## Install

```bash
conda create -n sage_mujoco python=3.11 && conda activate sage_mujoco
pip install -r sim2sim/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Run

From the repository root:

```bash
python sim2sim/run_sage.py --checkpoint results/<run>/models/iter_200.pt --out sim2sim_eval.json
MUJOCO_GL=egl python sim2sim/run_sage.py --checkpoint results/<run>/models/iter_200.pt \
    --num_episodes 1 --workers 1 --video sim2sim_ep0.mp4
```

The scene is read from the run's `run_args.json`; `--scene_json` selects another one.
