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

## Muscle-driven human

`--muscle_human` replaces carrier robot_2 with the muscle-driven MyoFullBody human, which runs in its own process
on [muscle-supermimic](https://github.com/kaibohex/muscle-supermimic) (Python 3.12):

```bash
git clone https://github.com/kaibohex/muscle-supermimic && cd muscle-supermimic && git checkout d73bbcb && uv sync
export MUSCLE_HUMAN_PYTHON=$PWD/.venv/bin/python
```

Its gait reference is built locally from AMASS. Set up [MuscleMimic](https://github.com/amathislab/musclemimic) with
GMR retargeting (AMASS, SMPL-H, `uv sync --extra smpl --extra gmr`, see its `docs/workflows.md`), then from the
repository root:

```bash
(cd <musclemimic> && uv run python <SAGE>/sim2sim/muscle_human/prepare_motions.py)
$MUSCLE_HUMAN_PYTHON sim2sim/muscle_human/build_data.py
```

Download `muscle_human.pt` from the release page and run:

```bash
python sim2sim/run_sage.py --checkpoint results/<run>/models/iter_200.pt --muscle_human muscle_human.pt
```
