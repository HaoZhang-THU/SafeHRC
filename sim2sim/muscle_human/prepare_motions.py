import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "egl")

from loco_mujoco.task_factories import AMASSDatasetConf, ImitationFactory

CLIPS = ["KIT/314/walking_medium07_poses", "KIT/4/WalkingStraightBackwards08_poses", "CMU/143/143_41_poses"]
GMR = {"src_human": "smplh", "target_fps": 30, "solver": "daqp", "damping": 0.5,
       "offset_to_ground": False, "use_velocity_limit": False, "verbose": False}


def main() -> None:
    for clip in CLIPS:
        conf = AMASSDatasetConf([clip])
        conf.retargeting_method, conf.gmr_config = "gmr", GMR
        ImitationFactory.make("MyoFullBody", amass_dataset_conf=conf, headless=True)
        print("ready", clip, flush=True)


if __name__ == "__main__":
    main()
