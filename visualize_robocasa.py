import sys
import av
import imageio
import numpy as np
import pandas as pd

ROOT = "/dataset_rc_rl/liwz7@xiaopeng.com/VLA_Programes/robocasa_data/robocasa365-pretrain-atomic"
CAMS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
ep_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

ep = pd.read_parquet(f"{ROOT}/meta/episodes").set_index("episode_index").loc[ep_idx]
df = pd.read_parquet(f"{ROOT}/data/chunk-{ep['data/chunk_index']:03d}/file-{ep['data/file_index']:03d}.parquet")
df = df[df.episode_index == ep_idx]
state = np.stack(df["observation.state"])
action = np.stack(df["action"])
print(ep["tasks"], "| state", state.shape, "| action", action.shape)


def read_video(cam):
    k = f"videos/observation.images.{cam}"
    path = f"{ROOT}/videos/observation.images.{cam}/chunk-{ep[k + '/chunk_index']:03d}/file-{ep[k + '/file_index']:03d}.mp4"
    t0 = ep[k + "/from_timestamp"]
    frames = []
    with av.open(path) as c:
        c.seek(int(t0 * av.time_base))
        for f in c.decode(video=0):
            if f.time >= t0 - 1e-3:
                frames.append(f.to_ndarray(format="rgb24"))
            if len(frames) == len(df):
                break
    return np.stack(frames)


frames = np.concatenate([read_video(c) for c in CAMS], axis=2)
imageio.mimsave(f"episode_{ep_idx}.mp4", frames, fps=20)
print("frames", frames.shape, "-> saved", f"episode_{ep_idx}.mp4")
