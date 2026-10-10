import sys
import imageio
import numpy as np
import ocbench

ep_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
d = np.load("data/block-single-task1-v0.npz")
ends = np.where(d["terminals"])[0] + 1
start, end = [0, *ends][ep_idx], ends[ep_idx]
print(f"episode {ep_idx}: steps {start}-{end} (length {end - start})")

env = ocbench.make("block-cpu-single-task1-v0", width=512, height=512)
env.reset()
frames = []
for qpos, qvel in zip(d["qpos"][start:end], d["qvel"][start:end]):
    env.unwrapped.set_state(qpos, qvel)
    frames.append(env.render())
env.close()
imageio.mimsave(f"ocbench_episode_{ep_idx}.mp4", frames, fps=50)
print("saved", f"ocbench_episode_{ep_idx}.mp4")
