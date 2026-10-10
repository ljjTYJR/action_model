# Read and process the generated oc-bench data.
import numpy as np

def read_data(path):
    data = np.load(path)
    print("data includes", data.files)
    # Each episode stores T actions and T + 1 observations (the extra one is the final observation).
    # Drop the final observation of every episode so that observations[i] pairs with actions[i].
    actions = data['actions']
    ends = np.where(data['terminals'])[0] + 1  # exclusive end of each episode, in action indices
    final_obs_idx = ends + np.arange(len(ends))  # episode k's observations are shifted by k rows
    observations = np.delete(data['observations'], final_obs_idx, axis=0)
    return observations, actions, ends


if __name__ == '__main__':
    data_path = 'data/block-single-task1-v0.npz'
    observations, actions, ends = read_data(data_path)
    print("observations shape", observations.shape)
    print("actions shape", actions.shape)
    print("episodes", len(ends))
