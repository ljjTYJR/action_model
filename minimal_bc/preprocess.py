# Preprocess the data: percentile-based action normalization.
import numpy as np


class ActionNormalizer:
    """Map each action dimension's [1st, 99th] percentile range to [-1, 1]."""

    def __init__(self, center, half_range, clip=5.0):
        self.center = center
        self.half_range = half_range
        self.clip = clip

    @classmethod
    def fit(cls, actions, percentile=1.0, clip=5.0):
        # Fit on the training set only, then reuse the same statistics everywhere.
        low = np.percentile(actions, percentile, axis=0)
        high = np.percentile(actions, 100 - percentile, axis=0)
        center, half_range = (low + high) / 2, (high - low) / 2
        center[-1], half_range[-1] = 0.0, 1.0  # gripper is already in [-1, 1]; leave it as is
        return cls(center.astype(np.float32), half_range.astype(np.float32), clip)

    def normalize(self, actions):
        # Env actions -> training targets. Clip rare outliers so they can't dominate the MSE.
        return np.clip((actions - self.center) / self.half_range, -self.clip, self.clip)

    def unnormalize(self, actions):
        # Model outputs -> env actions, which must lie in [-1, 1].
        return np.clip(actions * self.half_range + self.center, -1, 1)


if __name__ == '__main__':
    from data import read_data
    _, acts, _ = read_data('data/block-single-task1-v0-10k.npz')
    normalizer = ActionNormalizer.fit(acts)
    acts_n = normalizer.normalize(acts)
    print("center", normalizer.center.round(4))
    print("half_range", normalizer.half_range.round(4))
    print("fraction in [-1, 1] per dim", (np.abs(acts_n) <= 1).mean(0).round(3))
    print("max round-trip error inside the clip", np.abs(normalizer.unnormalize(acts_n) - acts)[np.abs(acts_n) < 5].max())
