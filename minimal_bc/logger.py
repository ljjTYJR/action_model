# Record, print and plot training / validation losses.
import torch
import matplotlib
matplotlib.use('Agg')  # headless: write image files without a display
import matplotlib.pyplot as plt

# Computer Modern Typewriter, the font of LaTeX \texttt. Used for titles, labels and legend only:
# it lacks the minus/times glyphs that log-scale tick labels need, so ticks keep the default font.
TT = 'cmtt10'


class Logger:
    def __init__(self, print_every=100):
        self.print_every = print_every
        self.train_losses = []  # every step's loss, kept on GPU to avoid a sync per step
        self.valid_log = []  # (step, loss) pairs

    def log_train(self, step, loss):
        self.train_losses.append(loss.detach())
        if step % self.print_every == 0:
            print(f"step {step:6d} | train loss {loss.item():.6f}")

    def log_valid(self, step, loss):
        self.valid_log.append((step, loss.item()))
        print(f"step {step:6d} | valid loss {loss.item():.6f}")

    def plot(self, path):
        train_log = list(enumerate(torch.stack(self.train_losses).tolist()))
        plot_losses(train_log, self.valid_log, path)


def ema(values, weight=0.99):
    # Exponential moving average, as in TensorBoard's smoothing slider.
    smoothed, last = [], values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def plot_losses(train_log, valid_log, path):
    # Two separate panels: white background, black frame and black lines.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, facecolor='white')
    for ax, log, title in zip(axes, [train_log, valid_log], ['Train loss', 'Valid loss']):
        steps, losses = zip(*log)
        if log is train_log:
            ax.plot(steps, losses, color='0.75', linewidth=1, label='raw')
            ax.plot(steps, ema(losses), color='black', linewidth=1.5, label='smoothed (EMA 0.99)')
            ax.legend(frameon=False, prop={'family': TT, 'size': 11})
        else:
            ax.plot(steps, losses, color='black', linewidth=1.5)
        ax.set_yscale('log')  # loss spans orders of magnitude
        ax.set_title(title, fontfamily=TT, fontsize=14)
        ax.set_xlabel('Step', fontfamily=TT, fontsize=12)
        ax.set_ylabel('MSE loss', fontfamily=TT, fontsize=12)
        for spine in ax.spines.values():
            spine.set_color('black')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor='white')
    print(f"saved {path}")
