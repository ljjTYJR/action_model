# Summarize runs/results.csv: one line per experiment, mean ± std over seeds.
import csv
from collections import defaultdict
import numpy as np

if __name__ == '__main__':
    rows = defaultdict(list)
    with open('runs/results.csv') as f:
        for r in csv.DictReader(f):
            rows[r['exp']].append(r)

    print(f"{'experiment':28s} {'seeds':>5s} {'success rate':>15s} {'valid loss':>19s}")
    for exp, rs in sorted(rows.items()):
        def stat(key):
            v = np.array([float(r[key]) for r in rs])
            return f"{v.mean():.4f} ± {v.std():.4f}"
        print(f"{exp:28s} {len(rs):5d} {stat('success_rate'):>15s} {stat('valid_loss'):>19s}")
