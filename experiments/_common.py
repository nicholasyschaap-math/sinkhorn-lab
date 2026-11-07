"""Shared helpers for the figure scripts: paths, resumable CSV output, styling.

Every run is appended to the CSV as soon as it finishes, so an interrupted
script loses at most the run in progress. Rerunning skips rows already present.
"""

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

# Reference palette (dataviz skill, references/palette.md), light surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]                            # categorical slots 1, 2
ORDINAL_BLUE = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ramp steps 250/400/550/700
STATUS = {"converged": "#0ca30c", "max_iter": "#fab219", "numerical_failure": "#d03b3b"}
STATUS_HATCH = {"converged": "", "max_iter": "..", "numerical_failure": "xx"}
STATUS_LABEL = {"converged": "converged", "max_iter": "hit iteration cap",
                "numerical_failure": "numerical failure"}


def single_thread():
    """Pin BLAS to one thread (call before importing numpy)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")


class ResumableCSV:
    """Append-only CSV keyed by a tuple of columns; skips keys already written."""

    def __init__(self, path, columns, key):
        self.path, self.columns, self.key = Path(path), list(columns), list(key)
        self.done = {}
        if self.path.exists():
            with self.path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames != self.columns:
                    raise RuntimeError(f"{self.path} has different columns; move it aside to regenerate")
                for row in reader:
                    self.done[tuple(row[k] for k in self.key)] = row
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="") as fh:
                csv.writer(fh).writerow(self.columns)

    def _k(self, values):
        return tuple(str(values[k]) for k in self.key)

    def has(self, **key_values):
        return self._k(key_values) in self.done

    def get(self, **key_values):
        return self.done[self._k(key_values)]

    def write(self, row):
        with self.path.open("a", newline="") as fh:
            csv.DictWriter(fh, self.columns).writerow(row)
        self.done[self._k(row)] = {k: str(v) for k, v in row.items()}


def style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2.0, "lines.markersize": 5,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    })
