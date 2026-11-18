"""Shared plotting style and helpers for the PBL-2 scripts (same palette as PBL-1)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PBL2 = Path("pbl-2")
SEED = 42
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
INK, INK2, MUTED, GRID, SURFACE, LIGHT = ("#0b0b0b", "#52514e", "#898781", "#e1e0d9",
                                          "#fcfcfb", "#c9c8c1")
PALETTE = [BLUE, ORANGE, AQUA, PURPLE, "#d1a21f", "#d4508f", "#3fa3b8", "#7a8c2e",
           "#b5562f", "#5466b8", "#2f8f5b", "#a0522d", "#6b6b6b", "#c27ba0", "#4a90a4"]
KM_PER_DEG = 111.32

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=150)
    plt.close(fig)


def md_table(df):
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def to_km(lat, lon, lat0, lon0):
    """Local equirectangular projection to kilometres. Over this ~45 km area its distance error
    against haversine is tiny (reported by the EDA script)."""
    x = (np.asarray(lon) - lon0) * KM_PER_DEG * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * KM_PER_DEG
    return np.column_stack([x, y])


def round_floats(obj, sig=6):
    if isinstance(obj, float):
        return float(f"{obj:.{sig}g}")
    if isinstance(obj, dict):
        return {k: round_floats(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, sig) for v in obj]
    if isinstance(obj, np.generic):
        return round_floats(obj.item(), sig)
    return obj


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(round_floats(obj), indent=2, ensure_ascii=False), encoding="utf-8")
