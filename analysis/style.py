"""One visual language for every figure in the paper."""
import matplotlib as mpl
import matplotlib.pyplot as plt

INK   = "#1a1a1a"
MUTED = "#8a8f98"
GRID  = "#dfe3e8"
ACC   = "#1f4e79"      # primary
ACC2  = "#b3472f"      # contrast
ACC3  = "#4a7c59"      # third
FILL  = "#e8eef4"

# One colour scheme for agents, used in every figure that draws them.
WORKER   = "#1f4e79"   # dark blue
SOLDIER  = "#d97706"   # amber
PREDATOR = "#7f1d1d"   # dark red
HULL_A   = "#cfdce8"   # parent collective, shaded
HULL_B   = "#f3ddb8"   # daughter collective, shaded

def use():
    mpl.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": INK, "axes.linewidth": 0.7,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "grid.alpha": 1.0, "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

def panel_label(ax, s, dx=-0.13, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")

def bare(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(GRID)
