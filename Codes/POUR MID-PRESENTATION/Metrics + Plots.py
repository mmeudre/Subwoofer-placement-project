#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 13:41:52 2026

@author: tokart
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import metrics as met

plt.close('all')


""" SETTINGS """

# None uses the default configuration folder defined in metrics.py.
CONFIG_FOLDER = None
# CONFIG_FOLDER = Path("/Users/tokart/.../Configs of Interest")

SCENARIO = "concert"
FREQUENCIES = np.arange(30.0, 121.0, 1.0)

ZONE_SIZE = 50.0
ZONE_STEP = met.zone_step
X_START = 3.0
Z_OBS = 1.6

TOLERANCE = 3.0
MAX_VARIANCE_GAP = 0.05
MIN_COVERAGE_GAP = 0.15

SMOOTHING_LENGTH = 3.0
REFLECTION = 1.0

NORMALIZE_MEDIAN = True
TARGET_MEDIAN = 100.0


""" LOAD CONFIGURATIONS """

folder = None if CONFIG_FOLDER is None else Path(CONFIG_FOLDER).expanduser()

modules = met.load_configs(folder)
modules = {name: module for name, module in modules.items() if module.SCENARIO == SCENARIO}

if len(modules) < 2:
    raise RuntimeError(
        f"At least two configurations are required for scenario '{SCENARIO}'. "
        "Check CONFIG_FOLDER."
    )


""" COMPUTE MAPS AND METRICS """

results = {}

for index, (name, module) in enumerate(modules.items(), start=1):
    print(f"[{index}/{len(modules)}] Computing {name}", flush=True)

    sources, tau, gain = module.array()

    centre = met.array_centre(sources)
    points = met.bright_zone(SCENARIO, step=ZONE_STEP, size=ZONE_SIZE, x_start=X_START, z=Z_OBS, centre=centre)

    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    q = met.weights(FREQUENCIES, tau, gain)
    p = met.pressure(points, sources, q, FREQUENCIES, reflection=REFLECTION)

    levels = met.spl(p)
    maps = levels.reshape(len(FREQUENCIES), len(ys), len(xs))

    deficit, excess, coverage = met.coverage(p, tolerance=TOLERANCE)

    # Used for selection only: displayed SPL maps are not smoothed.
    smooth_maps = gaussian_filter(maps, sigma=(0.0, SMOOTHING_LENGTH/dy, SMOOTHING_LENGTH/dx), mode="reflect")
    residual = maps - smooth_maps

    margin = max(1, int(np.ceil(SMOOTHING_LENGTH/min(dx, dy))))
    core = residual[:, margin:-margin, margin:-margin]

    if core.shape[1] == 0 or core.shape[2] == 0:
        raise ValueError("SMOOTHING_LENGTH is too large for the selected zone.")

    roughness = np.sqrt(np.mean(core**2, axis=(1, 2)))

    results[name] = {
        "maps": maps,
        "variance": met.spatial_variance(p),
        "coverage": coverage,
        "deficit": deficit,
        "excess": excess,
        "roughness": roughness,
        "xs": xs,
        "ys": ys
    }


""" SEARCH FOR CONTRASTING CONFIGURATIONS """

names = list(results)
candidates = []
largest_coverage_gap = 0.0

for i, name_a in enumerate(names):
    for name_b in names[i + 1:]:
        a = results[name_a]
        b = results[name_b]

        mean_variance = (a["variance"] + b["variance"])/2
        variance_gap = np.abs(a["variance"] - b["variance"])/np.maximum(mean_variance, 1e-12)

        for fi in np.flatnonzero(variance_gap <= MAX_VARIANCE_GAP):
            coverage_gap = abs(a["coverage"][fi] - b["coverage"][fi])
            largest_coverage_gap = max(largest_coverage_gap, coverage_gap)

            if a["coverage"][fi] >= b["coverage"][fi]:
                left_name, right_name = name_a, name_b
            else:
                left_name, right_name = name_b, name_a

            left = results[left_name]
            right = results[right_name]

            rough_left = float(left["roughness"][fi])
            rough_right = float(right["roughness"][fi])

            if coverage_gap < MIN_COVERAGE_GAP or rough_right <= rough_left:
                continue

            map_left = left["maps"][fi]
            map_right = right["maps"][fi]

            centred_left = map_left - np.median(map_left)
            centred_right = map_right - np.median(map_right)
            map_difference = np.sqrt(np.mean((centred_left - centred_right)**2))

            candidates.append({
                "left": left_name,
                "right": right_name,
                "fi": int(fi),
                "variance_gap": float(variance_gap[fi]),
                "coverage_gap": float(coverage_gap),
                "rough_left": rough_left,
                "roughness_gap": rough_right - rough_left,
                "map_difference": float(map_difference)
            })

if not candidates:
    raise RuntimeError(
        "No pair satisfies the selection criteria.\n"
        f"Largest coverage difference among variance-matched pairs: "
        f"{100*largest_coverage_gap:.1f} percentage points.\n"
        "Try MAX_VARIANCE_GAP = 0.10 or MIN_COVERAGE_GAP = 0.10."
    )


""" RANK CANDIDATES """

def percentile_ranks(values):
    values = np.asarray(values, dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)

    if len(unique) == 1:
        return np.zeros_like(values)

    midranks = np.cumsum(counts) - (counts + 1)/2
    return midranks[inverse]/max(len(values) - 1, 1)


left_coverages = [results[item["left"]]["coverage"][item["fi"]] for item in candidates]
regularity = [-item["rough_left"] for item in candidates]
coverage_gaps = [item["coverage_gap"] for item in candidates]
roughness_gaps = [item["roughness_gap"] for item in candidates]
map_differences = [item["map_difference"] for item in candidates]

scores = (
    percentile_ranks(left_coverages)
    + percentile_ranks(regularity)
    + percentile_ranks(coverage_gaps)
    + percentile_ranks(roughness_gaps)
    + percentile_ranks(map_differences)
)

order = sorted(range(len(candidates)), key=lambda i: (scores[i], -candidates[i]["variance_gap"]), reverse=True)
best = candidates[order[0]]

fi = best["fi"]
frequency = FREQUENCIES[fi]
selected = [best["left"], best["right"]]


""" PRINT RESULTS """

print("\nBest candidate pairs:")

for index in order[:5]:
    item = candidates[index]
    print(
        f"{FREQUENCIES[item['fi']]:g} Hz | "
        f"{item['left']} / {item['right']} | "
        f"variance difference: {100*item['variance_gap']:.2f}% | "
        f"coverage difference: {100*item['coverage_gap']:.1f} points"
    )

print(f"\nSelected frequency: {frequency:g} Hz")
print(f"Relative variance difference: {100*best['variance_gap']:.2f}%")

for name in selected:
    r = results[name]
    print(
        f"\n{name}\n"
        f"  Variance: {r['variance'][fi]:.3f} dB²\n"
        f"  Below range: {100*r['deficit'][fi]:.1f}%\n"
        f"  Coverage: {100*r['coverage'][fi]:.1f}%\n"
        f"  Above range: {100*r['excess'][fi]:.1f}%"
    )

if NORMALIZE_MEDIAN:
    print(f"\nDisplayed SPL medians are aligned to {TARGET_MEDIAN:g} dB.")


""" PREPARE BOTH TYPES OF MAP """

display_maps = []
classified_maps = []
extents = []

for name in selected:
    r = results[name]
    levels = r["maps"][fi].copy()

    # Classify original levels relative to their own spatial median.
    deviation = levels - np.median(levels)

    classes = np.ones(levels.shape, dtype=np.uint8)
    classes[deviation < -TOLERANCE] = 0
    classes[deviation > TOLERANCE] = 2
    classified_maps.append(classes)

    # A common level shift preserves variance and relative coverage.
    if NORMALIZE_MEDIAN:
        levels += TARGET_MEDIAN - np.median(levels)

    display_maps.append(levels)

    xs = r["xs"].copy()
    ys = r["ys"].copy()

    if SCENARIO == "stadium":
        xs -= xs.mean()
        ys -= ys.mean()

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    extents.append([xs[0] - dx/2, xs[-1] + dx/2, ys[0] - dy/2, ys[-1] + dy/2])

all_levels = np.concatenate([levels.ravel() for levels in display_maps])
vmin = 5*np.floor(all_levels.min()/5)
vmax = 5*np.ceil(all_levels.max()/5)

if vmax <= vmin:
    vmax = vmin + 5

coverage_definition = (
    f"Coverage: percentage of the audience area within "
    f"±{TOLERANCE:g} dB of the spatial median."
)


""" COMMON FIGURE SETTINGS """

FIGSIZE = (14, 6.5)

# Identical map positions in both figures.
MAP_POSITIONS = [
    [0.065, 0.20, 0.35, 0.64],
    [0.475, 0.20, 0.35, 0.64]
]

METRIC_BOX = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="red", linewidth=1.8)


def display_name(name):
    title = name.replace("_", " ")
    if title.lower() == "end-fire large":
        return "Large end-fire"
    return title


def create_figure():
    fig = plt.figure(figsize=FIGSIZE)
    axs = []

    for position in MAP_POSITIONS:
        ax = fig.add_axes(position)
        ax.set_xlabel("x [m]", fontsize=12)
        ax.tick_params(labelsize=11)
        axs.append(ax)

    axs[0].set_ylabel("y [m]", fontsize=12)
    axs[1].tick_params(labelleft=False)

    return fig, axs


def add_title_and_metric(ax, name, metric):
    ax.text(
        0.5, 1.15,
        f"{display_name(name)} — {frequency:g} Hz",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=15
    )

    ax.text(
        0.5, 1.045, metric,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=14, bbox=METRIC_BOX
    )


""" FIGURE 1 — SPATIAL SPL VARIANCE """

fig1, axs1 = create_figure()

for ax, name, levels, extent in zip(axs1, selected, display_maps, extents):
    r = results[name]

    image = ax.imshow(levels, origin="lower", extent=extent, cmap="inferno", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")

    metric = rf"$\sigma^2 = {r['variance'][fi]:.2f}\ \mathrm{{dB}}^2$"
    add_title_and_metric(ax, name, metric)

# Align the colorbar with the actual map height.
fig1.canvas.draw()
map_position = axs1[1].get_position()

cax = fig1.add_axes([0.87, map_position.y0, 0.018, map_position.height])
cbar = fig1.colorbar(image, cax=cax)
cbar.set_label("SPL [dB]", fontsize=12)
cbar.ax.tick_params(labelsize=11)

fig1.text(
    0.46, 0.055,
    "Spatial variance: mean squared deviation of SPL from its spatial mean.",
    ha="center", fontsize=11
)


""" FIGURE 2 — COVERAGE """

colors = ["royalblue", "limegreen", "red"]
cmap = ListedColormap(colors)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

fig2, axs2 = create_figure()

for ax, name, classes, extent in zip(axs2, selected, classified_maps, extents):
    r = results[name]

    ax.imshow(classes, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")

    metric = f"Coverage = {100*r['coverage'][fi]:.1f}%"
    add_title_and_metric(ax, name, metric)

legend_handles = [
    Patch(facecolor=colors[0], label=f"Below median\n− {TOLERANCE:g} dB"),
    Patch(facecolor=colors[1], label=f"Within ±{TOLERANCE:g} dB\nof median"),
    Patch(facecolor=colors[2], label=f"Above median\n+ {TOLERANCE:g} dB")
]

fig2.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(0.845, 0.52),
    frameon=False,
    fontsize=11,
    labelspacing=1.8,
    handlelength=1.4,
    handleheight=1.4
)

fig2.text(
    0.5, 0.055,
    f"Coverage: percentage of the audience area within ±{TOLERANCE:g} dB of the spatial median.",
    ha="center", fontsize=11
)

plt.show()