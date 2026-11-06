#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:22:15 2026

@author: tokart
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.close('all')


""" SETTINGS """

FIGSIZE = (14, 6.5)

SUB_W = 1.0
SUB_D = 0.8
SUB_H = 0.6

# Identical limits and physical scale for all panels.
XLIM = (-6, 6)
YLIM = (-6, 6)
ZLIM = (0, 4)

ELEVATION = 22
AZIMUTH = -58
ZOOM = 1.10


""" GEOMETRY HELPERS """

def source(x, y, z=SUB_H/2, angle=0, color="white"):
    # z is the cabinet centre height.
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "angle": float(angle),
        "color": color
    }


def corners_2d(src):
    angle = np.deg2rad(src["angle"])

    corners = np.array([
        [-SUB_W/2, -SUB_D/2],
        [SUB_W/2, -SUB_D/2],
        [SUB_W/2, SUB_D/2],
        [-SUB_W/2, SUB_D/2]
    ])

    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    return corners @ rotation.T + [src["x"], src["y"]]


def centre_configuration(sources):
    # Centre the complete configuration in x and y.
    corners = np.concatenate([corners_2d(src) for src in sources])
    centre_x, centre_y = (corners.min(axis=0) + corners.max(axis=0))/2

    # Place the bottom of the lowest cabinet at z = 0.
    bottom = min(src["z"] - SUB_H/2 for src in sources)

    return [
        source(
            src["x"] - centre_x,
            src["y"] - centre_y,
            src["z"] - bottom,
            src["angle"],
            src["color"]
        )
        for src in sources
    ]


""" ARRAY GEOMETRIES """

# Five cabinets, with the bottom of the first cabinet at z = 0.
vertical_stack = [
    source(0, 0, SUB_H/2 + i*SUB_H)
    for i in range(5)
]

# Nine cabinets on a semicircle of radius 5 m.
theta = np.linspace(0, np.pi, 9)

semicircular_array = [
    source(5*np.cos(t), 5*np.sin(t), angle=np.degrees(t) + 90)
    for t in theta
]

# Nine contiguous cabinets forming a physical arc.
angles = np.arange(-4, 5)*7.5
radius = SUB_W/(2*np.sin(np.deg2rad(7.5)/2))

physical_arcsub = [
    source(radius*np.sin(np.deg2rad(a)), radius*(1 - np.cos(np.deg2rad(a))), angle=a)
    for a in angles
]

# Six cabinets: + / - / + / + / - / + from bottom to top.
vertical_cardioid = [
    source(0, 0, SUB_H/2 + i*SUB_H, color="red" if i in (1, 4) else "white")
    for i in range(6)
]

# Two adjacent rows, three cabinets deep.
large_endfire = [
    source(x, y)
    for x in (-SUB_W/2, SUB_W/2)
    for y in (1.20, 0.0, -1.20)
]

# Geometry retained from the reference image.
d_arc = 1.20
side_offset = 2.2
side_angle = 25.0
a = np.deg2rad(side_angle)

gradient_endfire = [
    source(0, -d_arc),
    source(0, 0),
    source(0, d_arc),

    source(-side_offset, -d_arc, angle=-side_angle),
    source(-side_offset + d_arc*np.sin(a), -d_arc + d_arc*np.cos(a), angle=-side_angle),

    source(side_offset, -d_arc, angle=side_angle),
    source(side_offset - d_arc*np.sin(a), -d_arc + d_arc*np.cos(a), angle=side_angle)
]

configurations = [
    ("Vertical stack", vertical_stack),
    ("Semicircular array", semicircular_array),
    ("Physical arc-sub", physical_arcsub),
    ("Vertical cardioid", vertical_cardioid),
    ("Large end-fire", large_endfire),
    ("Gradient end-fire", gradient_endfire)
]


""" CABINET DRAWING """

def draw_sub_3d(ax, src):
    corners = corners_2d(src)

    bottom = [[x, y, src["z"] - SUB_H/2] for x, y in corners]
    top = [[x, y, src["z"] + SUB_H/2] for x, y in corners]

    faces = [bottom, top] + [
        [bottom[i], bottom[(i + 1) % 4], top[(i + 1) % 4], top[i]]
        for i in range(4)
    ]

    cabinet = Poly3DCollection(faces, facecolors=src["color"], edgecolors="black", linewidths=0.8)
    ax.add_collection3d(cabinet)


""" 3D SCENE """

def draw_scene(ax, sources, title):
    sources = centre_configuration(sources)

    for src in sources:
        draw_sub_3d(ax, src)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    ax.set_xlabel("x [m]", fontsize=9, labelpad=1)
    ax.set_ylabel("y [m]", fontsize=9, labelpad=1)
    ax.set_zlabel("z [m]", fontsize=9, labelpad=1)

    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_zlim(ZLIM)

    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([-5, 0, 5])
    ax.set_zticks([0, 2, 4])
    ax.tick_params(labelsize=8, pad=0)

    ax.set_box_aspect((np.ptp(XLIM), np.ptp(YLIM), np.ptp(ZLIM)), zoom=ZOOM)
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False


""" FIGURE """

fig = plt.figure(figsize=FIGSIZE)

gs = fig.add_gridspec(
    2, 3,
    left=0.035, right=0.965,
    bottom=0.14, top=0.92,
    wspace=0.12, hspace=0.30
)

for index, (title, sources) in enumerate(configurations):
    ax = fig.add_subplot(gs[index//3, index % 3], projection="3d")
    draw_scene(ax, sources, title)

fig.text(
    0.5, 0.055,
    "All configurations are evaluated in both ground-stacked and flown arrangements.",
    ha="center", fontsize=11
)

plt.show()