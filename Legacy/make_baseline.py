#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fige l'etat de "1ere config sub.py" comme reference de non-regression.
A n'executer qu'une fois. Ne PAS relancer apres modification du script source.

Garde-fou : refuse d'ecraser tests/baseline_v0.npz s'il existe deja, sauf si
la variable d'environnement FORCE_BASELINE vaut "1". Cette reference ne doit
jamais etre regeneree par accident.
"""
import os
import sys
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "1ere config sub.py")
OUT = os.path.join(HERE, "..", "tests", "baseline_v0.npz")

if os.path.exists(OUT) and os.environ.get("FORCE_BASELINE") != "1":
    sys.exit(
        f"Refus d'ecraser {OUT} (deja existant). "
        "Definir FORCE_BASELINE=1 pour forcer la regeneration."
    )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

spec = importlib.util.spec_from_file_location("v0", os.path.abspath(SRC))
v0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v0)
plt.close("all")

data = {"grid_x": v0.X, "grid_y": v0.Y, "freqs": v0.frequencies,
        "z_listener": np.array(v0.z_listener)}
for cfg in v0.CONFIGS:
    data[f"spl/{cfg.name}"] = v0.results[cfg.name]
    data[f"ref/{cfg.name}"] = np.array(v0.refs[cfg.name])
    data[f"geom/{cfg.name}"] = np.array(
        [[s.x, s.y, s.z, s.yaw_deg, s.delay_ms, s.gain_db, s.polarity]
         for s in cfg.subs], dtype=float)
    data[f"band/{cfg.name}"] = (v0.frequencies if cfg.freqs is None
                                else np.atleast_1d(cfg.freqs))
np.savez_compressed(OUT, **data)
print(f"{len(v0.CONFIGS)} configurations figees dans {OUT}")
