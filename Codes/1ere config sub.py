#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:06:53 2026

@author: tokart

Couverture SPL de differentes configurations de subwoofers.

Chaque configuration est une liste de caisses (Sub). Chaque caisse porte sa
position, son orientation et son traitement electronique (retard, gain,
polarite), ce qui permet de decrire aussi bien un simple line array au sol
qu'un end-fire, un cardioide ou un arc a delais.

Pour ajouter une configuration : ecrire une ligne dans la liste CONFIGS.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import List

plt.close('all')

# PARAMETRES GENERAUX
c = 343.0
f_min = 30
f_max = 124
n_freq = 94
frequencies = np.linspace(f_min, f_max, n_freq)

# DIMENSIONS D'UN SUBWOOFER
sub_width = 1.0
sub_depth = 0.8
sub_height = 0.6

# DISCRETISATION DE LA FACE AVANT
Ny = 7
Nz = 5

# ZONE DE SIMULATION
x = np.linspace(-40, 40, 100)
y = np.linspace(-20, 20, 100)
X, Y = np.meshgrid(x, y)
z_listener = 1.8


# DESCRIPTION D'UNE CAISSE ET D'UNE CONFIGURATION
@dataclass
class Sub:
    """Une caisse. (x, y, z) = centre geometrique de la caisse.

    yaw_deg : rotation autour de l'axe vertical, 0 = face avant vers +x.
    delay_ms, gain_db, polarity : traitement applique a la caisse.
    """
    x: float
    y: float
    z: float
    yaw_deg: float = 0.0
    delay_ms: float = 0.0
    gain_db: float = 0.0
    polarity: int = 1


@dataclass
class Config:
    name: str
    subs: List[Sub] = field(default_factory=list)
    # bande d'etude propre a la configuration ; None = bande globale 30-124 Hz
    freqs: object = None


# CREATION DES MONOPOLES SUR LA FACE AVANT D'UN SUB
def create_sub_sources(sub):
    """Grille de monopoles sur la face avant, tournee de yaw_deg autour du centre."""
    dy = np.linspace(-sub_width / 2, sub_width / 2, Ny)
    dz = np.linspace(-sub_height / 2 + 0.05, sub_height / 2 - 0.05, Nz)
    a = np.deg2rad(sub.yaw_deg)
    ca, sa = np.cos(a), np.sin(a)
    dx = sub_depth / 2
    sources = []
    for ys in dy:
        for zs in dz:
            sources.append([sub.x + dx * ca - ys * sa,
                            sub.y + dx * sa + ys * ca,
                            sub.z + zs])
    return np.array(sources)


# CALCUL SPL
def compute_spl(subs, freqs=None, Xg=None, Yg=None, z=None):
    """SPL large bande (moyenne energetique) aux points d'observation demandes.

    freqs  : bande d'etude, par defaut `frequencies` (30-124 Hz).
    Xg, Yg : points d'observation, par defaut la grille globale X, Y.
    z      : hauteur d'observation, par defaut z_listener.

    Appelee avec une seule frequence et un anneau de points, elle donne la
    directivite azimutale ; c'est la meme sommation de champ dans les deux cas.
    """
    freqs = frequencies if freqs is None else np.atleast_1d(freqs)
    Xg = X if Xg is None else Xg
    Yg = Y if Yg is None else Yg
    z = z_listener if z is None else z

    source_gain = 1.0 / (Ny * Nz)
    # une entree par monopole : position, poids reel, retard
    positions, amplitudes, delays = [], [], []
    for sub in subs:
        amp = source_gain * sub.polarity * 10.0 ** (sub.gain_db / 20.0)
        for p in create_sub_sources(sub):
            positions.append(p)
            amplitudes.append(amp)
            delays.append(sub.delay_ms * 1e-3)

    # Boucle source a l'exterieur, frequence a l'interieur : les distances ne
    # dependent pas de la frequence, on evite ainsi de les recalculer pour
    # chacune des len(freqs) frequences. Le champ complexe est garde pour toutes
    # les frequences a la fois, ce qui coute len(freqs) x grille et ne depend
    # donc pas du nombre de caisses.
    nf = len(freqs)
    p_total = np.zeros((nf,) + np.shape(Xg), dtype=complex)
    for (xs, ys, zs), amp, tau in zip(positions, amplitudes, delays):
        d2 = (Xg - xs)**2 + (Yg - ys)**2
        r_direct = np.sqrt(d2 + (z - zs)**2)
        r_image = np.sqrt(d2 + (z + zs)**2)
        for j, f in enumerate(freqs):
            k = 2 * np.pi * f / c
            H_direct = np.exp(-1j * k * r_direct) / r_direct
            H_image = np.exp(-1j * k * r_image) / r_image
            w = amp if tau == 0.0 else amp * np.exp(-2j * np.pi * f * tau)
            p_total[j] += w * (H_direct + H_image)

    # accumulation frequence par frequence, dans le meme ordre qu'avant
    energy_total = np.zeros_like(Xg)
    for j in range(nf):
        energy_total += np.abs(p_total[j])**2
    p_rms = np.sqrt(energy_total / nf)
    return 20 * np.log10(p_rms + 1e-12)


# CONSTRUCTEURS DE CONFIGURATIONS
def single(x=0.0, y=0.0, **kw):
    """Une caisse posee au sol."""
    return [Sub(x, y, sub_height / 2, **kw)]


def line(n, spacing=sub_width, x=0.0, y_center=0.0, **kw):
    """n caisses cote a cote au sol, alignees suivant y (array horizontal)."""
    ys = (np.arange(n) - (n - 1) / 2) * spacing + y_center
    return [Sub(x, float(yy), sub_height / 2, **kw) for yy in ys]


def stack(n, x=0.0, y=0.0, z_bottom=0.0, **kw):
    """n caisses empilees a la verticale, la premiere posee a z_bottom."""
    return [Sub(x, y, z_bottom + (i + 0.5) * sub_height, **kw) for i in range(n)]


def endfire(n, spacing, y=0.0, x_front=0.0, aim_deg=0.0, **kw):
    """n caisses en enfilade, retardees pour sommer dans la direction aim_deg.

    aim_deg = 0 : tir vers +x. L'axe de l'enfilade et l'orientation des caisses
    tournent ensemble. La caisse la plus en arriere joue en premier (retard
    nul) ; chaque caisse plus en avant est retardee du temps de parcours qui
    la separe, de sorte que tous les fronts d'onde partent ensemble vers
    l'avant et se soustraient vers l'arriere.

    (x_front, y) est la position de la caisse la plus avancee.
    """
    a = np.deg2rad(aim_deg)
    ca, sa = np.cos(a), np.sin(a)
    subs = []
    for i in range(n):
        back = (n - 1 - i) * spacing        # recul le long de l'axe de tir
        subs.append(Sub(x_front - back * ca, y - back * sa, sub_height / 2,
                        yaw_deg=aim_deg,
                        delay_ms=i * spacing / c * 1e3, **kw))
    return subs


def cardioid(spacing, y=0.0, x_front=0.0, **kw):
    """Paire cardioide : caisse arriere inversee et retardee de spacing / c.

    Rejection vers l'arriere (-x). Spacing usuel : quart d'onde a la frequence
    de calage.
    """
    return [Sub(x_front, y, sub_height / 2, **kw),
            Sub(x_front - spacing, y, sub_height / 2,
                delay_ms=spacing / c * 1e3, polarity=-1, **kw)]


def arc_physical(n, radius, spacing=sub_width, x_front=0.0, y_center=0.0, **kw):
    """Arc SUB PHYSIQUE : n caisses posees sur un arc convexe vers +x.

    Les caisses sont reparties avec un pas de `spacing` le long de l'arc ; les
    caisses exterieures reculent donc geometriquement ET pivotent vers
    l'exterieur. Le centre de courbure est derriere l'array, en x_front - radius.
    """
    d_ang = spacing / radius
    angles = (np.arange(n) - (n - 1) / 2) * d_ang
    return [Sub(x_front + radius * (np.cos(a) - 1.0),
                y_center + radius * np.sin(a),
                sub_height / 2, yaw_deg=np.rad2deg(a), **kw)
            for a in angles]


def arc_virtual(n, radius, spacing=sub_width, x=0.0, y_center=0.0, **kw):
    """Arc SUB VIRTUEL : n caisses alignees suivant y, courbure obtenue par delai.

    Chaque caisse est retardee du temps de parcours du recul qu'elle aurait sur
    un arc physique de meme rayon : radius - sqrt(radius^2 - dy^2).

    A la difference de l'arc physique, les caisses restent toutes face a +x :
    le delai reproduit le recul, jamais le pivotement.
    """
    subs = line(n, spacing=spacing, x=x, y_center=y_center, **kw)
    out = []
    for s in subs:
        dy = s.y - y_center
        recul = radius - np.sqrt(max(radius**2 - dy**2, 0.0))
        out.append(replace(s, delay_ms=s.delay_ms + recul / c * 1e3))
    return out


def flown(subs, z_bottom):
    """Remonte un groupe de caisses : z_bottom = hauteur du bas de la caisse."""
    dz = z_bottom + sub_height / 2 - min(s.z for s in subs)
    return [replace(s, z=s.z + dz) for s in subs]


def mirror(subs):
    """Symetrique par rapport au plan y = 0 (pour construire un systeme L/R)."""
    return [replace(s, y=-s.y, yaw_deg=-s.yaw_deg) for s in subs]


def shift(subs, dx=0.0, dy=0.0, dz=0.0):
    """Translate un groupe de caisses."""
    return [replace(s, x=s.x + dx, y=s.y + dy, z=s.z + dz) for s in subs]


def reversed_box(sub):
    """Passe une caisse en mode cardioide : retournee a 180 deg, polarite
    inversee, retardee du temps de parcours de la profondeur de caisse.

    Sa face avant recule donc de sub_depth : vers l'arriere elle arrive en meme
    temps que les caisses avant mais en opposition de phase -> annulation large
    bande. Vers l'avant elle arrive sub_depth * 2 / c plus tard, d'ou la perte
    de niveau en basse frequence caracteristique du cardioide.
    """
    return replace(sub,
                   yaw_deg=sub.yaw_deg + 180.0,
                   polarity=-sub.polarity,
                   delay_ms=sub.delay_ms + sub_depth / c * 1e3)


def cardioid_column(n=4, cardioid_index=1, x=0.0, y=0.0, z_bottom=0.0):
    """Colonne de n caisses empilees, dont une passee en cardioide."""
    subs = stack(n, x=x, y=y, z_bottom=z_bottom)
    subs[cardioid_index] = reversed_box(subs[cardioid_index])
    return subs


def cardioid_row(n=4, cardioid_index=1, x=0.0, y_center=0.0, spacing=sub_width):
    """Rangee de n caisses au sol cote a cote, dont une passee en cardioide."""
    subs = line(n, spacing=spacing, x=x, y_center=y_center)
    subs[cardioid_index] = reversed_box(subs[cardioid_index])
    return subs


def endfire_rows(n_rows, n_per_row, spacing, y_center=0.0, x_front=0.0,
                 aim_deg=0.0, row_pitch=sub_width, **kw):
    """n_rows lignes end-fire paralleles, decalees perpendiculairement au tir."""
    a = np.deg2rad(aim_deg)
    ca, sa = np.cos(a), np.sin(a)
    offs = (np.arange(n_rows) - (n_rows - 1) / 2) * row_pitch
    subs = []
    for o in offs:
        # decalage le long de la perpendiculaire a l'axe de tir : (-sa, ca)
        subs += endfire(n_per_row, spacing, aim_deg=aim_deg,
                        x_front=x_front - float(o) * sa,
                        y=y_center + float(o) * ca, **kw)
    return subs


def pinwheel_cluster(n_col=4, R=1.0, n_el=4, dz=sub_height, h_s=8.0,
                     x_center=0.0, y_center=0.0, pinwheel_offset_deg=0.0, **kw):
    """Grappe centrale volee : n_col colonnes verticales autour d'un axe commun.

    Geometrie (vue de dessus, colonne k = 0..n_col-1) :
      theta_k = k * 360 / n_col
      position azimutale de la colonne : theta_k + pinwheel_offset_deg
      centre de la colonne : (x_center + R cos, y_center + R sin) sur ce cercle
      orientation (yaw) de toutes les caisses de la colonne : theta_k

    Avec pinwheel_offset_deg = 0 chaque colonne regarde radialement vers
    l'exterieur (disposition en croix). L'offset decale la position azimutale
    par rapport a l'axe de rayonnement : a 90 deg les colonnes regardent
    tangentiellement et s'emboitent en moulinet.

    Le modele de caisse a une directivite propre (face avant discretisee), donc
    theta_k est bien applique a l'axe de rayonnement via yaw_deg, pas seulement
    a la position — conformement a l'hypothese de depart.

    n_el caisses par colonne, pas vertical dz, centre de la grappe a h_s
    au-dessus du sol.

    HYPOTHESE DE VALIDITE : l'omnidirectionnalite azimutale suppose que la
    distance entre colonnes adjacentes d = 2 R sin(pi / n_col) reste inferieure
    a lambda / 2 a la frequence haute de la bande. Au-dela, des lobes de reseau
    entrent dans le domaine visible. Voir check_grating_lobes().
    """
    subs = []
    # colonne centree verticalement sur h_s
    z_offsets = (np.arange(n_el) - (n_el - 1) / 2) * dz
    for k in range(n_col):
        theta = k * 360.0 / n_col
        phi = np.deg2rad(theta + pinwheel_offset_deg)
        xc = x_center + R * np.cos(phi)
        yc = y_center + R * np.sin(phi)
        for zo in z_offsets:
            subs.append(Sub(xc, yc, h_s + float(zo), yaw_deg=theta, **kw))
    return subs


def column_spacing(n_col, R):
    """Distance entre deux colonnes adjacentes sur le cercle de rayon R."""
    return 2.0 * R * np.sin(np.pi / n_col)


def check_grating_lobes(n_col, R, f_high, label=""):
    """Avertit si l'espacement entre colonnes adjacentes depasse lambda/2.

    Retourne (ok, d, f_limite). Au-dessus de f_limite = c / (2 d), des lobes de
    reseau apparaissent dans le domaine visible et l'hypothese
    d'omnidirectionnalite azimutale tombe.
    """
    d = column_spacing(n_col, R)
    f_limit = c / (2.0 * d)
    ok = f_high <= f_limit
    if ok:
        print(f"[OK]      {label} d = {d:.2f} m entre colonnes, lambda/2 "
              f"atteint a {f_limit:.0f} Hz > f_max = {f_high:.0f} Hz.")
    else:
        print(f"[ATTENTION] {label} d = {d:.2f} m entre colonnes > lambda/2 "
              f"des {f_limit:.0f} Hz, or la bande monte a {f_high:.0f} Hz.\n"
              f"            Des lobes de reseau apparaissent en azimut au-dela "
              f"de {f_limit:.0f} Hz : l'hypothese d'omnidirectionnalite n'est "
              f"plus valable sur le haut de la bande.\n"
              f"            Reduire R sous {c / (2.0 * f_high) / (2.0 * np.sin(np.pi / n_col)):.2f} m "
              f"ou limiter la bande a {f_limit:.0f} Hz.")
    return ok, d, f_limit


def endfire_from_rear(n, spacing, aim_deg, rear_x, rear_y, **kw):
    """End-fire ancre par sa caisse ARRIERE, utile pour poser un array lateral
    contre le bord d'un bloc existant sans que les deux se chevauchent."""
    subs = endfire(n, spacing, aim_deg=aim_deg, **kw)
    rear = subs[0]                      # la caisse arriere = retard nul
    return shift(subs, dx=rear_x - rear.x, dy=rear_y - rear.y)


# ---------------------------------------------------------------------------
# CONFIGURATIONS A SIMULER
# Ajouter ici une ligne par configuration a tester.
# ---------------------------------------------------------------------------
FLY_HEIGHT = 8.0        # hauteur du bas de caisse pour les configs volees [m]
LR_OFFSET = 10.0        # demi-ecart des deux arrays L/R de la config 3b [m]
                        # = arrays aux extremites d'une scene de 20 m de large
CARDIOID_INDEX = 1      # rang de la caisse cardioide dans chaque groupe de 4

REFERENCE = Config("1 subwoofer", single())

# --- reference et premiers essais ---------------------------------------
BASE = [
    REFERENCE,
    Config("10 subwoofers - horizontal array", line(10)),
    Config("10 subwoofers - vertical stack", stack(10)),
]

# --- 1) cardioide : 2 stacks de 4 accoles, 1 caisse sur 4 en cardioide ---
cardio_vertical = (cardioid_column(4, CARDIOID_INDEX, y=-sub_width / 2)
                   + cardioid_column(4, CARDIOID_INDEX, y=+sub_width / 2))
cardio_horizontal = (cardioid_row(4, CARDIOID_INDEX, y_center=-2 * sub_width)
                     + cardioid_row(4, CARDIOID_INDEX, y_center=+2 * sub_width))

CARDIOID = [
    Config("Cardioid - ground-stacked, vertical (2x4)", cardio_vertical),
    Config("Cardioid - ground-stacked, horizontal (2x4)", cardio_horizontal),
    Config(f"Cardioid - flown at {FLY_HEIGHT:.0f} m (2x4)",
           flown(cardio_vertical, FLY_HEIGHT)),
]

# --- 2) end-fire au sol : 2 puis 4 rangees, deux espacements -------------
ENDFIRE_GROUND = [
    Config(f"End-fire ground - {n} rows x 4, spacing {s:.1f} m",
           endfire_rows(n, 4, s))
    for s in (1.2, 0.8) for n in (2, 4)
]

# --- 3) end-fire vole : centre, puis deux arrays symetriques L/R ---------
ENDFIRE_FLOWN = []
for s in (1.2, 0.8):
    ENDFIRE_FLOWN.append(Config(
        f"End-fire flown - centred, 2 rows x 4, spacing {s:.1f} m",
        flown(endfire_rows(2, 4, s), FLY_HEIGHT)))
    ENDFIRE_FLOWN.append(Config(
        f"End-fire flown - L/R +/-{LR_OFFSET:.0f} m, 2x(2 rows x 4), spacing {s:.1f} m",
        flown(endfire_rows(2, 4, s, y_center=-LR_OFFSET), FLY_HEIGHT)
        + flown(endfire_rows(2, 4, s, y_center=+LR_OFFSET), FLY_HEIGHT)))

# --- 4) grappe centrale volee : colonnes verticales en croix / moulinet --
CLUSTER_N_COL = 4                       # nombre de colonnes
CLUSTER_R = 1.0                         # rayon axe central -> colonne [m]
CLUSTER_N_EL = 4                        # caisses par colonne
CLUSTER_DZ = sub_height                 # pas vertical dans une colonne [m]
CLUSTER_H_S = 8.0                       # hauteur du centre de grappe / sol [m]
CLUSTER_OFFSET = 0.0                    # 0 = croix (tir radial), 90 = moulinet
BAND_CORTEEL = np.linspace(30, 80, 51)  # bande d'etude 30-80 Hz (Corteel 2018)

cluster = pinwheel_cluster(n_col=CLUSTER_N_COL, R=CLUSTER_R, n_el=CLUSTER_N_EL,
                           dz=CLUSTER_DZ, h_s=CLUSTER_H_S,
                           pinwheel_offset_deg=CLUSTER_OFFSET)

_cluster_tag = f"{CLUSTER_N_COL} col x {CLUSTER_N_EL} el, R = {CLUSTER_R:.1f} m"
CLUSTER = [
    # meme bande que toutes les autres figures -> directement comparable
    Config(f"Flown cluster ({_cluster_tag}) - 30-124 Hz", cluster),
    # bande d'etude demandee ; reference recalculee dans la meme bande
    Config(f"Flown cluster ({_cluster_tag}) - 30-80 Hz", cluster,
           freqs=BAND_CORTEEL),
]

# --- 5) arc sub : physique (caisses sur l'arc) vs virtuel (arc par delai) -
ARC_N = 8
ARC_RADIUS = 15.0

ARC = [
    Config(f"Arc physical - {ARC_N} subs, R = {ARC_RADIUS:.0f} m",
           arc_physical(ARC_N, ARC_RADIUS)),
    Config(f"Arc virtual (delay only) - {ARC_N} subs, R = {ARC_RADIUS:.0f} m",
           arc_virtual(ARC_N, ARC_RADIUS)),
]

# --- 6) end-fire gradient : 2 arrays avant + 2 arrays lateraux a +/-60 ----
SIDE_AIM = 60.0         # angle de tir des arrays lateraux [deg]
SIDE_ANCHOR_Y = 1.5     # position en y de la caisse arriere des arrays lateraux

GRADIENT = []
for s in (1.2, 0.8):
    front = endfire_rows(2, 4, s)                      # 8 caisses vers +x
    side_left = endfire_from_rear(4, s, +SIDE_AIM, 0.0, +SIDE_ANCHOR_Y)
    side_right = endfire_from_rear(4, s, -SIDE_AIM, 0.0, -SIDE_ANCHOR_Y)
    GRADIENT.append(Config(
        f"End-fire gradient - 2 front + 2 side at +/-{SIDE_AIM:.0f} deg, "
        f"spacing {s:.1f} m",
        front + side_left + side_right))

# regroupement en figures : (titre, nombre de colonnes, configurations)
FIGURES = [
    ("Subwoofer SPL distribution - 30-124 Hz", 1, BASE),
    ("Cardioid arrays - 8 subs (2 stacks of 4, 1 box in 4 cardioid)", 1, CARDIOID),
    ("End-fire ground-stacked - 4 boxes per row", 2, ENDFIRE_GROUND),
    (f"End-fire flown at {FLY_HEIGHT:.0f} m - 4 boxes per row", 2, ENDFIRE_FLOWN),
    (f"Flown cluster - {CLUSTER_N_COL} vertical columns around a common axis "
     f"(h_s = {CLUSTER_H_S:.0f} m)", 2, CLUSTER),
    (f"Arc subwoofer array - physical vs virtual, R = {ARC_RADIUS:.0f} m", 2, ARC),
    ("End-fire gradient - front + side arrays, 16 subs", 2, GRADIENT),
]

CONFIGS = [cfg for _, _, group in FIGURES for cfg in group]


# CALCUL
# controle de validite de la grappe : lobes de reseau en azimut
check_grating_lobes(CLUSTER_N_COL, CLUSTER_R, BAND_CORTEEL[-1],
                    label="grappe, bande 30-80 Hz :")
check_grating_lobes(CLUSTER_N_COL, CLUSTER_R, f_max,
                    label="grappe, bande 30-124 Hz :")

# Les configurations sont independantes : on les repartit sur les coeurs.
# Threads et non processus -- numpy relache le GIL dans ses boucles internes,
# et on evite le "spawn" de Windows (qui reexecute le script dans chaque
# enfant et supporte mal un nom de fichier contenant des espaces).
# Chaque configuration reste calculee exactement comme avant : le resultat est
# identique bit pour bit, seul l'ordre d'execution entre configurations change.
N_WORKERS = min(os.cpu_count() or 1, len(CONFIGS))


def _compute_one(cfg):
    return cfg.name, compute_spl(cfg.subs, freqs=cfg.freqs)


print(f"Calcul de {len(CONFIGS)} configurations "
      f"({sum(len(c.subs) for c in CONFIGS)} caisses) sur {N_WORKERS} threads...")
_t0 = time.time()
with ThreadPoolExecutor(max_workers=N_WORKERS) as _pool:
    results = dict(_pool.map(_compute_one, CONFIGS))
print(f"Calcul termine en {time.time() - _t0:.1f} s")

# REFERENCE = MAX DU CAS 1 SUB, evaluee dans la meme bande que la config tracee,
# pour que l'echelle de couleur garde le meme sens d'une figure a l'autre
_ref_cache = {}


def reference_level(freqs):
    key = "default" if freqs is None else tuple(np.round(np.atleast_1d(freqs), 6))
    if key not in _ref_cache:
        _ref_cache[key] = np.max(compute_spl(REFERENCE.subs, freqs=freqs))
    return _ref_cache[key]


refs = {cfg.name: reference_level(cfg.freqs) for cfg in CONFIGS}
ref = reference_level(None)


# AFFICHAGE
def sub_footprint(sub):
    """Contour de la caisse vue de dessus, oriente suivant yaw_deg."""
    a = np.deg2rad(sub.yaw_deg)
    ca, sa = np.cos(a), np.sin(a)
    corners = [(-sub_depth / 2, -sub_width / 2), (sub_depth / 2, -sub_width / 2),
               (sub_depth / 2, sub_width / 2), (-sub_depth / 2, sub_width / 2)]
    return [(sub.x + dx * ca - dy * sa, sub.y + dx * sa + dy * ca)
            for dx, dy in corners]


def plot_azimuthal_directivity(subs, freqs=(30, 50, 80), radius=25.0,
                               n_az=181, title=None):
    """SPL en fonction de l'azimut (0-360 deg) a plusieurs frequences.

    Les points sont pris sur un cercle de rayon `radius` dans le plan du public
    (z = z_listener). Chaque courbe est normalisee sur sa propre moyenne, ce qui
    isole la forme de directivite de l'ecart de niveau entre frequences : une
    config parfaitement omnidirectionnelle donnerait un cercle.
    """
    az = np.linspace(0, 2 * np.pi, n_az)
    Xr = (radius * np.cos(az))[None, :]
    Yr = (radius * np.sin(az))[None, :]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2),
                             subplot_kw={}, constrained_layout=True)
    axes[1].remove()
    ax_pol = fig.add_subplot(1, 2, 2, projection="polar")
    ax_lin = axes[0]

    for f in freqs:
        spl = compute_spl(subs, freqs=[f], Xg=Xr, Yg=Yr)[0]
        rel = spl - spl.mean()
        ripple = spl.max() - spl.min()
        lab = f"{f:.0f} Hz  (ondulation {ripple:.1f} dB)"
        ax_lin.plot(np.rad2deg(az), rel, linewidth=1.6, label=lab)
        ax_pol.plot(az, rel, linewidth=1.6, label=lab)

    ax_lin.set_xlabel("Azimut [deg]", fontsize=10)
    ax_lin.set_ylabel("Niveau relatif a la moyenne azimutale [dB]", fontsize=10)
    ax_lin.set_xlim(0, 360)
    ax_lin.set_xticks(np.arange(0, 361, 45))
    ax_lin.grid(True, linewidth=0.5, alpha=0.4)
    ax_lin.legend(fontsize=9)
    ax_lin.tick_params(labelsize=9)

    ax_pol.set_theta_zero_location("E")
    ax_pol.grid(True, linewidth=0.5, alpha=0.4)
    ax_pol.tick_params(labelsize=8)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


def plot_configs(configs, results, refs, ncols=1, title=None):
    nrows = int(np.ceil(len(configs) / ncols))
    fig_h = 3.0 * nrows + 1.0
    fig = plt.figure(figsize=(7.2 * ncols, fig_h))
    # on reserve une bande fixe de 0.6 pouce en haut pour le titre general,
    # sinon il chevauche les titres de panneaux de la premiere rangee
    top = 1.0 - 0.6 / fig_h
    gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[30] * ncols + [1],
                          hspace=0.32, wspace=0.12,
                          left=0.09, right=0.93, top=top, bottom=0.06)
    cax = fig.add_subplot(gs[:, ncols])

    levels = np.arange(-30, 22, 2)
    cmap = "viridis"
    contour = None

    for i, cfg in enumerate(configs):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        contour = ax.contourf(X, Y, results[cfg.name] - refs[cfg.name],
                              levels=levels, cmap=cmap, extend="both")
        for sub in cfg.subs:
            # pointille = caisse en l'air (volee), plein = posee au sol
            is_flown = sub.z - sub_height / 2 > 0.01
            ax.add_patch(Polygon(sub_footprint(sub), closed=True,
                                 facecolor="white", edgecolor="black",
                                 linewidth=1.2,
                                 linestyle="--" if is_flown else "-"))
        ax.set_title(cfg.name, fontsize=11, fontweight="bold", pad=5)
        ax.set_xlim(-40, 40)
        ax.set_ylim(-20, 20)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")
        ax.set_xlabel("x [m]", fontsize=10)
        ax.set_ylabel("y [m]", fontsize=10)
        ax.set_xticks(np.arange(-40, 41, 10))
        ax.set_yticks(np.arange(-20, 21, 10))
        ax.tick_params(labelsize=9)
        ax.grid(True, linewidth=0.5, alpha=0.4)

    cbar = fig.colorbar(contour, cax=cax, ticks=np.arange(-30, 19, 6))
    cbar.set_label("Level relative to 1-sub maximum [dB]", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    pos = cbar.ax.get_position()
    cbar.ax.set_position([pos.x0 - 0.03, pos.y0, pos.width, pos.height])

    if title:
        # on reduit la police si le titre est trop long pour la largeur de figure
        fs = min(15.0, 460.0 * ncols / (0.62 * max(len(title), 1)))
        fig.suptitle(title, fontsize=fs, fontweight="bold",
                     x=0.5, y=1.0 - 0.22 / fig_h)
    return fig


for title, ncols, group in FIGURES:
    plot_configs(group, results, refs, ncols=ncols, title=title)

# directivite azimutale de la grappe : la config est-elle vraiment omni ?
plot_azimuthal_directivity(
    cluster, freqs=(30, 50, 80), radius=25.0,
    title=f"Flown cluster - horizontal directivity at 25 m ({_cluster_tag})")

plt.show()
