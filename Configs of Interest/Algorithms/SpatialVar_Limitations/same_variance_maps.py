"""Limites de la variance spatiale : des cartes de SPL tres differentes, meme sigma^2.

sigma^2 = (1/M) sum_m (L_m - Lbar)^2 est une somme de carres positifs : sigma^2 = 0
impose L_m = Lbar partout (carte uniforme). Une carte trouee ne peut donc pas avoir
une variance nulle. En revanche, a variance egale et non nulle, on peut construire
des couvertures tres differentes :

    B : decroissance en 1/r d'une source ponctuelle  -> sigma^2 = S (reference)
    C1 : fond uniforme + trous et pics etroits       -> sigma^2 = S
    C2 : fond uniforme + trous et pics larges        -> sigma^2 = S
    D : les niveaux de B melanges au hasard          -> sigma^2 = S (meme histogramme)

Une sigma^2 faible est donc necessaire mais pas suffisante pour attester de
l'homogeneite : la variance ne voit ni la forme de la distribution des niveaux
(beaucoup de petits ecarts ou quelques trous profonds), ni leur disposition dans
la zone.

P90-P10 : ecart entre le niveau depasse par 10 % de la zone et celui depasse par
90 % ; les 80 % centraux du public sont dans cette fenetre. Il ignore les 10 % les
plus faibles et les 10 % les plus forts : robuste aux creux isoles, mais aveugle a
un defaut qui touche moins de 10 % de la zone (voir C1).

Bright zone : celle de metrics.py (concert, 50 x 50 m, pas 50/70 m, z = 1.6 m).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from metrics import bright_zone

L0 = 100.0                        # niveau de reference [dB SPL]
src = np.array([0.0, 0.0, 0.3])   # source ponctuelle de la carte B (caisse au sol, mi-hauteur)
rng = np.random.default_rng(0)

zone = bright_zone('concert')
x, y = zone[:, 0], zone[:, 1]
n = int(round(np.sqrt(len(zone))))


def stats(L):
    """sigma^2, ecart interdecile P90-P10 et fraction de la zone a plus de 6 dB sous la mediane."""
    D = L - np.median(L)
    p10, p90 = np.percentile(D, [10, 90])
    return np.var(L), p90 - p10, 100*np.mean(D < -6), D.min(), D.max()


# B : decroissance en 1/r, recentree sur L0
r = np.linalg.norm(zone - src, axis=1)
L_B = -20*np.log10(r)
L_B += L0 - L_B.mean()
S = np.var(L_B)

def holes_and_peaks(n_holes, n_peaks, width, margin, seed=0):
    """Fond a L0 + trous et pics gaussiens d'ecart-type `width` [m], amplitude
    ajustee pour que sigma^2 = S exactement."""
    rng = np.random.default_rng(seed)
    centres = np.c_[rng.uniform(x.min() + margin, x.max() - margin, n_holes + n_peaks),
                    rng.uniform(y.min() + margin, y.max() - margin, n_holes + n_peaks)]
    signs = np.r_[-np.ones(n_holes), 0.6*np.ones(n_peaks)]
    g = np.zeros(len(zone))
    for (cx, cy), s in zip(centres, signs):
        g += s*np.exp(-((x - cx)**2 + (y - cy)**2)/(2*width**2))
    g -= g.mean()
    return L0 + g*np.sqrt(S/np.var(g))


# C1 : trous et pics etroits et profonds ; C2 : larges et moins profonds
L_C1 = holes_and_peaks(14, 10, width=1.2, margin=3)
L_C2 = holes_and_peaks(10, 7, width=2.5, margin=5)

# D : permutation aleatoire des points de B
L_D = rng.permutation(L_B)

maps = [('B : décroissance en 1/r', L_B),
        ('C1 : trous et pics étroits', L_C1),
        ('C2 : trous et pics larges', L_C2),
        ('D : B mélangée', L_D)]

fig, axes = plt.subplots(2, 2, figsize=(12, 11), sharex=True, sharey=True, layout='constrained')
extent = [x.min(), x.max(), y.min(), y.max()]
for ax, (title, L) in zip(axes.flat, maps):
    var, idr, frac, dmin, dmax = stats(L)
    im = ax.imshow(L.reshape(n, n), origin='lower', extent=extent, cmap='inferno',
                   vmin=L0 - 25, vmax=L0 + 15, interpolation='nearest')
    ax.set_title(rf'$\bf{{{title.split(" :")[0]}}}$ : {title.split(": ")[1]}' + '\n'
                 + rf'$\sigma^2$ = {var:.1f} dB$^2$ ; P90-P10 = {idr:.1f} dB ; {frac:.1f} % < -6 dB' + '\n'
                 + f'min / max vs médiane : {dmin:+.1f} / {dmax:+.1f} dB', fontsize=10)
    print(f'{title:26s} sigma2 = {var:6.2f} dB2 | P90-P10 = {idr:5.1f} dB | '
          f'{frac:5.1f} % sous -6 dB | min {dmin:+6.1f} max {dmax:+6.1f} dB')
for ax in axes[1]:
    ax.set_xlabel('x [m]')
for ax in axes[:, 0]:
    ax.set_ylabel('y [m]')
fig.colorbar(im, ax=axes, shrink=0.6, label='SPL [dB]')
fig.suptitle('Même variance spatiale, couvertures différentes\n'
             r'bright zone concert $50 \times 50$ m, $z$ = 1.6 m', fontsize=13, fontweight='bold')
fig.savefig(HERE / 'same_variance_maps.png', dpi=150, bbox_inches='tight')

if __name__ == '__main__':
    plt.show()
