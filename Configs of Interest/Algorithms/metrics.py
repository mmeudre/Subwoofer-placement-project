"""Metriques communes a toutes les configurations : bright zone, variance
spatiale du SPL en fonction de la frequence, effort.

Convention d'axes (commune a tous les scripts de Configs of Interest/) :
    x : axe de tir, la foule est du coté x > 0
    y : axe lateral, les zones sont centrees sur y = 0
    z : vertical, sol en z = 0
    x = 0 : plan de la face avant la plus en avant de l'array

Description d'un array : une ligne par caisse.
    sources : (L, 3) centres acoustiques [m]
    tau     : (L,)   retards appliques par le DSP [s]
    gain    : (L,)   gains signes (la polarite est portee par le signe)

Le poids d'une caisse a la frequence f est q_l(f) = gain_l * exp(-j 2 pi f tau_l) :
un retard en secondes, pas une phase figee a une frequence de reglage.
"""
import numpy as np

c = 343.0
rho = 1.2
Q = 1e-3
p_ref = 20e-6

frequencies = np.arange(30, 121, 1.0)

# 70 mailles sur 50 m : pas de 0.714 m, juste sous lambda/4 a 120 Hz (0.7146 m)
zone_step = 50.0/70

# fenetres de couverture autour de la mediane [dB]
tolerances = (3.0, 6.0)


# ============================================================
# BRIGHT ZONE
# ============================================================

def bright_zone(kind, step=zone_step, size=50.0, x_start=3.0, z=1.6, centre=(0.0, 0.0)):
    """Points de mesure (M, 3), au centre de chaque maille de cote `step`.

    kind = 'concert' : carre de x_start a x_start + size en x, centre en y = 0.
    kind = 'stadium' : carre centre en `centre` (x, y), sous l'array.
    """
    n = int(round(size/step))
    offsets = (np.arange(n) + 0.5)*step

    if kind == 'concert':
        xs = x_start + offsets
        ys = offsets - size/2
    elif kind == 'stadium':
        xs = centre[0] + offsets - size/2
        ys = centre[1] + offsets - size/2
    else:
        raise ValueError(f"kind doit valoir 'concert' ou 'stadium', pas {kind!r}")

    X, Y = np.meshgrid(xs, ys)
    return np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)]


def array_centre(sources):
    """Centre (x, y) de l'array, pour placer une zone 'stadium' dessous."""
    sources = np.asarray(sources, float)
    return sources[:, 0].mean(), sources[:, 1].mean()


# ============================================================
# MODELE ACOUSTIQUE : MONOPOLES + IMAGE PAR LE SOL
# ============================================================

def weights(f, tau, gain):
    """q[f, l] = gain_l * exp(-j 2 pi f tau_l), de forme (F, L)."""
    f = np.atleast_1d(f)[:, None]
    return np.asarray(gain, float)[None, :]*np.exp(-1j*2*np.pi*f*np.asarray(tau, float)[None, :])


def transfer_matrix(points, sources, f, reflection=1.0):
    """G[m, l] a la frequence f : pression au point m pour la caisse l de
    debit Q, plus son image par le sol ponderee par `reflection`."""
    k = 2*np.pi*f/c
    points = np.asarray(points, float)[:, None, :]
    sources = np.asarray(sources, float)[None, :, :]
    image = sources*np.array([1.0, 1.0, -1.0])

    r_direct = np.maximum(np.linalg.norm(points - sources, axis=2), 0.05)
    r_image = np.maximum(np.linalg.norm(points - image, axis=2), 0.05)
    return 1j*rho*c*k*Q/(4*np.pi)*(np.exp(-1j*k*r_direct)/r_direct
                                   + reflection*np.exp(-1j*k*r_image)/r_image)


def pressure(points, sources, q, f, reflection=1.0):
    """p[f, m] pour des poids q[f, l]. Une frequence a la fois, pour ne pas
    stocker G sur toute la bande."""
    f = np.atleast_1d(f)
    return np.array([transfer_matrix(points, sources, fi, reflection) @ qi
                     for fi, qi in zip(f, q)])


# ============================================================
# METRIQUES
# ============================================================

def spl(p):
    """L[f, m] = 20 log10(|p| / p_ref) [dB]."""
    return 20*np.log10(np.maximum(np.abs(p), 1e-12)/p_ref)


def spatial_variance(p):
    """sigma^2(f) = (1/M) sum_m (L_m(f) - Lbar(f))^2 [dB^2].
    Independante de p_ref : un decalage commun des L_m ne la change pas."""
    return np.var(spl(p), axis=-1)


def spatial_std(p):
    """sigma(f) [dB] : ecart typique d'un point au niveau moyen de la zone."""
    return np.sqrt(spatial_variance(p))


def coverage(p, tolerance=3.0):
    """Couverture utile : part de la zone a +/- tolerance [dB] de la mediane, a chaque f.

    La mediane est le niveau de l'auditeur typique (la moitie de la zone au-dessus,
    l'autre en dessous) ; contrairement a la moyenne, des trous profonds ne la
    deplacent pas. Les points etant repartis uniformement, une part des points
    est une part de la surface de public.
    Renvoie (deficit, exces, couverture), chacun (F,) entre 0 et 1 :
        deficit    : part a plus de `tolerance` dB sous la mediane (trous)
        exces      : part a plus de `tolerance` dB au-dessus (pics)
        couverture : 1 - deficit - exces
    """
    L = spl(p)
    D = L - np.median(L, axis=-1, keepdims=True)
    deficit = np.mean(D < -tolerance, axis=-1)
    excess = np.mean(D > tolerance, axis=-1)
    return deficit, excess, 1 - deficit - excess


def band_average(g, f=frequencies):
    """Moyenne de g(f) sur la bande, ponderee par 1/f : chaque octave pese autant.
    Sur la grille lineaire, l'octave 60-120 Hz compte deux fois plus de points que
    30-60 Hz ; le poids 1/f compense (df/f = d ln f)."""
    return np.average(g, weights=1/np.asarray(f, float), axis=-1)


def effort(q, p, p_inphase):
    """E_ff(f) = 10 log10(q^H q / |q_r|^2) [dB] (eq. 16 de Vindrola et al.).

    q_r : toutes les caisses en phase a gain unitaire, recalees pour donner la
    meme energie moyenne dans la zone que les poids q.
        q          : (F, L) poids evalues
        p          : (F, M) pression produite par q
        p_inphase  : (F, M) pression produite par q = 1 sur toutes les caisses
    """
    scale2 = np.mean(np.abs(p)**2, axis=-1)/np.mean(np.abs(p_inphase)**2, axis=-1)
    L = q.shape[-1]
    return 10*np.log10(np.sum(np.abs(q)**2, axis=-1)/(L*scale2))


def evaluate(sources, tau, gain, points, f=frequencies, reflection=1.0):
    """sigma^2(f), sigma(f), couverture utile a +/- 3 et 6 dB et E_ff(f) d'un array sur une zone donnee.
    Les grandeurs de couverture sont indexees par la tolerance : r['coverage'][3.0], etc."""
    q = weights(f, tau, gain)
    p = pressure(points, sources, q, f, reflection)
    p_inphase = pressure(points, sources, np.ones_like(q), f, reflection)
    r = {'f': np.atleast_1d(f),
         'variance': spatial_variance(p),
         'std': spatial_std(p),
         'deficit': {}, 'excess': {}, 'coverage': {},
         'effort': effort(q, p, p_inphase)}
    for tol in tolerances:
        r['deficit'][tol], r['excess'][tol], r['coverage'][tol] = coverage(p, tol)
    return r


def summarize(r):
    """Scalaires d'une config sur la bande : sigma^2 moyennee (1/f) et, pour chaque
    tolerance, couverture moyenne (1/f), couverture a la pire frequence, deficit et
    exces moyens (1/f). Couvertures, deficits et exces en %."""
    f = r['f']
    s = {'variance': band_average(r['variance'], f)}
    for tol in tolerances:
        s[f'coverage_{tol:g}'] = 100*band_average(r['coverage'][tol], f)
        s[f'coverage_min_{tol:g}'] = 100*r['coverage'][tol].min()
        s[f'deficit_{tol:g}'] = 100*band_average(r['deficit'][tol], f)
        s[f'excess_{tol:g}'] = 100*band_average(r['excess'][tol], f)
    return s


# ============================================================
# CONFIGURATIONS DE Configs of Interest/
# ============================================================

def load_configs(folder=None):
    """Importe chaque script de config (ceux qui exposent SCENARIO et array()).
    Renvoie {nom : module}, dans l'ordre alphabetique des fichiers."""
    import importlib.util
    from pathlib import Path
    folder = Path(__file__).resolve().parent.parent if folder is None else Path(folder)
    configs = {}
    for path in sorted(folder.glob('*.py')):
        spec = importlib.util.spec_from_file_location(f'config_{len(configs)}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'SCENARIO') and hasattr(module, 'array'):
            configs[path.stem] = module
    return configs


def evaluate_config(module, f=frequencies, step=zone_step):
    """Metriques d'une config dans son scenario, a sa hauteur Z_START."""
    sources, tau, gain = module.array()
    zone = bright_zone(module.SCENARIO, step=step, centre=array_centre(sources))
    return evaluate(sources, tau, gain, zone, f)


if __name__ == '__main__':
    configs = load_configs()
    results = {}
    for name, module in configs.items():
        print(f'{name} ({module.SCENARIO})')
        results[name] = evaluate_config(module)

    # Tableau des metriques, par scenario, trie par sigma^2 moyennee croissante.
    # Les configs ne se comparent qu'a l'interieur d'un meme scenario.
    t3, t6 = (f'{tol:g}' for tol in tolerances)
    header = (f"{'Configuration':28s} {'sigma2':>7s} | {'+/-' + t3 + ' dB : couv.':>17s} {'pire f':>7s} "
              f"{'deficit':>7s} {'exces':>6s} | {'+/-' + t6 + ' dB : couv.':>17s} {'pire f':>7s} "
              f"{'deficit':>7s} {'exces':>6s}")
    for scenario in ('concert', 'stadium'):
        rows = sorted(((name, summarize(r)) for name, r in results.items()
                       if configs[name].SCENARIO == scenario), key=lambda row: row[1]['variance'])
        print(f"\n{scenario.upper()} : sigma2 en dB^2 (moyenne 1/f sur {frequencies[0]:g}-{frequencies[-1]:g} Hz), "
              f"couvertures en % de la zone a +/- tolerance de la mediane")
        print(header)
        print('-'*len(header))
        for name, s in rows:
            print(f"{name:28s} {s['variance']:7.1f} | "
                  f"{s['coverage_' + t3]:17.1f} {s['coverage_min_' + t3]:7.1f} "
                  f"{s['deficit_' + t3]:7.1f} {s['excess_' + t3]:6.1f} | "
                  f"{s['coverage_' + t6]:17.1f} {s['coverage_min_' + t6]:7.1f} "
                  f"{s['deficit_' + t6]:7.1f} {s['excess_' + t6]:6.1f}")

    # Trace de sigma^2(f) pour toutes les configs : desactive, illisible en l'etat
    # (26 courbes qui se croisent ; on compare les configs par le tableau ci-dessus).
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(16, 8))
    # colors = plt.get_cmap('tab20').colors
    # linestyles = {'concert': '-', 'stadium': '--'}
    # for i, (name, r) in enumerate(results.items()):
    #     scenario = configs[name].SCENARIO
    #     ax.plot(r['f'], r['variance'], color=colors[i % len(colors)], linestyle=linestyles[scenario],
    #             linewidth=1.5, label=f'{name} ({scenario})')
    # ax.set_xlabel(r'$f$ [Hz]')
    # ax.set_ylabel(r'Variance spatiale $\sigma^2$ [dB$^2$]')
    # ax.set_xlim(frequencies[0], frequencies[-1])
    # ax.grid(True, alpha=0.3)
    # ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    # ax.set_title(r'Variance spatiale du SPL sur la bright zone ($50 \times 50\ \mathrm{m}$, '
    #              r'pas $0.71\ \mathrm{m}$; trait plein : concert, tirets : stadium',
    #              fontsize=12, fontweight='bold')
    # fig.tight_layout()
    # plt.show()
