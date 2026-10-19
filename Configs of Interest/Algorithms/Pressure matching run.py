import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from pathlib import Path

import pressure_matching as pm

plt.close('all')


# ============================================================
# CONFIGURATION TO OPTIMISE
# ============================================================

CONFIG = Path(__file__).parent.parent / 'Side Aim end-fire Martin.py'

# Bright zone [m], observation plane z = z_obs of the configuration
bright_x = (10, 50)
bright_y = (-20, 20)
n_mesh_x, n_mesh_y = 15, 11


def load_config(path):
    """Execute the configuration up to its acoustic model, i.e. parameters
    and geometry only : the configuration's own figures are not drawn."""
    src = Path(path).read_text(encoding='utf-8')
    src = src.split('# ACOUSTIC MODEL')[0]
    ns = {'__file__': str(path)}
    exec(compile(src, str(path), 'exec'), ns)
    return ns


cfg = load_config(CONFIG)
f, z_obs, X, Y = cfg['f'], cfg['z_obs'], cfg['X'], cfg['Y']
boxes, footprint = cfg['boxes'], cfg['footprint']
levels = np.linspace(-30, 30, 21)   # relative to the bright-zone mean level
band = 3                            # +-band around the mean level [dB]
z_start = cfg['z_ground']

sources = np.array([s[:3] for s in cfg['source_positions'](z_start)])
tau = np.array([s[3] for s in cfg['source_positions'](z_start)])


# ============================================================
# PRESSURE MATCHING
# ============================================================

mx, my = np.meshgrid(np.linspace(*bright_x, n_mesh_x), np.linspace(*bright_y, n_mesh_y))
mesh = np.c_[mx.ravel(), my.ravel(), np.full(mx.size, z_obs)]
G_B = pm.transfer_matrix(mesh, sources, f)

q_endfire = np.exp(-1j*2*np.pi*f*tau)
q_pm = pm.uniform_level_target(G_B)

# Same mean level in the bright zone for both, to compare at equal level
bright = ((X >= bright_x[0]) & (X <= bright_x[1])
          & (Y >= bright_y[0]) & (Y <= bright_y[1]))
points = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z_obs)]
G_map = pm.transfer_matrix(points, sources, f)


def pressure_map(q):
    return (G_map @ q).reshape(X.shape)


def mean_level(q):
    return np.mean(20*np.log10(np.abs(pressure_map(q)[bright])))


q_pm = q_pm*10**((mean_level(q_endfire) - mean_level(q_pm))/20)

p_endfire = pressure_map(q_endfire)
p_pm = pressure_map(q_pm)


# ============================================================
# EVALUATION
# ============================================================

source_zone = np.zeros_like(X, dtype=bool)
for xb, yb, yaw, t in boxes:
    source_zone |= ((np.abs(X-xb) <= cfg['sub_d']/2 + cfg['evaluation_margin'])
                    & (np.abs(Y-yb) <= cfg['sub_w']/2 + cfg['evaluation_margin']))
evaluation_zone = ~source_zone

ref_dB = mean_level(q_endfire)
SPL_endfire = 20*np.log10(np.maximum(np.abs(p_endfire), 1e-12)) - ref_dB
SPL_pm = 20*np.log10(np.maximum(np.abs(p_pm), 1e-12)) - ref_dB


def zone_stats(SPL):
    s = SPL[bright]
    return s.max() - s.min(), s.std()


dL_endfire, std_endfire = zone_stats(SPL_endfire)
dL_pm, std_pm = zone_stats(SPL_pm)

print(f'{len(boxes)} subwoofers, f = {f:g} Hz, bright zone x {bright_x} y {bright_y}')
print(f'  end-fire : dL = {dL_endfire:.2f} dB   std = {std_endfire:.2f} dB')
print(f'  PM       : dL = {dL_pm:.2f} dB   std = {std_pm:.2f} dB'
      f'   effort = {pm.effort(G_B, q_pm):.1f} dB')


# ============================================================
# FIGURE
# ============================================================

def plot_map(ax, SPL, title):
    im = ax.contourf(X, Y, np.clip(SPL, levels[0], levels[-1]),
                     levels=levels, cmap='viridis')
    ax.contour(X, Y, SPL, levels=[-band, band], colors='red', linewidths=1)
    ax.add_patch(Rectangle((bright_x[0], bright_y[0]),
                           bright_x[1]-bright_x[0], bright_y[1]-bright_y[0],
                           fill=False, edgecolor='white', linestyle='--', linewidth=1.2))
    for xb, yb, yaw, t in boxes:
        ax.add_patch(Polygon(footprint(xb, yb, yaw), closed=True,
                             facecolor='white', edgecolor='black', linewidth=1.0))
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_xlim(X[0, 0], X[0, -1])
    ax.set_ylim(Y[0, 0], Y[-1, 0])
    ax.set_aspect('equal')
    return im


fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.04], height_ratios=[1.5, 1],
                      left=0.05, right=0.93, bottom=0.08, top=0.93,
                      wspace=0.22, hspace=0.38)

ax_end = fig.add_subplot(gs[0, 0])
ax_pm = fig.add_subplot(gs[0, 1])
cax = fig.add_subplot(gs[0, 2])
ax_gain = fig.add_subplot(gs[1, 0])
ax_phase = fig.add_subplot(gs[1, 1])

im = plot_map(ax_end, SPL_endfire,
              rf'End-fire delays : $\Delta L = {dL_endfire:.1f}\ \mathrm{{dB}}$ in the bright zone')
plot_map(ax_pm, SPL_pm,
         rf'Pressure matching : $\Delta L = {dL_pm:.1f}\ \mathrm{{dB}}$ in the bright zone')

cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r'Level relative to the bright-zone mean [dB]', fontsize=12)

idx = np.arange(1, len(boxes) + 1)
ax_gain.bar(idx - 0.2, 20*np.log10(np.abs(q_endfire)), width=0.4, label='End-fire')
ax_gain.bar(idx + 0.2, 20*np.log10(np.abs(q_pm)), width=0.4, label='PM')
ax_gain.set_ylabel(r'$|q|$ [dB]')
ax_gain.set_xlabel('Subwoofer')
ax_gain.set_xticks(idx)
ax_gain.legend()

ax_phase.bar(idx - 0.2, np.rad2deg(np.angle(q_endfire)), width=0.4, label='End-fire')
ax_phase.bar(idx + 0.2, np.rad2deg(np.angle(q_pm)), width=0.4, label='PM')
ax_phase.set_ylabel(r'$\arg\,q$ [deg]')
ax_phase.set_xlabel('Subwoofer')
ax_phase.set_xticks(idx)
ax_phase.set_ylim(-180, 180)

fig.text(0.5, 0.01,
         rf'$f = {f:g}\ \mathrm{{Hz}}$, $z = {z_obs:g}\ \mathrm{{m}}$, equal mean level in the '
         rf'bright zone (dashed), red : $\pm {band}$ dB, $\sigma_L$ : {std_endfire:.2f} $\to$ {std_pm:.2f} dB, '
         rf'effort {pm.effort(G_B, q_pm):.1f} dB',
         ha='center', va='bottom')

plt.show()
