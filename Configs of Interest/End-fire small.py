import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.close('all')

NAME = 'End-fire small'

# ============================================================
# 1. GLOBAL PARAMETERS
# ============================================================
c = 343
rho = 1.2
f = 80                      # frequence des cartes
Q = 1e-3
k = 2*np.pi*f/c
omega = 2*np.pi*f
z_obs = 1.6
z_ground = 0
z_flown = 10
sub_w = 1.0                 # largeur, perpendiculaire a l'axe de tir
sub_d = 0.8                 # profondeur, selon l'axe de tir
sub_h = 0.6
evaluation_margin = 0.5
target = -3
level_min, level_max = -20, 0
levels = np.linspace(level_min, level_max, 21)
N_xy, N_vertical = 350, 260
x = np.linspace(-50, 50, N_xy)
y = np.linspace(-50, 50, N_xy)
z = np.linspace(0, 50, N_vertical)
X, Y = np.meshgrid(x, y)
X_xz, Z_xz = np.meshgrid(x, z)
y_slice = 0

# Scenario : 'concert' (foule en x > 0) ou 'stadium' (zone sous l'array)
SCENARIO = 'concert'
Z_START = z_ground

# ============================================================
# 2. ARRAY GEOMETRY
# ============================================================
# Convention commune : x = axe de tir (foule en x > 0), y lateral, z vertical.
# Chaque caisse : centre (x, y), bas zb au-dessus de z_start, direction de tir
# yaw [deg], retard tau [s] et gain signe applique par le DSP.
# Centre acoustique : centre de la caisse, a mi-hauteur. Dans une cardioide
# (on_face=True), il est sur la face du haut-parleur, a sub_d/2 selon yaw.
def box(xb, yb, zb=0.0, yaw=0.0, tau=0.0, gain=1.0, facecolor='white', on_face=False):
    return {'x': float(xb), 'y': float(yb), 'z': float(zb), 'yaw': float(yaw),
            'tau': float(tau), 'gain': float(gain), 'facecolor': facecolor,
            'on_face': bool(on_face)}


def geometry():
    # 2 piles x 2 caisses, espacement lambda/4 a f_design ; la pile avant est retardee
    f_design = 80
    spacing = c/(4*f_design)
    tau_small = spacing/c
    B = []
    for i in range(2):
        zb = i*sub_h
        B += [box(spacing/2, 0, zb, tau=tau_small),
              box(-spacing/2, 0, zb)]
    return B


def footprint(b):
    """Les 4 coins de l'empreinte au sol d'une caisse."""
    a = np.deg2rad(b['yaw'])
    ca, sa = np.cos(a), np.sin(a)
    corners = [(+sub_d/2, +sub_w/2), (+sub_d/2, -sub_w/2),
               (-sub_d/2, -sub_w/2), (-sub_d/2, +sub_w/2)]
    return np.array([(b['x'] + u*ca - v*sa, b['y'] + u*sa + v*ca) for u, v in corners])


def boxes():
    """Caisses de la config. En concert, x = 0 sur la face avant la plus en avant."""
    B = geometry()
    if SCENARIO == 'concert':
        dx = -max(footprint(b)[:, 0].max() for b in B)
        B = [dict(b, x=b['x'] + dx) for b in B]
    return B


def array(z_start=Z_START):
    """Interface commune : sources (L, 3) [m], tau (L,) [s], gain (L,) signe."""
    B = boxes()
    sources = []
    for b in B:
        a = np.deg2rad(b['yaw'])
        shift = sub_d/2 if b['on_face'] else 0.0
        sources.append([b['x'] + shift*np.cos(a), b['y'] + shift*np.sin(a), z_start + b['z'] + sub_h/2])
    sources = np.array(sources)
    tau = np.array([b['tau'] for b in B])
    gain = np.array([b['gain'] for b in B])
    return sources, tau, gain

# ============================================================
# 3. ACOUSTIC MODEL: MONOPOLES + RIGID GROUND IMAGE
# ============================================================
def monopole_field(A, B, C, xs, ys, zs):
    rd = np.maximum(np.sqrt((A-xs)**2 + (B-ys)**2 + (C-zs)**2), 0.05)
    ri = np.maximum(np.sqrt((A-xs)**2 + (B-ys)**2 + (C+zs)**2), 0.05)
    return 1j*rho*c*k*Q/(4*np.pi)*(np.exp(-1j*k*rd)/rd + np.exp(-1j*k*ri)/ri)


def pressure(A, B, C, z_start):
    sources, tau, gain = array(z_start)
    p = np.zeros(np.broadcast(A, B, C).shape, dtype=complex)
    for (xs, ys, zs), t, g in zip(sources, tau, gain):
        p += g*np.exp(-1j*omega*t)*monopole_field(A, B, C, xs, ys, zs)
    return p


def evaluation_mask():
    excluded = np.zeros_like(X, dtype=bool)
    for b in boxes():
        fp = footprint(b)
        excluded |= ((X >= fp[:, 0].min()-evaluation_margin) & (X <= fp[:, 0].max()+evaluation_margin)
                     & (Y >= fp[:, 1].min()-evaluation_margin) & (Y <= fp[:, 1].max()+evaluation_margin))
    return ~excluded


def spl_maps(z_start):
    p_xy = pressure(X, Y, z_obs, z_start)
    ref = np.max(np.abs(p_xy[evaluation_mask()]))
    xy = 20*np.log10(np.maximum(np.abs(p_xy), 1e-12)/ref)
    xz = 20*np.log10(np.maximum(np.abs(pressure(X_xz, y_slice, Z_xz, z_start)), 1e-12)/ref)
    return np.clip(xy, level_min, level_max), np.clip(xz, level_min, level_max)

# ============================================================
# 4. SUBWOOFER DRAWING HELPERS
# ============================================================
def draw_sub_3d(ax, b, z_start):
    z0 = z_start + b['z']
    z1 = z0 + sub_h
    corners = footprint(b)
    bottom = [[px, py, z0] for px, py in corners]
    top = [[px, py, z1] for px, py in corners]
    faces = [bottom, top] + [[bottom[i], bottom[(i+1)%4], top[(i+1)%4], top[i]] for i in range(4)]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=b['facecolor'], edgecolors='black', linewidths=0.8))


def draw_3d_scene(ax, z_start, title):
    xx = np.array([[-10, 10], [-10, 10]])
    yy = np.array([[-10, -10], [10, 10]])
    ax.plot_surface(xx, yy, np.zeros((2, 2)), color='lightgray', alpha=0.25, linewidth=0)
    ax.plot_surface(xx, yy, np.full((2, 2), z_obs), color='limegreen', alpha=0.2, linewidth=0)
    xs = np.array([[-10, 10], [-10, 10]])
    zz = np.array([[0, 0], [15, 15]])
    ax.plot_surface(xs, np.full((2, 2), y_slice), zz, color='blue', alpha=0.2, linewidth=0)
    for b in boxes():
        draw_sub_3d(ax, b, z_start)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_zlabel(r'$z$ [m]')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 15)
    ax.set_box_aspect((20, 20, 15))
    ax.view_init(elev=22, azim=-58)


def plot_horizontal(ax, SPL, title=None):
    im = ax.contourf(X, Y, SPL, levels=levels, cmap='inferno', extend='neither')
    if SPL.min() <= target <= SPL.max():
        ax.contour(X, Y, SPL, levels=[target], colors='red', linewidths=1, linestyles='solid')
    for b in boxes():
        rect = Rectangle((-sub_d/2, -sub_w/2), sub_d, sub_w, facecolor=b['facecolor'], edgecolor='black', linewidth=1)
        rect.set_transform(Affine2D().rotate_deg(b['yaw']).translate(b['x'], b['y']) + ax.transData)
        ax.add_patch(rect)
    if title:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xticks(np.arange(-50, 51, 10))
    ax.set_yticks(np.arange(-50, 51, 10))
    ax.set_aspect('equal')
    return im


def plot_vertical(ax, SPL, z_start, title=None):
    ax.contourf(X_xz, Z_xz, SPL, levels=levels, cmap='inferno')
    if SPL.min() <= target <= SPL.max():
        ax.contour(X_xz, Z_xz, SPL, levels=[target], colors='red', linewidths=1, linestyles='solid')
    for b in boxes():
        fp = footprint(b)
        if fp[:, 1].min() <= y_slice <= fp[:, 1].max():
            ax.add_patch(Rectangle((fp[:, 0].min(), z_start + b['z']), np.ptp(fp[:, 0]), sub_h,
                                   facecolor=b['facecolor'], edgecolor='black', linewidth=1))
    ax.axhline(0, color='black', linewidth=1)
    ax.axhline(z_obs, color='limegreen', linewidth=1, alpha=0.5)
    if title:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$z$ [m]')
    ax.set_xlim(-50, 50)
    ax.set_ylim(0, 50)
    ax.set_xticks(np.arange(-50, 51, 10))
    ax.set_yticks(np.arange(0, 51, 10))

# ============================================================
# 5. FIGURE: TWO HEIGHTS
# ============================================================
def simulate_configuration():
    spl_ground_xy, spl_ground_xz = spl_maps(z_ground)
    spl_flown_xy, spl_flown_xz = spl_maps(z_flown)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(NAME, fontsize=15, fontweight='bold')
    gs = fig.add_gridspec(2, 4, width_ratios=[1.1, 1, 1, 0.05], left=0.05, right=0.94, bottom=0.08, top=0.90, wspace=0.32, hspace=0.35)
    ax3d_g = fig.add_subplot(gs[0, 0], projection='3d')
    axxy_g = fig.add_subplot(gs[0, 1])
    axxz_g = fig.add_subplot(gs[0, 2])
    ax3d_f = fig.add_subplot(gs[1, 0], projection='3d')
    axxy_f = fig.add_subplot(gs[1, 1])
    axxz_f = fig.add_subplot(gs[1, 2])
    cax = fig.add_subplot(gs[:, 3])

    draw_3d_scene(ax3d_g, z_ground, rf'Ground array ($z={z_ground:g}$ m)')
    im = plot_horizontal(axxy_g, spl_ground_xy, 'Horizontal plan')
    plot_vertical(axxz_g, spl_ground_xz, z_ground, 'Vertical plan (XZ)')
    draw_3d_scene(ax3d_f, z_flown, rf'Flown array ($z={z_flown:g}$ m)')
    plot_horizontal(axxy_f, spl_flown_xy)
    plot_vertical(axxz_f, spl_flown_xz, z_flown)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Normalised level [dB]', fontsize=12)
    cbar.ax.axhline(target, color='red', linewidth=1, linestyle='solid')
    return fig

# ============================================================
# 6. RUN CONFIGURATION
# ============================================================
if __name__ == '__main__':
    print('Simulating:', NAME)
    fig = simulate_configuration()
    plt.show()
