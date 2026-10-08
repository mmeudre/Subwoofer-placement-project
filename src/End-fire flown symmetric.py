import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.close('all')


# ============================================================
# PARAMETERS
# ============================================================

# Acoustic parameters
c = 343
rho = 1.2
f = 80
Q = 1e-3
k = 2*np.pi*f/c

# Observation area
z_obs = 1.6
x = np.linspace(-40, 55, 400)
y = np.linspace(-30, 30, 400)
X, Y = np.meshgrid(x, y)

# Vertical section
z_min = 0
z_max = 10
y_slice = 0

# Subwoofer dimensions
sub_w = 1.0
sub_d = 0.8
sub_h = 0.6

# Array height
z_flown = 8

# End-fire arrays (two, symmetric about y = 0 and decoupled; one spacing per figure row)
n_rows = 2          # lignes cote a cote
n_deep = 4          # caisses en profondeur
spacings = [1.2, 0.8]   # pas end-fire [m]
lr_offset = 10      # position en y des deux arrays symetriques [m]

# Evaluation and display
evaluation_margin = 0.5
target = -3
level_min = -40
level_max = 0
levels = np.linspace(level_min, level_max, 21)


# ============================================================
# ARRAY GEOMETRY
# ============================================================

def endfire_rows(spacing, dy=0.0):
    """Lignes end-fire paralleles tirant vers +x. Une caisse = (x, y, yaw_deg, tau)."""
    boxes = []
    for o in (np.arange(n_rows) - (n_rows-1)/2)*sub_w:
        for i in range(n_deep):
            boxes.append(((i - (n_deep-1))*spacing, o + dy, 0.0, i*spacing/c))
    return boxes


def build_boxes(spacing):
    return endfire_rows(spacing, dy=+lr_offset) + endfire_rows(spacing, dy=-lr_offset)


def source_positions(boxes, z_start):
    """Centre acoustique de chaque caisse : centre geometrique avance de
    sub_d/2 le long de l'axe de tir."""
    return [(xb + sub_d/2, yb, z_start + sub_h/2, tau)
            for xb, yb, yaw, tau in boxes]


def footprint(xb, yb, yaw):
    """Les 4 coins de l'empreinte au sol d'une caisse."""
    a = np.deg2rad(yaw)
    ca, sa = np.cos(a), np.sin(a)
    corners = [(+sub_d/2, +sub_w/2), (+sub_d/2, -sub_w/2),
               (-sub_d/2, -sub_w/2), (-sub_d/2, +sub_w/2)]
    return [(xb + u*ca - v*sa, yb + u*sa + v*ca) for u, v in corners]


# ============================================================
# ACOUSTIC MODEL
# ============================================================

def monopole_top_view(X, Y, z_obs, xs, ys, zs):
    r_direct = np.sqrt((X-xs)**2 + (Y-ys)**2 + (z_obs-zs)**2)
    r_image = np.sqrt((X-xs)**2 + (Y-ys)**2 + (z_obs+zs)**2)

    r_direct = np.maximum(r_direct, 0.05)
    r_image = np.maximum(r_image, 0.05)

    p_direct = 1j*rho*c*k*Q*np.exp(-1j*k*r_direct)/(4*np.pi*r_direct)
    p_image = 1j*rho*c*k*Q*np.exp(-1j*k*r_image)/(4*np.pi*r_image)

    return p_direct + p_image


def array_top_view(boxes, X, Y, z_obs, z_start):
    p = np.zeros_like(X, dtype=complex)
    for xs, ys, zs, tau in source_positions(boxes, z_start):
        p += np.exp(-1j*2*np.pi*f*tau)*monopole_top_view(X, Y, z_obs, xs, ys, zs)
    return p


# ============================================================
# CONFIGURATIONS AND PRESSURE FIELDS
# ============================================================

configs = [dict(boxes=build_boxes(d), spacing=d) for d in spacings]
N_sub = len(configs[0]['boxes'])

for cfg in configs:
    boxes = cfg['boxes']

    source_zone = np.zeros_like(X, dtype=bool)
    for xb, yb, yaw, tau in boxes:
        source_zone |= ((np.abs(X-xb) <= sub_d/2 + evaluation_margin)
                        & (np.abs(Y-yb) <= sub_w/2 + evaluation_margin))
    evaluation_zone = ~source_zone

    p = array_top_view(boxes, X, Y, z_obs, z_flown)
    ref = np.max(np.abs(p[evaluation_zone]))
    SPL_raw = 20*np.log10(np.maximum(np.abs(p), 1e-12)/ref)

    cfg['gradient'] = (np.max(SPL_raw[evaluation_zone])
                       - np.min(SPL_raw[evaluation_zone]))
    cfg['SPL'] = np.clip(SPL_raw, level_min, level_max)


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def config_title(cfg):
    return (rf'{N_sub} subwoofers in 2 decoupled end-fires, '
            rf'$d = {cfg["spacing"]:.1f}\ \mathrm{{m}}$ from $z = {z_flown:g}\ \mathrm{{m}}$')


def draw_sub_box(ax, xb, yb, yaw, z_bottom):
    base = footprint(xb, yb, yaw)
    top = [(u, v, z_bottom + sub_h) for u, v in base]
    bot = [(u, v, z_bottom) for u, v in base]
    faces = [bot, top]
    for i in range(4):
        j = (i+1) % 4
        faces.append([bot[i], bot[j], top[j], top[i]])
    ax.add_collection3d(Poly3DCollection(faces, facecolor='white',
                                         edgecolor='black', linewidths=0.6))


def draw_3d_scene(ax, cfg, z_start):
    lim = 14
    xx = np.array([[-lim, lim], [-lim, lim]])
    yy = np.array([[-lim, -lim], [lim, lim]])
    zz_ground = np.zeros((2, 2))
    zz_obs = z_obs*np.ones((2, 2))

    xx_slice = np.array([[-lim, lim], [-lim, lim]])
    yy_slice = y_slice*np.ones((2, 2))
    zz_slice = np.array([[z_min, z_min], [z_max, z_max]])

    ax.plot_surface(xx, yy, zz_ground, color='lightgray', alpha=0.25, linewidth=0)
    ax.plot_surface(xx, yy, zz_obs, color='limegreen', alpha=0.2, linewidth=0)
    ax.plot_surface(xx_slice, yy_slice, zz_slice, color='blue', alpha=0.2, linewidth=0)

    for xb, yb, yaw, tau in cfg['boxes']:
        draw_sub_box(ax, xb, yb, yaw, z_start)

    ax.set_title(config_title(cfg), fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_zlabel(r'$z$ [m]')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect((28, 28, 10))
    ax.view_init(elev=22, azim=-58)


def plot_horizontal(ax, cfg, title=None):
    im = ax.contourf(X, Y, cfg['SPL'], levels=levels, cmap='viridis')
    ax.contour(X, Y, cfg['SPL'], levels=[target], colors='red', linewidths=1, linestyles='solid')

    for xb, yb, yaw, tau in cfg['boxes']:
        ax.add_patch(Polygon(footprint(xb, yb, yaw), closed=True,
                             facecolor='white', edgecolor='black', linewidth=1.0))

    if title is not None:
        ax.set_title(title, fontsize=12)

    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_aspect('equal')

    return im


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.35, 0.05],
                      left=0.04, right=0.91, bottom=0.21, top=0.93,
                      wspace=0.22, hspace=0.26)

cax = fig.add_subplot(gs[:, 2])

axes_3d = []
axes_xy = []
for row, cfg in enumerate(configs):
    ax3d = fig.add_subplot(gs[row, 0], projection='3d')
    axxy = fig.add_subplot(gs[row, 1])
    draw_3d_scene(ax3d, cfg, z_flown)
    im = plot_horizontal(axxy, cfg, r'Horizontal plane' if row == 0 else None)
    axes_3d.append(ax3d)
    axes_xy.append(axxy)


# ============================================================
# INFORMATION BOXES
# ============================================================

# Each box is centred under the panel it describes
axes_xy[-1].apply_aspect()
x_3d = 0.5 * sum(axes_3d[-1].get_position().intervalx)
x_maps = 0.5 * sum(axes_xy[-1].get_position().intervalx)
x_cbar = 0.5 * sum(cax.get_position().intervalx)
y_box = 0.07

gradient_box = r'$\Delta L = L_{\mathrm{max}} - L_{\mathrm{min}}$'
for cfg in configs:
    gradient_box += '\n' + (rf'$d = {cfg["spacing"]:.1f}\ \mathrm{{m}}$ : '
                            rf'$\Delta L = {cfg["gradient"]:.2f}\ \mathrm{{dB}}$')
fig.text(x_cbar, y_box, gradient_box, ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

delays_ms = ' / '.join(f'{1000*d/c:.2f}' for d in spacings)
array_box = (rf'Two arrays : {n_rows} rows x {n_deep} deep each, at $y = \pm {lr_offset:g}\ \mathrm{{m}}$,' + '\n'
             + rf'flown at $z = {z_flown:g}\ \mathrm{{m}}$, decoupled (far apart);' + '\n'
             + rf'end-fire spacing $d = {" / ".join(f"{d:.2f}" for d in spacings)}\ \mathrm{{m}}$, '
             + rf'$\tau = {delays_ms}\ \mathrm{{ms}}$ per step' + '\n'
             + rf'maps computed at $f = {f:g}\ \mathrm{{Hz}}$')
fig.text(x_maps, y_box, array_box, ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

green_line = Line2D([0], [0], color='limegreen', linewidth=1, alpha=0.5)
blue_line = Line2D([0], [0], color='blue', linewidth=1, alpha=0.5)
fig.legend([green_line, blue_line],
           [rf'Observation plane : $z = {z_obs:g}\ \mathrm{{m}}$',
            rf'Slice plane : $y = {y_slice:g}\ \mathrm{{m}}$'],
           loc='center', bbox_to_anchor=(x_3d, y_box), frameon=True,
           edgecolor='black', fancybox=True)


# ============================================================
# COLORBAR
# ============================================================

cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r'Normalised level [dB]', fontsize=12)
cbar.ax.axhline(target, color='red', linewidth=1, linestyle='solid')
cbar.ax.annotate(rf'${target:g}\ \mathrm{{dB}}$', xy=(0, target),
                 xycoords=('axes fraction', 'data'), xytext=(-4, 0),
                 textcoords='offset points', ha='right', va='center',
                 color='red', fontsize=10)

plt.show()
