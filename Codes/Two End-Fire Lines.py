import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

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
omega = 2*np.pi*f

# Observation area
z_obs = 1.6
x = np.linspace(-50, 50, 500)
y = np.linspace(-50, 50, 500)
X, Y = np.meshgrid(x, y)

# Vertical section YZ
z_min = 0
z_max = 50
z_yz = np.linspace(z_min, z_max, 500)
Y_yz, Z_yz = np.meshgrid(y, z_yz)
x_slice = 0

# 3D display limits
x_3d_min = -10
x_3d_max = 10
y_3d_min = -10
y_3d_max = 10
z_3d_min = 0
z_3d_max = 15

# Source heights
z_ground = 0
z_flown = 8

# Subwoofer dimensions
sub_w = 1.0
sub_h = 0.8
sub_z = 0.6

# Decoupled end-fire configuration
N_arrays = 2
N_deep = 3

d = 0.8
tau = d/c
tau_ms = tau*1000

# Two separated arrays
x_arrays = np.array([-10.0, 10.0])

# End-fire depth along y
# 0 -> tau -> 2*tau from rear to front
# Main radiation toward -y
y_deep = np.array([d, 0.0, -d])

# Evaluation and display
evaluation_margin = 0.5
target = -3
level_min = -20
level_max = 0
levels = np.linspace(level_min, level_max, 21)


# ============================================================
# ARRAY GEOMETRY
# ============================================================

sources = []

for x_array in x_arrays:
    for i, y_source in enumerate(y_deep):
        sources.append({'x': x_array, 'y': y_source, 'delay': i*tau})

N_sub = len(sources)


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


def monopole_yz_view(Y, Z, x_slice, xs, ys, zs):
    r_direct = np.sqrt((x_slice-xs)**2 + (Y-ys)**2 + (Z-zs)**2)
    r_image = np.sqrt((x_slice-xs)**2 + (Y-ys)**2 + (Z+zs)**2)

    r_direct = np.maximum(r_direct, 0.05)
    r_image = np.maximum(r_image, 0.05)

    p_direct = 1j*rho*c*k*Q*np.exp(-1j*k*r_direct)/(4*np.pi*r_direct)
    p_image = 1j*rho*c*k*Q*np.exp(-1j*k*r_image)/(4*np.pi*r_image)

    return p_direct + p_image


def decoupled_endfire_top_view(X, Y, z_array):
    p = np.zeros_like(X, dtype=complex)

    for src in sources:
        weight = np.exp(-1j*omega*src['delay'])
        p += weight*monopole_top_view(X, Y, z_obs, src['x'], src['y'], z_array)

    return p


def decoupled_endfire_yz_view(Y, Z, z_array):
    p = np.zeros_like(Y, dtype=complex)

    for src in sources:
        weight = np.exp(-1j*omega*src['delay'])
        p += weight*monopole_yz_view(Y, Z, x_slice, src['x'], src['y'], z_array)

    return p


# ============================================================
# LABELS
# ============================================================

def config_title(z):
    return rf'{N_sub} subwoofers in 2 decoupled end-fires, $d = {d:g}$ m from $z = {z:g}$ m'


def config_label(z):
    return rf'$z = {z:g}\ \mathrm{{m}}$'


# ============================================================
# EVALUATION AREA
# ============================================================

source_zone = np.zeros_like(X, dtype=bool)

for src in sources:
    source_zone |= (np.abs(X-src['x']) <= sub_w/2 + evaluation_margin) & (np.abs(Y-src['y']) <= sub_h/2 + evaluation_margin)

evaluation_zone = ~source_zone


# ============================================================
# PRESSURE FIELDS
# ============================================================

p_ground_xy = decoupled_endfire_top_view(X, Y, z_ground)
p_flown_xy = decoupled_endfire_top_view(X, Y, z_flown)

ref_ground = np.max(np.abs(p_ground_xy[evaluation_zone]))
ref_flown = np.max(np.abs(p_flown_xy[evaluation_zone]))

SPL_ground_xy_raw = 20*np.log10(np.maximum(np.abs(p_ground_xy), 1e-12)/ref_ground)
SPL_flown_xy_raw = 20*np.log10(np.maximum(np.abs(p_flown_xy), 1e-12)/ref_flown)

gradient_ground = np.max(SPL_ground_xy_raw[evaluation_zone]) - np.min(SPL_ground_xy_raw[evaluation_zone])
gradient_flown = np.max(SPL_flown_xy_raw[evaluation_zone]) - np.min(SPL_flown_xy_raw[evaluation_zone])

SPL_ground_xy = np.clip(SPL_ground_xy_raw, level_min, level_max)
SPL_flown_xy = np.clip(SPL_flown_xy_raw, level_min, level_max)

p_ground_yz = decoupled_endfire_yz_view(Y_yz, Z_yz, z_ground)
p_flown_yz = decoupled_endfire_yz_view(Y_yz, Z_yz, z_flown)

SPL_ground_yz_raw = 20*np.log10(np.maximum(np.abs(p_ground_yz), 1e-12)/ref_ground)
SPL_flown_yz_raw = 20*np.log10(np.maximum(np.abs(p_flown_yz), 1e-12)/ref_flown)

SPL_ground_yz = np.clip(SPL_ground_yz_raw, level_min, level_max)
SPL_flown_yz = np.clip(SPL_flown_yz_raw, level_min, level_max)


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def draw_sub_box(ax, xs, ys, z_source):
    z_box = max(0, z_source-sub_z/2)
    ax.bar3d(xs-sub_w/2, ys-sub_h/2, z_box, sub_w, sub_h, sub_z, color='white', edgecolor='black', shade=False)


def draw_3d_scene(ax, z_source):
    xx = np.array([[x_3d_min, x_3d_max], [x_3d_min, x_3d_max]])
    yy = np.array([[y_3d_min, y_3d_min], [y_3d_max, y_3d_max]])

    zz_ground = np.zeros((2, 2))
    zz_obs = z_obs*np.ones((2, 2))

    yy_slice = np.array([[y_3d_min, y_3d_max], [y_3d_min, y_3d_max]])
    xx_slice = x_slice*np.ones((2, 2))
    zz_slice = np.array([[z_3d_min, z_3d_min], [z_3d_max, z_3d_max]])

    ax.plot_surface(xx, yy, zz_ground, color='lightgray', alpha=0.25, linewidth=0)
    ax.plot_surface(xx, yy, zz_obs, color='limegreen', alpha=0.2, linewidth=0)
    ax.plot_surface(xx_slice, yy_slice, zz_slice, color='blue', alpha=0.2, linewidth=0)

    for src in sources:
        draw_sub_box(ax, src['x'], src['y'], z_source)

    ax.set_title(config_title(z_source), fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_zlabel(r'$z$ [m]')

    ax.set_xlim(x_3d_min, x_3d_max)
    ax.set_ylim(y_3d_min, y_3d_max)
    ax.set_zlim(z_3d_min, z_3d_max)

    ax.set_box_aspect((20, 20, 15))
    ax.view_init(elev=22, azim=-58)


def plot_horizontal(ax, SPL, title=None):
    im = ax.contourf(X, Y, SPL, levels=levels, cmap='inferno')
    ax.contour(X, Y, SPL, levels=[target], colors='red', linewidths=1, linestyles='solid')

    for src in sources:
        ax.add_patch(Rectangle((src['x']-sub_w/2, src['y']-sub_h/2), sub_w, sub_h, facecolor='white', edgecolor='black', linewidth=1.5))

    if title is not None:
        ax.set_title(title, fontsize=12)

    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')

    return im


def plot_vertical(ax, SPL, title=None):
    ax.contourf(Y_yz, Z_yz, SPL, levels=levels, cmap='inferno')
    ax.contour(Y_yz, Z_yz, SPL, levels=[target], colors='red', linewidths=1, linestyles='solid')

    ax.axhline(0, color='black', linewidth=1)
    ax.axhline(z_obs, color='limegreen', linewidth=1, alpha=0.5)

    if title is not None:
        ax.set_title(title, fontsize=12)

    ax.set_xlabel(r'$y$ [m]')
    ax.set_ylabel(r'$z$ [m]')
    ax.set_xlim(-50, 50)
    ax.set_ylim(0, 50)


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 4, width_ratios=[1.1, 1, 1, 0.05], left=0.06, right=0.92, bottom=0.16, top=0.93, wspace=0.28, hspace=0.28)

ax3d_ground = fig.add_subplot(gs[0, 0], projection='3d')
axxy_ground = fig.add_subplot(gs[0, 1])
ayz_ground = fig.add_subplot(gs[0, 2])

ax3d_flown = fig.add_subplot(gs[1, 0], projection='3d')
axxy_flown = fig.add_subplot(gs[1, 1])
ayz_flown = fig.add_subplot(gs[1, 2])

cax = fig.add_subplot(gs[:, 3])

draw_3d_scene(ax3d_ground, z_ground)
im = plot_horizontal(axxy_ground, SPL_ground_xy, r'Horizontal plan')
plot_vertical(ayz_ground, SPL_ground_yz, r'Vertical plan')

draw_3d_scene(ax3d_flown, z_flown)
plot_horizontal(axxy_flown, SPL_flown_xy)
plot_vertical(ayz_flown, SPL_flown_yz)


# ============================================================
# INFORMATION BOXES
# ============================================================

gradient_box = r'$\Delta L = L_{\mathrm{max}} - L_{\mathrm{min}}$' + '\n' + config_label(z_ground) + rf' : $\Delta L = {gradient_ground:.2f}\ \mathrm{{dB}}$' + '\n' + config_label(z_flown) + rf' : $\Delta L = {gradient_flown:.2f}\ \mathrm{{dB}}$'
fig.text(0.935, 0.05, gradient_box, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

endfire_box = rf'Two arrays: $1$ row $\times$ $3$ deep each, at $x = \pm 10\ \mathrm{{m}}$;' + '\n' + rf'end-fire spacing $d = {d:.2f}\ \mathrm{{m}}$, $\tau = {tau_ms:.2f}\ \mathrm{{ms}}$ per step;' + '\n' + rf'delays: $0,\ {tau_ms:.2f},\ {2*tau_ms:.2f}\ \mathrm{{ms}}$; main radiation toward $-y$' + '\n' + rf'maps computed at $f = {f:g}\ \mathrm{{Hz}}$'
fig.text(0.47, 0.05, endfire_box, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

green_line = Line2D([0], [0], color='limegreen', linewidth=1, alpha=0.5)
blue_line = Line2D([0], [0], color='blue', linewidth=1, alpha=0.5)
fig.legend([green_line, blue_line], [rf'Observation plane : $z = {z_obs:g}\ \mathrm{{m}}$', rf'Slice plane : $x = {x_slice:g}\ \mathrm{{m}}$'], loc='lower center', bbox_to_anchor=(0.74, 0.025), frameon=True, edgecolor='black', fancybox=True)


# ============================================================
# COLORBAR
# ============================================================

cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r'Normalised level [dB]', fontsize=12)
cbar.ax.axhline(target, color='red', linewidth=1, linestyle='solid')

plt.show()