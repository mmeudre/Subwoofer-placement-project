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

# Observation area
z_obs = 1.5
x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)

# Vertical section
z_min = 0
z_max = 14
z_xz = np.linspace(z_min, z_max, 500)
X_xz, Z_xz = np.meshgrid(x, z_xz)
y_slice = 0

# Vertical stack geometry
N_sub = 5
sub_w = 1.0
sub_d = 0.8
sub_h = 0.6
x_sub = 0
y_sub = 0
z_start_1 = 0
z_start_2 = 10
z_subs_1 = z_start_1 + sub_h/2 + np.arange(N_sub)*sub_h
z_subs_2 = z_start_2 + sub_h/2 + np.arange(N_sub)*sub_h

# Evaluation and display
evaluation_margin = 0.5
target = -3
level_min = -20
level_max = 0
levels = np.linspace(level_min, level_max, 21)


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


def monopole_xz_view(X, Z, y_slice, xs, ys, zs):
    r_direct = np.sqrt((X-xs)**2 + (y_slice-ys)**2 + (Z-zs)**2)
    r_image = np.sqrt((X-xs)**2 + (y_slice-ys)**2 + (Z+zs)**2)
    r_direct = np.maximum(r_direct, 0.05)
    r_image = np.maximum(r_image, 0.05)
    p_direct = 1j*rho*c*k*Q*np.exp(-1j*k*r_direct)/(4*np.pi*r_direct)
    p_image = 1j*rho*c*k*Q*np.exp(-1j*k*r_image)/(4*np.pi*r_image)
    return p_direct + p_image


def vertical_stack_top_view(X, Y, z_obs, z_subs):
    p = np.zeros_like(X, dtype=complex)
    for zs in z_subs:
        p += monopole_top_view(X, Y, z_obs, x_sub, y_sub, zs)
    return p


def vertical_stack_xz_view(X, Z, y_slice, z_subs):
    p = np.zeros_like(X, dtype=complex)
    for zs in z_subs:
        p += monopole_xz_view(X, Z, y_slice, x_sub, y_sub, zs)
    return p


# ============================================================
# EVALUATION AREA
# ============================================================

source_zone = (np.abs(X-x_sub) <= sub_w/2 + evaluation_margin) & (np.abs(Y-y_sub) <= sub_d/2 + evaluation_margin)
evaluation_zone = ~source_zone


# ============================================================
# PRESSURE FIELDS
# ============================================================

p_1_xy = vertical_stack_top_view(X, Y, z_obs, z_subs_1)
p_2_xy = vertical_stack_top_view(X, Y, z_obs, z_subs_2)

ref_1 = np.max(np.abs(p_1_xy[evaluation_zone]))
ref_2 = np.max(np.abs(p_2_xy[evaluation_zone]))

SPL_1_xy_raw = 20*np.log10(np.maximum(np.abs(p_1_xy), 1e-12)/ref_1)
SPL_2_xy_raw = 20*np.log10(np.maximum(np.abs(p_2_xy), 1e-12)/ref_2)

gradient_1 = np.max(SPL_1_xy_raw[evaluation_zone]) - np.min(SPL_1_xy_raw[evaluation_zone])
gradient_2 = np.max(SPL_2_xy_raw[evaluation_zone]) - np.min(SPL_2_xy_raw[evaluation_zone])

SPL_1_xy = np.clip(SPL_1_xy_raw, level_min, level_max)
SPL_2_xy = np.clip(SPL_2_xy_raw, level_min, level_max)

p_1_xz = vertical_stack_xz_view(X_xz, Z_xz, y_slice, z_subs_1)
p_2_xz = vertical_stack_xz_view(X_xz, Z_xz, y_slice, z_subs_2)

SPL_1_xz = np.clip(20*np.log10(np.maximum(np.abs(p_1_xz), 1e-12)/ref_1), level_min, level_max)
SPL_2_xz = np.clip(20*np.log10(np.maximum(np.abs(p_2_xz), 1e-12)/ref_2), level_min, level_max)


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def config_title(z_start):
    return rf'{N_sub} stacked subwoofers vertically from $z = {z_start:g}\ \mathrm{{m}}$'


def draw_sub_box(ax, z_bottom):
    ax.bar3d(x_sub-sub_w/2, y_sub-sub_d/2, z_bottom, sub_w, sub_d, sub_h, color='white', edgecolor='black', shade=False)


def draw_3d_scene(ax, z_start):
    xx = np.array([[-5, 5], [-5, 5]])
    yy = np.array([[-5, -5], [5, 5]])
    zz_ground = np.zeros((2, 2))
    zz_obs = z_obs*np.ones((2, 2))
    xx_slice = np.array([[-5, 5], [-5, 5]])
    yy_slice = y_slice*np.ones((2, 2))
    zz_slice = np.array([[z_min, z_min], [z_max, z_max]])

    ax.plot_surface(xx, yy, zz_ground, color='lightgray', alpha=0.25, linewidth=0)
    ax.plot_surface(xx, yy, zz_obs, color='limegreen', alpha=0.2, linewidth=0)
    ax.plot_surface(xx_slice, yy_slice, zz_slice, color='blue', alpha=0.2, linewidth=0)

    for i in range(N_sub):
        draw_sub_box(ax, z_start+i*sub_h)

    ax.set_title(config_title(z_start), fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_zlabel(r'$z$ [m]')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect((10, 10, 14))
    ax.view_init(elev=22, azim=-58)


def plot_horizontal(ax, SPL, title=None):
    im = ax.contourf(X, Y, SPL, levels=levels, cmap='inferno')
    ax.contour(X, Y, SPL, levels=[target], colors='red', linewidths=1, linestyles='solid')
    ax.add_patch(Rectangle((x_sub-sub_w/2, y_sub-sub_d/2), sub_w, sub_d, facecolor='white', edgecolor='black', linewidth=1.5))
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    return im


def plot_vertical(ax, SPL, z_start, title=None):
    ax.contourf(X_xz, Z_xz, SPL, levels=levels, cmap='inferno')
    ax.contour(X_xz, Z_xz, SPL, levels=[target], colors='red', linewidths=1, linestyles='solid')
    for i in range(N_sub):
        ax.add_patch(Rectangle((x_sub-sub_w/2, z_start+i*sub_h), sub_w, sub_h, facecolor='white', edgecolor='black', linewidth=1.5))
    ax.axhline(0, color='black', linewidth=1)
    ax.axhline(z_obs, color='limegreen', linewidth=1, alpha=0.5)
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$z$ [m]')
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 14)


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 4, width_ratios=[1.1, 1, 1, 0.05], left=0.06, right=0.92, bottom=0.16, top=0.93, wspace=0.28, hspace=0.28)

ax3d_1 = fig.add_subplot(gs[0, 0], projection='3d')
axxy_1 = fig.add_subplot(gs[0, 1])
axxz_1 = fig.add_subplot(gs[0, 2])
ax3d_2 = fig.add_subplot(gs[1, 0], projection='3d')
axxy_2 = fig.add_subplot(gs[1, 1])
axxz_2 = fig.add_subplot(gs[1, 2])
cax = fig.add_subplot(gs[:, 3])

draw_3d_scene(ax3d_1, z_start_1)
im = plot_horizontal(axxy_1, SPL_1_xy, r'Horizontal plan')
plot_vertical(axxz_1, SPL_1_xz, z_start_1, r'Vertical plan')

draw_3d_scene(ax3d_2, z_start_2)
plot_horizontal(axxy_2, SPL_2_xy)
plot_vertical(axxz_2, SPL_2_xz, z_start_2)


# ============================================================
# INFORMATION BOXES
# ============================================================

gradient_box = r'$\Delta L = L_{\mathrm{max}} - L_{\mathrm{min}}$' + '\n' + rf'$z = {z_start_1:g}\ \mathrm{{m}}$ : $\Delta L = {gradient_1:.2f}\ \mathrm{{dB}}$' + '\n' + rf'$z = {z_start_2:g}\ \mathrm{{m}}$ : $\Delta L = {gradient_2:.2f}\ \mathrm{{dB}}$'
fig.text(0.935, 0.05, gradient_box, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

green_line = Line2D([0], [0], color='limegreen', linewidth=1, alpha=0.5)
fig.legend([green_line], [rf'Observation plane : $z = {z_obs:g}\ \mathrm{{m}}$'], loc='lower center', bbox_to_anchor=(0.74, 0.025), frameon=True, edgecolor='black', fancybox=True)


# ============================================================
# COLORBAR
# ============================================================

cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r'Normalised level [dB]', fontsize=12)
cbar.ax.axhline(target, color='red', linewidth=1, linestyle='solid')

plt.show()