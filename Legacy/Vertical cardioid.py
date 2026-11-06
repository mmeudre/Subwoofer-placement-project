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
z_yz = np.linspace(z_min, z_max, 500)
Y_yz, Z_yz = np.meshgrid(y, z_yz)
x_slice = 0

# Vertical cardioid geometry
N_sub = 3
sub_w = 1.0
sub_d = 0.8
sub_h = 0.6
x_sub = 0
y_sub = 0
z_start_1 = 0
z_start_2 = 10
z_subs_1 = z_start_1 + sub_h/2 + np.arange(N_sub)*sub_h
z_subs_2 = z_start_2 + sub_h/2 + np.arange(N_sub)*sub_h

# Acoustic centres and orientation
z_front_1 = np.array([z_subs_1[0], z_subs_1[2]])
z_front_2 = np.array([z_subs_2[0], z_subs_2[2]])
y_front = -sub_d/2
z_rear_1 = z_subs_1[1]
z_rear_2 = z_subs_2[1]
y_rear = sub_d/2

# Cardioid processing
cardioid_distance = sub_d
delay = cardioid_distance/c
delay_ms = delay*1000
rear_gain = 2.0
rear_phase = -np.exp(-1j*2*np.pi*f*delay)

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


def monopole_yz_view(Y, Z, x_slice, xs, ys, zs):
    r_direct = np.sqrt((x_slice-xs)**2 + (Y-ys)**2 + (Z-zs)**2)
    r_image = np.sqrt((x_slice-xs)**2 + (Y-ys)**2 + (Z+zs)**2)
    r_direct = np.maximum(r_direct, 0.05)
    r_image = np.maximum(r_image, 0.05)
    p_direct = 1j*rho*c*k*Q*np.exp(-1j*k*r_direct)/(4*np.pi*r_direct)
    p_image = 1j*rho*c*k*Q*np.exp(-1j*k*r_image)/(4*np.pi*r_image)
    return p_direct + p_image


def vertical_cardioid_top_view(X, Y, z_obs, z_front, z_rear):
    p = np.zeros_like(X, dtype=complex)
    for zs in z_front:
        p += monopole_top_view(X, Y, z_obs, x_sub, y_front, zs)
    p += rear_gain*rear_phase*monopole_top_view(X, Y, z_obs, x_sub, y_rear, z_rear)
    return p


def vertical_cardioid_yz_view(Y, Z, x_slice, z_front, z_rear):
    p = np.zeros_like(Y, dtype=complex)
    for zs in z_front:
        p += monopole_yz_view(Y, Z, x_slice, x_sub, y_front, zs)
    p += rear_gain*rear_phase*monopole_yz_view(Y, Z, x_slice, x_sub, y_rear, z_rear)
    return p


# ============================================================
# EVALUATION AREA
# ============================================================

source_zone = (np.abs(X-x_sub) <= sub_w/2 + evaluation_margin) & (np.abs(Y-y_sub) <= sub_d/2 + evaluation_margin)
evaluation_zone = ~source_zone


# ============================================================
# PRESSURE FIELDS
# ============================================================

p_1_xy = vertical_cardioid_top_view(X, Y, z_obs, z_front_1, z_rear_1)
p_2_xy = vertical_cardioid_top_view(X, Y, z_obs, z_front_2, z_rear_2)

ref_1 = np.max(np.abs(p_1_xy[evaluation_zone]))
ref_2 = np.max(np.abs(p_2_xy[evaluation_zone]))

SPL_1_xy_raw = 20*np.log10(np.maximum(np.abs(p_1_xy), 1e-12)/ref_1)
SPL_2_xy_raw = 20*np.log10(np.maximum(np.abs(p_2_xy), 1e-12)/ref_2)

gradient_1 = np.max(SPL_1_xy_raw[evaluation_zone]) - np.min(SPL_1_xy_raw[evaluation_zone])
gradient_2 = np.max(SPL_2_xy_raw[evaluation_zone]) - np.min(SPL_2_xy_raw[evaluation_zone])

SPL_1_xy = np.clip(SPL_1_xy_raw, level_min, level_max)
SPL_2_xy = np.clip(SPL_2_xy_raw, level_min, level_max)

p_1_yz = vertical_cardioid_yz_view(Y_yz, Z_yz, x_slice, z_front_1, z_rear_1)
p_2_yz = vertical_cardioid_yz_view(Y_yz, Z_yz, x_slice, z_front_2, z_rear_2)

SPL_1_yz = np.clip(20*np.log10(np.maximum(np.abs(p_1_yz), 1e-12)/ref_1), level_min, level_max)
SPL_2_yz = np.clip(20*np.log10(np.maximum(np.abs(p_2_yz), 1e-12)/ref_2), level_min, level_max)


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def config_title(z_start):
    return rf'{N_sub} subwoofers in a vertical cardioid from $z = {z_start:g}\ \mathrm{{m}}$'


def draw_sub_box(ax, z_bottom):
    ax.bar3d(x_sub-sub_w/2, y_sub-sub_d/2, z_bottom, sub_w, sub_d, sub_h, color='white', edgecolor='black', shade=False)


def draw_3d_scene(ax, z_start):
    xx = np.array([[-5, 5], [-5, 5]])
    yy = np.array([[-5, -5], [5, 5]])
    zz_ground = np.zeros((2, 2))
    zz_obs = z_obs*np.ones((2, 2))
    xx_slice = x_slice*np.ones((2, 2))
    yy_slice = np.array([[-5, 5], [-5, 5]])
    zz_slice = np.array([[z_min, z_min], [z_max, z_max]])

    ax.plot_surface(xx, yy, zz_ground, color='lightgray', alpha=0.25, linewidth=0)
    ax.plot_surface(xx, yy, zz_obs, color='limegreen', alpha=0.2, linewidth=0)
    ax.plot_surface(xx_slice, yy_slice, zz_slice, color='blue', alpha=0.2, linewidth=0)

    for i in range(N_sub):
        draw_sub_box(ax, z_start+i*sub_h)

    z_bottom = z_start + sub_h/2
    z_middle = z_start + 1.5*sub_h
    z_top = z_start + 2.5*sub_h
    ax.quiver(x_sub, -sub_d/2, z_bottom, 0, -1, 0, length=1.0, normalize=True, color='black', linewidth=2.5, arrow_length_ratio=0.25)
    ax.quiver(x_sub, sub_d/2, z_middle, 0, 1, 0, length=1.0, normalize=True, color='black', linewidth=2.5, arrow_length_ratio=0.25)
    ax.quiver(x_sub, -sub_d/2, z_top, 0, -1, 0, length=1.0, normalize=True, color='black', linewidth=2.5, arrow_length_ratio=0.25)

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


def plot_vertical(ax, SPL, z_start, z_subs, title=None):
    ax.contourf(Y_yz, Z_yz, SPL, levels=levels, cmap='inferno')
    ax.contour(Y_yz, Z_yz, SPL, levels=[target], colors='red', linewidths=1, linestyles='solid')
    for i in range(N_sub):
        ax.add_patch(Rectangle((-sub_d/2, z_start+i*sub_h), sub_d, sub_h, facecolor='white', edgecolor='black', linewidth=1.5))
    ax.arrow(0, z_subs[0], -0.55, 0, width=0.02, head_width=0.13, head_length=0.15, color='black', length_includes_head=True)
    ax.arrow(0, z_subs[2], -0.55, 0, width=0.02, head_width=0.13, head_length=0.15, color='black', length_includes_head=True)
    ax.arrow(0, z_subs[1], 0.55, 0, width=0.02, head_width=0.13, head_length=0.15, color='black', length_includes_head=True)
    ax.axhline(0, color='black', linewidth=1)
    ax.axhline(z_obs, color='limegreen', linewidth=1, alpha=0.5)
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel(r'$y$ [m]')
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
axyz_1 = fig.add_subplot(gs[0, 2])
ax3d_2 = fig.add_subplot(gs[1, 0], projection='3d')
axxy_2 = fig.add_subplot(gs[1, 1])
axyz_2 = fig.add_subplot(gs[1, 2])
cax = fig.add_subplot(gs[:, 3])

draw_3d_scene(ax3d_1, z_start_1)
im = plot_horizontal(axxy_1, SPL_1_xy, r'Horizontal plan')
plot_vertical(axyz_1, SPL_1_yz, z_start_1, z_subs_1, r'Vertical plan')

draw_3d_scene(ax3d_2, z_start_2)
plot_horizontal(axxy_2, SPL_2_xy)
plot_vertical(axyz_2, SPL_2_yz, z_start_2, z_subs_2)


# ============================================================
# INFORMATION BOXES
# ============================================================

gradient_box = r'$\Delta L = L_{\mathrm{max}} - L_{\mathrm{min}}$' + '\n' + rf'$z = {z_start_1:g}\ \mathrm{{m}}$ : $\Delta L = {gradient_1:.2f}\ \mathrm{{dB}}$' + '\n' + rf'$z = {z_start_2:g}\ \mathrm{{m}}$ : $\Delta L = {gradient_2:.2f}\ \mathrm{{dB}}$'
fig.text(0.935, 0.05, gradient_box, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

cardioid_box = r'Sub of the centre, polarity reversed with ' + rf'$\tau = {delay_ms:.2f}\ \mathrm{{ms}}$'
fig.text(0.47, 0.05, cardioid_box, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

green_line = Line2D([0], [0], color='limegreen', linewidth=1, alpha=0.5)
fig.legend([green_line], [rf'Observation plane : $z = {z_obs:g}\ \mathrm{{m}}$'], loc='lower center', bbox_to_anchor=(0.74, 0.025), frameon=True, edgecolor='black', fancybox=True)


# ============================================================
# COLORBAR
# ============================================================

cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r'Normalised level [dB]', fontsize=12)
cbar.ax.axhline(target, color='red', linewidth=1, linestyle='solid')

plt.show()
