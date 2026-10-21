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

# Subwoofer geometry
N_sub = 3
spacing = 1.0
sub_w = 1.0
sub_d = 0.8
sub_h = 0.6
x_subs = np.array([-spacing, 0, spacing])
y_subs = np.zeros(N_sub)
z_ground = 0
z_flown = 10

# Acoustic centres and orientation
x_front = np.array([-spacing, spacing])
y_front = np.array([-sub_d/2, -sub_d/2])
x_rear = 0
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


def cardioid_top_view(X, Y, z_obs, z_array):
    p = np.zeros_like(X, dtype=complex)
    for xs, ys in zip(x_front, y_front):
        p += monopole_top_view(X, Y, z_obs, xs, ys, z_array)
    p += rear_gain*rear_phase*monopole_top_view(X, Y, z_obs, x_rear, y_rear, z_array)
    return p


def cardioid_yz_view(Y, Z, x_slice, z_array):
    p = np.zeros_like(Y, dtype=complex)
    for xs, ys in zip(x_front, y_front):
        p += monopole_yz_view(Y, Z, x_slice, xs, ys, z_array)
    p += rear_gain*rear_phase*monopole_yz_view(Y, Z, x_slice, x_rear, y_rear, z_array)
    return p


# ============================================================
# EVALUATION AREA
# ============================================================

source_zone = np.zeros_like(X, dtype=bool)
for xs, ys in zip(x_subs, y_subs):
    source_zone |= (np.abs(X-xs) <= sub_w/2 + evaluation_margin) & (np.abs(Y-ys) <= sub_d/2 + evaluation_margin)

evaluation_zone = ~source_zone


# ============================================================
# PRESSURE FIELDS
# ============================================================

p_ground_xy = cardioid_top_view(X, Y, z_obs, z_ground)
p_flown_xy = cardioid_top_view(X, Y, z_obs, z_flown)

ref_ground = np.max(np.abs(p_ground_xy[evaluation_zone]))
ref_flown = np.max(np.abs(p_flown_xy[evaluation_zone]))

SPL_ground_xy_raw = 20*np.log10(np.maximum(np.abs(p_ground_xy), 1e-12)/ref_ground)
SPL_flown_xy_raw = 20*np.log10(np.maximum(np.abs(p_flown_xy), 1e-12)/ref_flown)

gradient_ground = np.max(SPL_ground_xy_raw[evaluation_zone]) - np.min(SPL_ground_xy_raw[evaluation_zone])
gradient_flown = np.max(SPL_flown_xy_raw[evaluation_zone]) - np.min(SPL_flown_xy_raw[evaluation_zone])

SPL_ground_xy = np.clip(SPL_ground_xy_raw, level_min, level_max)
SPL_flown_xy = np.clip(SPL_flown_xy_raw, level_min, level_max)

p_ground_yz = cardioid_yz_view(Y_yz, Z_yz, x_slice, z_ground)
p_flown_yz = cardioid_yz_view(Y_yz, Z_yz, x_slice, z_flown)

SPL_ground_yz = np.clip(20*np.log10(np.maximum(np.abs(p_ground_yz), 1e-12)/ref_ground), level_min, level_max)
SPL_flown_yz = np.clip(20*np.log10(np.maximum(np.abs(p_flown_yz), 1e-12)/ref_flown), level_min, level_max)


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def config_title(z):
    return rf'{N_sub} subwoofers in a horizontal cardioid from $z = {z:g}\ \mathrm{{m}}$'


def draw_sub_box(ax, xs, ys, z0):
    ax.bar3d(xs-sub_w/2, ys-sub_d/2, z0, sub_w, sub_d, sub_h, color='white', edgecolor='black', shade=False)


def draw_3d_scene(ax, z_array):
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

    for xs, ys in zip(x_subs, y_subs):
        draw_sub_box(ax, xs, ys, z_array)

    z_arrow = z_array + sub_h/2
    ax.quiver(-spacing, -sub_d/2, z_arrow, 0, -1, 0, length=1.0, normalize=True, color='black', linewidth=2.5, arrow_length_ratio=0.25)
    ax.quiver(spacing, -sub_d/2, z_arrow, 0, -1, 0, length=1.0, normalize=True, color='black', linewidth=2.5, arrow_length_ratio=0.25)
    ax.quiver(0, sub_d/2, z_arrow, 0, 1, 0, length=1.0, normalize=True, color='black', linewidth=2.5, arrow_length_ratio=0.25)

    ax.set_title(config_title(z_array), fontsize=12, fontweight='bold')
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
    for xs, ys in zip(x_subs, y_subs):
        ax.add_patch(Rectangle((xs-sub_w/2, ys-sub_d/2), sub_w, sub_d, facecolor='white', edgecolor='black', linewidth=1.5))
    ax.arrow(-spacing, 0, 0, -0.55, width=0.025, head_width=0.15, head_length=0.15, color='black', length_includes_head=True)
    ax.arrow(spacing, 0, 0, -0.55, width=0.025, head_width=0.15, head_length=0.15, color='black', length_includes_head=True)
    ax.arrow(0, 0, 0, 0.55, width=0.025, head_width=0.15, head_length=0.15, color='black', length_includes_head=True)
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    return im


def plot_vertical(ax, SPL, z_array, title=None):
    ax.contourf(Y_yz, Z_yz, SPL, levels=levels, cmap='inferno')
    ax.contour(Y_yz, Z_yz, SPL, levels=[target], colors='red', linewidths=1, linestyles='solid')
    ax.add_patch(Rectangle((-sub_d/2, z_array), sub_d, sub_h, facecolor='white', edgecolor='black', linewidth=1.5))
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

ax3d_ground = fig.add_subplot(gs[0, 0], projection='3d')
axxy_ground = fig.add_subplot(gs[0, 1])
axyz_ground = fig.add_subplot(gs[0, 2])
ax3d_flown = fig.add_subplot(gs[1, 0], projection='3d')
axxy_flown = fig.add_subplot(gs[1, 1])
axyz_flown = fig.add_subplot(gs[1, 2])
cax = fig.add_subplot(gs[:, 3])

draw_3d_scene(ax3d_ground, z_ground)
im = plot_horizontal(axxy_ground, SPL_ground_xy, r'Horizontal plan')
plot_vertical(axyz_ground, SPL_ground_yz, z_ground, r'Vertical plan')

draw_3d_scene(ax3d_flown, z_flown)
plot_horizontal(axxy_flown, SPL_flown_xy)
plot_vertical(axyz_flown, SPL_flown_yz, z_flown)


# ============================================================
# INFORMATION BOXES
# ============================================================

gradient_box = r'$\Delta L = L_{\mathrm{max}} - L_{\mathrm{min}}$' + '\n' + rf'$z = {z_ground:g}\ \mathrm{{m}}$ : $\Delta L = {gradient_ground:.2f}\ \mathrm{{dB}}$' + '\n' + rf'$z = {z_flown:g}\ \mathrm{{m}}$ : $\Delta L = {gradient_flown:.2f}\ \mathrm{{dB}}$'
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
