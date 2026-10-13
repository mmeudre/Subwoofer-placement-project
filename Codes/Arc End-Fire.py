import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
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
omega = 2*np.pi*f

# Observation area
z_obs = 1.5
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
z_flown = 10

# Subwoofer dimensions
sub_w = 1.0
sub_h = 0.8
sub_z = 0.6

# End-fire parameters
d = 1.20
tau = 3.50e-3
tau_ms = tau*1000
side_angle = 25
side_offset = 2.2

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

# Central array: 1 row x 3 deep
sources.append({'x': 0.0, 'y': -d, 'angle': 0, 'delay': 2*tau})
sources.append({'x': 0.0, 'y': 0.0, 'angle': 0, 'delay': tau})
sources.append({'x': 0.0, 'y': d, 'angle': 0, 'delay': 0.0})

# Side arrays: 1 row x 2 deep at ±25°
a = np.deg2rad(side_angle)

# Left side
x_left_front = -side_offset
y_left_front = -d
x_left_rear = x_left_front + d*np.sin(a)
y_left_rear = y_left_front + d*np.cos(a)

sources.append({'x': x_left_front, 'y': y_left_front, 'angle': -side_angle, 'delay': tau})
sources.append({'x': x_left_rear, 'y': y_left_rear, 'angle': -side_angle, 'delay': 0.0})

# Right side
x_right_front = side_offset
y_right_front = -d
x_right_rear = x_right_front - d*np.sin(a)
y_right_rear = y_right_front + d*np.cos(a)

sources.append({'x': x_right_front, 'y': y_right_front, 'angle': side_angle, 'delay': tau})
sources.append({'x': x_right_rear, 'y': y_right_rear, 'angle': side_angle, 'delay': 0.0})

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


def endfire_top_view(X, Y, z_array):
    p = np.zeros_like(X, dtype=complex)
    for src in sources:
        weight = np.exp(-1j*omega*src['delay'])
        p += weight*monopole_top_view(X, Y, z_obs, src['x'], src['y'], z_array)
    return p


def endfire_yz_view(Y, Z, z_array):
    p = np.zeros_like(Y, dtype=complex)
    for src in sources:
        weight = np.exp(-1j*omega*src['delay'])
        p += weight*monopole_yz_view(Y, Z, x_slice, src['x'], src['y'], z_array)
    return p


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def projected_dims(angle_deg):
    angle = np.deg2rad(angle_deg)
    proj_x = np.abs(sub_w*np.cos(angle)) + np.abs(sub_h*np.sin(angle))
    proj_y = np.abs(sub_w*np.sin(angle)) + np.abs(sub_h*np.cos(angle))
    return proj_x, proj_y


def rotated_corners(xc, yc, angle_deg):
    angle = np.deg2rad(angle_deg)
    corners = np.array([[-sub_w/2, -sub_h/2], [sub_w/2, -sub_h/2], [sub_w/2, sub_h/2], [-sub_w/2, sub_h/2]])
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = corners @ R.T
    corners[:, 0] += xc
    corners[:, 1] += yc
    return corners


def config_title(z):
    return rf'{N_sub} subwoofers in an end-fire array from $z = {z:g}\ \mathrm{{m}}$'


def config_label(z):
    return rf'$z = {z:g}\ \mathrm{{m}}$'


# ============================================================
# EVALUATION AREA
# ============================================================

source_zone = np.zeros_like(X, dtype=bool)

for src in sources:
    proj_x, proj_y = projected_dims(src['angle'])
    source_zone |= (np.abs(X-src['x']) <= proj_x/2 + evaluation_margin) & (np.abs(Y-src['y']) <= proj_y/2 + evaluation_margin)

evaluation_zone = ~source_zone


# ============================================================
# PRESSURE FIELDS
# ============================================================

p_ground_xy = endfire_top_view(X, Y, z_ground)
p_flown_xy = endfire_top_view(X, Y, z_flown)

ref_ground = np.max(np.abs(p_ground_xy[evaluation_zone]))
ref_flown = np.max(np.abs(p_flown_xy[evaluation_zone]))

SPL_ground_xy_raw = 20*np.log10(np.maximum(np.abs(p_ground_xy), 1e-12)/ref_ground)
SPL_flown_xy_raw = 20*np.log10(np.maximum(np.abs(p_flown_xy), 1e-12)/ref_flown)

gradient_ground = np.max(SPL_ground_xy_raw[evaluation_zone]) - np.min(SPL_ground_xy_raw[evaluation_zone])
gradient_flown = np.max(SPL_flown_xy_raw[evaluation_zone]) - np.min(SPL_flown_xy_raw[evaluation_zone])

SPL_ground_xy = np.clip(SPL_ground_xy_raw, level_min, level_max)
SPL_flown_xy = np.clip(SPL_flown_xy_raw, level_min, level_max)

p_ground_yz = endfire_yz_view(Y_yz, Z_yz, z_ground)
p_flown_yz = endfire_yz_view(Y_yz, Z_yz, z_flown)

SPL_ground_yz = np.clip(20*np.log10(np.maximum(np.abs(p_ground_yz), 1e-12)/ref_ground), level_min, level_max)
SPL_flown_yz = np.clip(20*np.log10(np.maximum(np.abs(p_flown_yz), 1e-12)/ref_flown), level_min, level_max)


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def add_sub_top(ax, src):
    rect = Rectangle((-sub_w/2, -sub_h/2), sub_w, sub_h, facecolor='white', edgecolor='black', linewidth=1.5)
    rect.set_transform(Affine2D().rotate_deg(src['angle']).translate(src['x'], src['y']) + ax.transData)
    ax.add_patch(rect)


def draw_sub_3d(ax, src, z_source):
    z0 = max(0, z_source-sub_z/2)
    z1 = z0 + sub_z
    xy = rotated_corners(src['x'], src['y'], src['angle'])
    bottom = [[px, py, z0] for px, py in xy]
    top = [[px, py, z1] for px, py in xy]

    faces = [
        [bottom[0], bottom[1], bottom[2], bottom[3]],
        [top[0], top[1], top[2], top[3]],
        [bottom[0], bottom[1], top[1], top[0]],
        [bottom[1], bottom[2], top[2], top[1]],
        [bottom[2], bottom[3], top[3], top[2]],
        [bottom[3], bottom[0], top[0], top[3]]
    ]

    ax.add_collection3d(Poly3DCollection(faces, facecolors='white', edgecolors='black', linewidths=1.2))


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
        draw_sub_3d(ax, src, z_source)

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
        add_sub_top(ax, src)

    if title is not None:
        ax.set_title(title, fontsize=12)

    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    return im


def plot_vertical(ax, SPL, z_source, title=None):
    ax.contourf(Y_yz, Z_yz, SPL, levels=levels, cmap='inferno')
    ax.contour(Y_yz, Z_yz, SPL, levels=[target], colors='red', linewidths=1, linestyles='solid')

    for src in sources:
        proj_x, proj_y = projected_dims(src['angle'])
        if np.abs(src['x']-x_slice) <= proj_x/2:
            ax.add_patch(Rectangle((src['y']-proj_y/2, max(0, z_source-sub_z/2)), proj_y, sub_z, facecolor='white', edgecolor='black', linewidth=1.5))

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
plot_vertical(ayz_ground, SPL_ground_yz, z_ground, r'Vertical plan')

draw_3d_scene(ax3d_flown, z_flown)
plot_horizontal(axxy_flown, SPL_flown_xy)
plot_vertical(ayz_flown, SPL_flown_yz, z_flown)


# ============================================================
# INFORMATION BOXES
# ============================================================

gradient_box = r'$\Delta L = L_{\mathrm{max}} - L_{\mathrm{min}}$' + '\n' + config_label(z_ground) + rf' : $\Delta L = {gradient_ground:.2f}\ \mathrm{{dB}}$' + '\n' + config_label(z_flown) + rf' : $\Delta L = {gradient_flown:.2f}\ \mathrm{{dB}}$'
fig.text(0.935, 0.05, gradient_box, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

endfire_box = r'Central array: $1$ row $\times$ $3$ deep;' + '\n' + rf'side arrays: $1$ row $\times$ $2$ deep at $\pm {side_angle}^\circ$;' + '\n' + rf'end-fire spacing $d = {d:.2f}\ \mathrm{{m}}$, $\tau = {tau_ms:.2f}\ \mathrm{{ms}}$ per step' + '\n' + rf'maps computed at $f = {f:g}\ \mathrm{{Hz}}$'
fig.text(0.47, 0.05, endfire_box, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

green_line = Line2D([0], [0], color='limegreen', linewidth=1, alpha=0.5)
fig.legend([green_line], [rf'Observation plane : $z = {z_obs:g}\ \mathrm{{m}}$'], loc='lower center', bbox_to_anchor=(0.74, 0.025), frameon=True, edgecolor='black', fancybox=True)


# ============================================================
# COLORBAR
# ============================================================

cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r'Normalised level [dB]', fontsize=12)
cbar.ax.axhline(target, color='red', linewidth=1, linestyle='solid')

plt.show()