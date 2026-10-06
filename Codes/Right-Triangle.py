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

# Source heights
z_ground = 0
z_flown = 10

# Subwoofer dimensions
sub_w = 1.0
sub_h = 0.8
sub_z = 0.6

# Triangle geometry
square_half_size = 5.0
side_positions = np.array([-10/3, 0.0, 10/3])

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

# Top horizontal line
for xs in side_positions:
    sources.append({'x': xs, 'y': square_half_size, 'angle': 0})

# Left vertical line
for ys in side_positions:
    sources.append({'x': -square_half_size, 'y': ys, 'angle': 90})

# Diagonal line
for pos in side_positions:
    sources.append({'x': pos, 'y': pos, 'angle': 45})

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


def monopole_xz_view(X, Z, y_slice, xs, ys, zs):
    r_direct = np.sqrt((X-xs)**2 + (y_slice-ys)**2 + (Z-zs)**2)
    r_image = np.sqrt((X-xs)**2 + (y_slice-ys)**2 + (Z+zs)**2)
    r_direct = np.maximum(r_direct, 0.05)
    r_image = np.maximum(r_image, 0.05)
    p_direct = 1j*rho*c*k*Q*np.exp(-1j*k*r_direct)/(4*np.pi*r_direct)
    p_image = 1j*rho*c*k*Q*np.exp(-1j*k*r_image)/(4*np.pi*r_image)
    return p_direct + p_image


def triangle_array_top_view(X, Y, z_obs, z_array):
    p = np.zeros_like(X, dtype=complex)
    for src in sources:
        p += monopole_top_view(X, Y, z_obs, src['x'], src['y'], z_array)
    return p


def triangle_array_xz_view(X, Z, y_slice, z_array):
    p = np.zeros_like(X, dtype=complex)
    for src in sources:
        p += monopole_xz_view(X, Z, y_slice, src['x'], src['y'], z_array)
    return p


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def deg2rad(angle):
    return np.deg2rad(angle)


def projected_dims(angle_deg):
    theta = deg2rad(angle_deg)
    proj_x = np.abs(sub_w*np.cos(theta)) + np.abs(sub_h*np.sin(theta))
    proj_y = np.abs(sub_w*np.sin(theta)) + np.abs(sub_h*np.cos(theta))
    return proj_x, proj_y


def config_title(z):
    return rf'{N_sub} subwoofers in a right-triangle array from $z = {z:g}\ \mathrm{{m}}$'


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

p_ground_xy = triangle_array_top_view(X, Y, z_obs, z_ground)
p_flown_xy = triangle_array_top_view(X, Y, z_obs, z_flown)

ref_ground = np.max(np.abs(p_ground_xy[evaluation_zone]))
ref_flown = np.max(np.abs(p_flown_xy[evaluation_zone]))

SPL_ground_xy_raw = 20*np.log10(np.maximum(np.abs(p_ground_xy), 1e-12)/ref_ground)
SPL_flown_xy_raw = 20*np.log10(np.maximum(np.abs(p_flown_xy), 1e-12)/ref_flown)

gradient_ground = np.max(SPL_ground_xy_raw[evaluation_zone]) - np.min(SPL_ground_xy_raw[evaluation_zone])
gradient_flown = np.max(SPL_flown_xy_raw[evaluation_zone]) - np.min(SPL_flown_xy_raw[evaluation_zone])

SPL_ground_xy = np.clip(SPL_ground_xy_raw, level_min, level_max)
SPL_flown_xy = np.clip(SPL_flown_xy_raw, level_min, level_max)

p_ground_xz = triangle_array_xz_view(X_xz, Z_xz, y_slice, z_ground)
p_flown_xz = triangle_array_xz_view(X_xz, Z_xz, y_slice, z_flown)

SPL_ground_xz = np.clip(20*np.log10(np.maximum(np.abs(p_ground_xz), 1e-12)/ref_ground), level_min, level_max)
SPL_flown_xz = np.clip(20*np.log10(np.maximum(np.abs(p_flown_xz), 1e-12)/ref_flown), level_min, level_max)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def rotated_corners_2d(xc, yc, angle_deg, w, h):
    theta = deg2rad(angle_deg)
    corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = corners @ R.T
    rotated[:, 0] += xc
    rotated[:, 1] += yc
    return rotated


def add_sub_top_view(ax, src):
    rect = Rectangle((-sub_w/2, -sub_h/2), sub_w, sub_h, facecolor='white', edgecolor='black', linewidth=1.5)
    rect.set_transform(Affine2D().rotate_deg(src['angle']).translate(src['x'], src['y']) + ax.transData)
    ax.add_patch(rect)


def draw_sub_box_3d(ax, src, z_source):
    z0 = max(0, z_source-sub_z/2)
    z1 = z0 + sub_z
    bottom_xy = rotated_corners_2d(src['x'], src['y'], src['angle'], sub_w, sub_h)
    bottom = [[p[0], p[1], z0] for p in bottom_xy]
    top = [[p[0], p[1], z1] for p in bottom_xy]

    faces = [
        [bottom[0], bottom[1], bottom[2], bottom[3]],
        [top[0], top[1], top[2], top[3]],
        [bottom[0], bottom[1], top[1], top[0]],
        [bottom[1], bottom[2], top[2], top[1]],
        [bottom[2], bottom[3], top[3], top[2]],
        [bottom[3], bottom[0], top[0], top[3]]
    ]

    poly = Poly3DCollection(faces, facecolors='white', edgecolors='black', linewidths=1.2)
    ax.add_collection3d(poly)


def draw_3d_scene(ax, z_source):
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

    for src in sources:
        draw_sub_box_3d(ax, src, z_source)

    ax.set_title(config_title(z_source), fontsize=12, fontweight='bold')
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

    for src in sources:
        add_sub_top_view(ax, src)

    if title is not None:
        ax.set_title(title, fontsize=12)

    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    return im


def plot_vertical(ax, SPL, z_source, title=None):
    ax.contourf(X_xz, Z_xz, SPL, levels=levels, cmap='inferno')
    ax.contour(X_xz, Z_xz, SPL, levels=[target], colors='red', linewidths=1, linestyles='solid')

    for src in sources:
        proj_x, proj_y = projected_dims(src['angle'])
        if np.abs(src['y']-y_slice) <= proj_y/2:
            ax.add_patch(Rectangle((src['x']-proj_x/2, max(0, z_source-sub_z/2)), proj_x, sub_z, facecolor='white', edgecolor='black', linewidth=1.5))

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

ax3d_ground = fig.add_subplot(gs[0, 0], projection='3d')
axxy_ground = fig.add_subplot(gs[0, 1])
axxz_ground = fig.add_subplot(gs[0, 2])

ax3d_flown = fig.add_subplot(gs[1, 0], projection='3d')
axxy_flown = fig.add_subplot(gs[1, 1])
axxz_flown = fig.add_subplot(gs[1, 2])

cax = fig.add_subplot(gs[:, 3])

draw_3d_scene(ax3d_ground, z_ground)
im = plot_horizontal(axxy_ground, SPL_ground_xy, r'Horizontal plan')
plot_vertical(axxz_ground, SPL_ground_xz, z_ground, r'Vertical plan')

draw_3d_scene(ax3d_flown, z_flown)
plot_horizontal(axxy_flown, SPL_flown_xy)
plot_vertical(axxz_flown, SPL_flown_xz, z_flown)


# ============================================================
# INFORMATION BOXES
# ============================================================

gradient_box = r'$\Delta L = L_{\mathrm{max}} - L_{\mathrm{min}}$' + '\n' + config_label(z_ground) + rf' : $\Delta L = {gradient_ground:.2f}\ \mathrm{{dB}}$' + '\n' + config_label(z_flown) + rf' : $\Delta L = {gradient_flown:.2f}\ \mathrm{{dB}}$'
fig.text(0.935, 0.05, gradient_box, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

triangle_box = r'Three subwoofer lines forming a right triangle of $10 \times 10\ \mathrm{m}$'
fig.text(0.47, 0.05, triangle_box, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

green_line = Line2D([0], [0], color='limegreen', linewidth=1, alpha=0.5)
fig.legend([green_line], [rf'Observation plane : $z = {z_obs:g}\ \mathrm{{m}}$'], loc='lower center', bbox_to_anchor=(0.74, 0.025), frameon=True, edgecolor='black', fancybox=True)


# ============================================================
# COLORBAR
# ============================================================

cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r'Normalised level [dB]', fontsize=12)
cbar.ax.axhline(target, color='red', linewidth=1, linestyle='solid')

plt.show()