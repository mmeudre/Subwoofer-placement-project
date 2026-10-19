import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.close('all')

# ============================================================
# 1. GLOBAL PARAMETERS
# ============================================================
c = 343
rho = 1.2
f = 80
Q = 1e-3
k = 2*np.pi*f/c
omega = 2*np.pi*f
z_obs = 1.5
z_ground = 0
z_flown = 10
sub_w = 1.0
sub_d = 0.8
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
Y_yz, Z_yz = np.meshgrid(y, z)
x_slice = 0

# ============================================================
# 2. ARRAY GEOMETRIES
# ============================================================
def source(xs, ys, zs=0, angle=0, weight=1+0j, facecolor='white', acoustic_y=None):
    return {'x': float(xs), 'y': float(ys), 'z': float(zs), 'angle': float(angle), 'weight': complex(weight), 'facecolor': facecolor, 'acoustic_y': float(ys if acoustic_y is None else acoustic_y)}


def offset_sources(base, z_offset):
    out = []
    for src in base:
        out.append(source(src['x'], src['y'], src['z']+z_offset, src['angle'], src['weight'], src['facecolor'], src['acoustic_y']))
    return out


def geometries():
    configs = {}

    # --------------------------------------------------------
    # Arc end-fire - central 3 deep + 2 side branches = 7 subs
    # --------------------------------------------------------
    d_arc = 1.20
    tau_arc = 3.50e-3
    side_angle = 25
    side_offset = 2.2
    a = np.deg2rad(side_angle)
    arc_endfire = [
        source(0.0, -d_arc, 0, 0, np.exp(-1j*omega*2*tau_arc), 'white'),
        source(0.0, 0.0, 0, 0, np.exp(-1j*omega*tau_arc), 'white'),
        source(0.0, d_arc, 0, 0, 1.0, 'white')
    ]

    x_left_front = -side_offset
    y_left_front = -d_arc
    x_left_rear = x_left_front + d_arc*np.sin(a)
    y_left_rear = y_left_front + d_arc*np.cos(a)
    arc_endfire += [
        source(x_left_front, y_left_front, 0, -side_angle, np.exp(-1j*omega*tau_arc), 'white'),
        source(x_left_rear, y_left_rear, 0, -side_angle, 1.0, 'white')
    ]

    x_right_front = side_offset
    y_right_front = -d_arc
    x_right_rear = x_right_front - d_arc*np.sin(a)
    y_right_rear = y_right_front + d_arc*np.cos(a)
    arc_endfire += [
        source(x_right_front, y_right_front, 0, side_angle, np.exp(-1j*omega*tau_arc), 'white'),
        source(x_right_rear, y_right_rear, 0, side_angle, 1.0, 'white')
    ]
    configs['Arc end-fire'] = arc_endfire

    return configs

# ============================================================
# 3. ACOUSTIC MODEL: MONOPOLES + RIGID GROUND IMAGE
# ============================================================
def monopole_field(A, B, C, xs, ys, zs):
    rd = np.maximum(np.sqrt((A-xs)**2 + (B-ys)**2 + (C-zs)**2), 0.05)
    ri = np.maximum(np.sqrt((A-xs)**2 + (B-ys)**2 + (C+zs)**2), 0.05)
    return 1j*rho*c*k*Q/(4*np.pi)*(np.exp(-1j*k*rd)/rd + np.exp(-1j*k*ri)/ri)


def pressure_xy(sources):
    p = np.zeros_like(X, dtype=complex)
    for src in sources:
        p += src['weight']*monopole_field(X, Y, z_obs, src['x'], src['acoustic_y'], src['z'])
    return p


def pressure_yz(sources):
    p = np.zeros_like(Y_yz, dtype=complex)
    for src in sources:
        p += src['weight']*monopole_field(x_slice, Y_yz, Z_yz, src['x'], src['acoustic_y'], src['z'])
    return p


def evaluation_mask(sources):
    excluded = np.zeros_like(X, dtype=bool)
    for src in sources:
        a = np.deg2rad(src['angle'])
        dx = abs(sub_w*np.cos(a)) + abs(sub_d*np.sin(a))
        dy = abs(sub_w*np.sin(a)) + abs(sub_d*np.cos(a))
        excluded |= (abs(X-src['x']) <= dx/2+evaluation_margin) & (abs(Y-src['y']) <= dy/2+evaluation_margin)
    return ~excluded


def spl_maps(sources):
    p_xy = pressure_xy(sources)
    ref = np.max(np.abs(p_xy[evaluation_mask(sources)]))
    xy = 20*np.log10(np.maximum(np.abs(p_xy), 1e-12)/ref)
    yz = 20*np.log10(np.maximum(np.abs(pressure_yz(sources)), 1e-12)/ref)
    return np.clip(xy, level_min, level_max), np.clip(yz, level_min, level_max)

# ============================================================
# 4. SUBWOOFER DRAWING HELPERS
# ============================================================
def corners_2d(xs, ys, angle):
    a = np.deg2rad(angle)
    corners = np.array([[-sub_w/2, -sub_d/2], [sub_w/2, -sub_d/2], [sub_w/2, sub_d/2], [-sub_w/2, sub_d/2]])
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return corners @ R.T + [xs, ys]


def draw_sub_3d(ax, src):
    xs, ys, zs, angle = src['x'], src['y'], src['z'], src['angle']
    z0 = max(0, zs-sub_h/2)
    z1 = z0+sub_h
    corners = corners_2d(xs, ys, angle)
    bottom = [[px, py, z0] for px, py in corners]
    top = [[px, py, z1] for px, py in corners]
    faces = [bottom, top] + [[bottom[i], bottom[(i+1)%4], top[(i+1)%4], top[i]] for i in range(4)]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=src['facecolor'], edgecolors='black', linewidths=0.8))


def draw_3d_scene(ax, sources, title):
    xx = np.array([[-10, 10], [-10, 10]])
    yy = np.array([[-10, -10], [10, 10]])
    ax.plot_surface(xx, yy, np.zeros((2, 2)), color='lightgray', alpha=0.25, linewidth=0)
    ax.plot_surface(xx, yy, np.full((2, 2), z_obs), color='limegreen', alpha=0.2, linewidth=0)
    ys = np.array([[-10, 10], [-10, 10]])
    zz = np.array([[0, 0], [15, 15]])
    ax.plot_surface(np.zeros((2, 2)), ys, zz, color='blue', alpha=0.2, linewidth=0)
    for src in sources:
        draw_sub_3d(ax, src)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_zlabel(r'$z$ [m]')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 15)
    ax.set_box_aspect((20, 20, 15))
    ax.view_init(elev=22, azim=-58)


def plot_horizontal(ax, SPL, sources, title=None):
    im = ax.contourf(X, Y, SPL, levels=levels, cmap='inferno', extend='neither')
    if SPL.min() <= target <= SPL.max():
        ax.contour(X, Y, SPL, levels=[target], colors='red', linewidths=1, linestyles='solid')
    for src in sources:
        rect = Rectangle((-sub_w/2, -sub_d/2), sub_w, sub_d, facecolor=src['facecolor'], edgecolor='black', linewidth=1)
        rect.set_transform(Affine2D().rotate_deg(src['angle']).translate(src['x'], src['y']) + ax.transData)
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


def plot_vertical(ax, SPL, sources, title=None):
    ax.contourf(Y_yz, Z_yz, SPL, levels=levels, cmap='inferno')
    if SPL.min() <= target <= SPL.max():
        ax.contour(Y_yz, Z_yz, SPL, levels=[target], colors='red', linewidths=1, linestyles='solid')
    for src in sources:
        corners = corners_2d(src['x'], src['y'], src['angle'])
        if corners[:, 0].min() <= x_slice <= corners[:, 0].max():
            ax.add_patch(Rectangle((corners[:, 1].min(), max(0, src['z']-sub_h/2)), np.ptp(corners[:, 1]), sub_h, facecolor=src['facecolor'], edgecolor='black', linewidth=1))
    ax.axhline(0, color='black', linewidth=1)
    ax.axhline(z_obs, color='limegreen', linewidth=1, alpha=0.5)
    if title:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel(r'$y$ [m]')
    ax.set_ylabel(r'$z$ [m]')
    ax.set_xlim(-50, 50)
    ax.set_ylim(0, 50)
    ax.set_xticks(np.arange(-50, 51, 10))
    ax.set_yticks(np.arange(0, 51, 10))

# ============================================================
# 5. FIGURES: ONE PER CONFIGURATION, TWO HEIGHTS
# ============================================================
def simulate_configuration(name, base):
    ground = offset_sources(base, z_ground)
    flown = offset_sources(base, z_flown)
    spl_ground_xy, spl_ground_yz = spl_maps(ground)
    spl_flown_xy, spl_flown_yz = spl_maps(flown)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(name, fontsize=15, fontweight='bold')
    gs = fig.add_gridspec(2, 4, width_ratios=[1.1, 1, 1, 0.05], left=0.05, right=0.94, bottom=0.08, top=0.90, wspace=0.32, hspace=0.35)
    ax3d_g = fig.add_subplot(gs[0, 0], projection='3d')
    axxy_g = fig.add_subplot(gs[0, 1])
    axyz_g = fig.add_subplot(gs[0, 2])
    ax3d_f = fig.add_subplot(gs[1, 0], projection='3d')
    axxy_f = fig.add_subplot(gs[1, 1])
    axyz_f = fig.add_subplot(gs[1, 2])
    cax = fig.add_subplot(gs[:, 3])

    draw_3d_scene(ax3d_g, ground, r'Ground array ($z=0$ m)')
    im = plot_horizontal(axxy_g, spl_ground_xy, ground, 'Horizontal plan')
    plot_vertical(axyz_g, spl_ground_yz, ground, 'Vertical plan (YZ)')
    draw_3d_scene(ax3d_f, flown, r'Flown array ($z=10$ m)')
    plot_horizontal(axxy_f, spl_flown_xy, flown)
    plot_vertical(axyz_f, spl_flown_yz, flown)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Normalised level [dB]', fontsize=12)
    cbar.ax.axhline(target, color='red', linewidth=1, linestyle='solid')
    return fig

# ============================================================
# 6. RUN CONFIGURATION
# ============================================================
if __name__ == '__main__':
    name = 'Arc end-fire'
    print('Simulating:', name)
    simulate_configuration(name, geometries()[name])
    plt.show()
