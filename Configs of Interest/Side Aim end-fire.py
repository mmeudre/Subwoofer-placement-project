import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
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
z_max = 14
y_slice = 0

# Subwoofer dimensions
sub_w = 1.0
sub_d = 0.8
sub_h = 0.6

# Array heights (bas des caisses)
z_ground = 0
z_flown = 10

# Scenario : 'concert' (foule en x > 0) ou 'stadium' (zone sous l'array)
SCENARIO = 'concert'
Z_START = z_ground

# Fan geometry
n_rows = 1          # lignes cote a cote
n_front = 3         # caisses en profondeur, array central
n_side = 2          # caisses en profondeur, arrays lateraux
f_design = 80       # frequence d'accord du rejet arriere [Hz]
spacing = c/(4*f_design)   # pas end-fire : lambda/4 a f_design (1.07 m)
side_aim = 0       # ouverture des arrays lateraux [deg]
gap = 1.0           # jeu entre central et lateraux
side_advance = -0.5  # avance des lateraux le long de leur axe de tir [m]
delay = spacing/c
delay_ms = delay*1000

# Evaluation and display
evaluation_margin = 0.5
target = -3
level_min = -40
level_max = 0
levels = np.linspace(level_min, level_max, 21)


# ============================================================
# ARRAY GEOMETRY
# ============================================================

def endfire_rows(n_rows, n_depth, aim_deg, dx=0.0, dy=0.0):
    a = np.deg2rad(aim_deg)
    ca, sa = np.cos(a), np.sin(a)
    boxes = []
    for o in (np.arange(n_rows) - (n_rows-1)/2)*sub_w:
        for i in range(n_depth):
            d = (i - (n_depth-1))*spacing
            boxes.append((-o*sa + d*ca + dx, o*ca + d*sa + dy, aim_deg, i*delay))
    return boxes


def footprint(xb, yb, yaw):
    """Les 4 coins de l'empreinte au sol d'une caisse."""
    a = np.deg2rad(yaw)
    ca, sa = np.cos(a), np.sin(a)
    corners = [(+sub_d/2, +sub_w/2), (+sub_d/2, -sub_w/2),
               (-sub_d/2, -sub_w/2), (-sub_d/2, +sub_w/2)]
    return [(xb + u*ca - v*sa, yb + u*sa + v*ca) for u, v in corners]


offset = n_rows*sub_w + gap
boxes = endfire_rows(n_rows, n_front, 0.0)

for sign in (+1, -1):
    a = np.deg2rad(sign*side_aim)
    boxes += endfire_rows(n_rows, n_side, sign*side_aim,
                          dx=-sign*offset*np.sin(a) + side_advance*np.cos(a),
                          dy=sign*offset*np.cos(a) + side_advance*np.sin(a))

# Convention commune : x = 0 sur la face avant la plus en avant de l'array
x_front = max(max(u for u, v in footprint(xb, yb, yaw)) for xb, yb, yaw, tau in boxes)
boxes = [(xb - x_front, yb, yaw, tau) for xb, yb, yaw, tau in boxes]

N_sub = len(boxes)


def source_positions(z_start):
    """Centre acoustique de chaque caisse : centre de la caisse, a mi-hauteur."""
    return [(xb, yb, z_start + sub_h/2, tau) for xb, yb, yaw, tau in boxes]


def array(z_start=Z_START):
    """Interface commune : sources (L, 3) [m], tau (L,) [s], gain (L,) signe."""
    S = source_positions(z_start)
    sources = np.array([s[:3] for s in S])
    tau = np.array([s[3] for s in S])
    return sources, tau, np.ones(len(S))


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


def fan_top_view(X, Y, z_obs, z_start):
    p = np.zeros_like(X, dtype=complex)
    for xs, ys, zs, tau in source_positions(z_start):
        p += np.exp(-1j*2*np.pi*f*tau)*monopole_top_view(X, Y, z_obs, xs, ys, zs)
    return p


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def config_title(z):
    return rf'{N_sub} subwoofers in a circular gradient end-fire from $z = {z:g}\ \mathrm{{m}}$'


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


def draw_3d_scene(ax, z_start):
    xx = np.array([[-8, 8], [-8, 8]])
    yy = np.array([[-8, -8], [8, 8]])
    zz_ground = np.zeros((2, 2))
    zz_obs = z_obs*np.ones((2, 2))

    xx_slice = np.array([[-8, 8], [-8, 8]])
    yy_slice = y_slice*np.ones((2, 2))
    zz_slice = np.array([[z_min, z_min], [z_max, z_max]])

    ax.plot_surface(xx, yy, zz_ground, color='lightgray', alpha=0.25, linewidth=0)
    ax.plot_surface(xx, yy, zz_obs, color='limegreen', alpha=0.2, linewidth=0)
    ax.plot_surface(xx_slice, yy_slice, zz_slice, color='blue', alpha=0.2, linewidth=0)

    for xb, yb, yaw, tau in boxes:
        draw_sub_box(ax, xb, yb, yaw, z_start)

    ax.set_title(config_title(z_start), fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_zlabel(r'$z$ [m]')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect((16, 16, 14))
    ax.view_init(elev=22, azim=-58)


def plot_horizontal(ax, SPL, title=None):
    im = ax.contourf(X, Y, SPL, levels=levels, cmap='inferno')
    ax.contour(X, Y, SPL, levels=[target], colors='red', linewidths=1, linestyles='solid')

    for xb, yb, yaw, tau in boxes:
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
# BRIGHT ZONE (SEPARATE WINDOW)
# ============================================================

# Meme zone que bright_zone() dans Algorithms/metrics.py : carre de zone_size a z_obs ;
# concert : x de zone_x_start a zone_x_start + zone_size, centre en y = 0 ;
# stadium : centre sous l'array (moyenne des centres acoustiques).
zone_size = 50.0
zone_x_start = 3.0
zone_step = 50.0/70        # 70 mailles sur 50 m : pas de 0.714 m, comme metrics.py


def bright_zone_extent():
    """(x_min, x_max, y_min, y_max) de la bright zone du scenario."""
    if SCENARIO == 'concert':
        return zone_x_start, zone_x_start + zone_size, -zone_size/2, zone_size/2
    sources = array()[0]
    cx, cy = sources[:, 0].mean(), sources[:, 1].mean()
    return cx - zone_size/2, cx + zone_size/2, cy - zone_size/2, cy + zone_size/2


def bright_zone_points():
    """Points de controle : centres des mailles de cote zone_step (comme metrics.py)."""
    x0, x1, y0, y1 = bright_zone_extent()
    offsets = (np.arange(int(round(zone_size/zone_step))) + 0.5)*zone_step
    return np.meshgrid(x0 + offsets, y0 + offsets)


def bright_zone_figure(z_start=Z_START):
    """Array a sa hauteur d'evaluation et bright zone : vue 3D et vue de dessus."""
    x0, x1, y0, y1 = bright_zone_extent()
    corners = np.array([c for xb, yb, yaw, tau in boxes for c in footprint(xb, yb, yaw)])
    xmin, xmax = min(x0, corners[:, 0].min()) - 5, max(x1, corners[:, 0].max()) + 5
    ymin, ymax = min(y0, corners[:, 1].min()) - 5, max(y1, corners[:, 1].max()) + 5
    zmax = max(15, z_start + sub_h + 3)

    fig = plt.figure(figsize=(17, 7.5))
    fig.suptitle(rf'Side Aim end-fire : bright zone ({SCENARIO}, $z = {z_obs:g}$ m), array at $z = {z_start:g}$ m',
                 fontsize=14, fontweight='bold')

    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax3d.plot_surface(np.array([[xmin, xmax], [xmin, xmax]]), np.array([[ymin, ymin], [ymax, ymax]]),
                      np.zeros((2, 2)), color='lightgray', alpha=0.25, linewidth=0)
    ax3d.plot_surface(np.array([[x0, x1], [x0, x1]]), np.array([[y0, y0], [y1, y1]]),
                      np.full((2, 2), z_obs), color='navy', alpha=0.7, linewidth=0)
    xp, yp = bright_zone_points()
    for xb, yb, yaw, tau in boxes:
        draw_sub_box(ax3d, xb, yb, yaw, z_start)
    ax3d.set_title('3D view', fontsize=12)
    ax3d.set_xlabel(r'$x$ [m]')
    ax3d.set_ylabel(r'$y$ [m]')
    ax3d.set_zlabel(r'$z$ [m]')
    ax3d.set_xlim(xmin, xmax)
    ax3d.set_ylim(ymin, ymax)
    ax3d.set_zlim(0, zmax)
    ax3d.set_box_aspect((xmax - xmin, ymax - ymin, 0.35*max(xmax - xmin, ymax - ymin)))
    ax3d.view_init(elev=22, azim=-58)

    ax = fig.add_subplot(1, 2, 2)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor='navy', alpha=0.7, edgecolor='none'))
    ax.scatter(xp, yp, s=0.5, color='white', linewidths=0)
    for xb, yb, yaw, tau in boxes:
        ax.add_patch(Polygon(footprint(xb, yb, yaw), closed=True,
                             facecolor='white', edgecolor='black', linewidth=1.0))
    ax.set_title(rf'Top view : {xp.size} control points, step ${zone_step:.2f}$ m', fontsize=12)
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    return fig


if __name__ == '__main__':
    # ============================================================
    # EVALUATION AREA
    # ============================================================

    source_zone = np.zeros_like(X, dtype=bool)
    for xb, yb, yaw, tau in boxes:
        source_zone |= ((np.abs(X-xb) <= sub_d/2 + evaluation_margin)
                        & (np.abs(Y-yb) <= sub_w/2 + evaluation_margin))
    evaluation_zone = ~source_zone


    # ============================================================
    # PRESSURE FIELDS
    # ============================================================

    p_ground_xy = fan_top_view(X, Y, z_obs, z_ground)
    p_flown_xy = fan_top_view(X, Y, z_obs, z_flown)

    ref_ground = np.max(np.abs(p_ground_xy[evaluation_zone]))
    ref_flown = np.max(np.abs(p_flown_xy[evaluation_zone]))

    SPL_ground_xy_raw = 20*np.log10(np.maximum(np.abs(p_ground_xy), 1e-12)/ref_ground)
    SPL_flown_xy_raw = 20*np.log10(np.maximum(np.abs(p_flown_xy), 1e-12)/ref_flown)

    SPL_ground_xy = np.clip(SPL_ground_xy_raw, level_min, level_max)
    SPL_flown_xy = np.clip(SPL_flown_xy_raw, level_min, level_max)


    # ============================================================
    # FIGURE
    # ============================================================

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.35, 0.05],
                          left=0.04, right=0.91, bottom=0.21, top=0.93,
                          wspace=0.22, hspace=0.26)

    ax3d_ground = fig.add_subplot(gs[0, 0], projection='3d')
    axxy_ground = fig.add_subplot(gs[0, 1])

    ax3d_flown = fig.add_subplot(gs[1, 0], projection='3d')
    axxy_flown = fig.add_subplot(gs[1, 1])

    cax = fig.add_subplot(gs[:, 2])

    draw_3d_scene(ax3d_ground, z_ground)
    im = plot_horizontal(axxy_ground, SPL_ground_xy, r'Horizontal plane')

    draw_3d_scene(ax3d_flown, z_flown)
    plot_horizontal(axxy_flown, SPL_flown_xy)


    # ============================================================
    # INFORMATION BOXES
    # ============================================================

    # Each box is centred under the panel it describes
    axxy_flown.apply_aspect()
    x_3d = 0.5 * sum(ax3d_flown.get_position().intervalx)
    x_maps = 0.5 * sum(axxy_flown.get_position().intervalx)
    y_box = 0.07

    fan_box = (rf'Central array : {n_rows} rows x {n_front} deep;' + '\n'
               + rf'side arrays : {n_rows} rows x {n_side} deep at $\pm {side_aim:g}^\circ$;' + '\n'
               + rf'end-fire spacing $d = {spacing:.2f}\ \mathrm{{m}}$, $\tau = {delay_ms:.2f}\ \mathrm{{ms}}$ per step' + '\n'
               + rf'maps computed at $f = {f:g}\ \mathrm{{Hz}}$')
    fig.text(x_maps, y_box, fan_box, ha='center', va='center',
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

    fig_zone = bright_zone_figure()

    plt.show()
