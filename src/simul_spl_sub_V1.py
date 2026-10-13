import os
from dataclasses import dataclass, replace
from functools import cached_property
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

C = 343.0
SUB_W, SUB_D, SUB_H = 1.0, 0.8, 0.6
BAND = np.linspace(30, 124, 94)
BAND.setflags(write=False)
RHO = 1.2
Q_SOURCE = 1e-3
# Hauteur d'observation du projet : le plan horizontal sur lequel toutes les
# cartes SPL de dessus sont calculees, et la hauteur affichee par plot_slice
# et plot_3d_scene. UNE SEULE constante porte cette valeur -- les fabriques
# Zone.grid_xy / Zone.pit / Zone.ring la prennent pour defaut, au lieu de la
# repeter en dur comme avant.
#
# Historique : 1,5 m en dur dans les 7 scripts de Codes/ pour l'affichage,
# contre 1,8 m pour les zones de calcul -- incoherence corrigee par la ronde
# de correction 3, qui avait unifie a 1,8 m. Ramenee a 1,6 m le 06/10/2026
# sur demande de Martin : hauteur d'oreille d'un auditeur debout, plus
# representative de la foule que 1,8 m.
Z_LISTENER = 1.6

@dataclass(frozen=True)
class Sub:
    """Une caisse. (x, y, z) = centre geometrique. yaw_deg : 0 = face vers +x."""
    x: float
    y: float
    z: float
    yaw_deg: float = 0.0
    delay_ms: float = 0.0
    gain_db: float = 0.0
    polarity: int = 1

    def reversed(self) -> "Sub":
        """Mode cardioide : retournee 180 deg, polarite inversee, retardee du
        temps de parcours de la profondeur de caisse."""
        return replace(self, yaw_deg=self.yaw_deg + 180.0,
                       polarity=-self.polarity,
                       delay_ms=self.delay_ms + SUB_D / C * 1e3)

    def moved(self, dx=0.0, dy=0.0, dz=0.0) -> "Sub":
        return replace(self, x=self.x + dx, y=self.y + dy, z=self.z + dz)


class SubArray:
    """Collection immuable de caisses, et les operations de groupe."""

    def __init__(self, subs):
        self._subs = tuple(subs)

    def __len__(self):            return len(self._subs)
    def __iter__(self):           return iter(self._subs)
    def __getitem__(self, i):     return self._subs[i]
    def __add__(self, other):     return SubArray(self._subs + tuple(other))
    def __repr__(self):           return f"SubArray({len(self)} caisses)"

    @cached_property
    def positions(self):
        arr = np.array([[s.x, s.y, s.z] for s in self._subs], dtype=np.float64)
        if arr.size == 0:
            arr = arr.reshape(0, 3)
        arr.setflags(write=False)
        return arr

    @cached_property
    def yaw(self):
        arr = np.deg2rad(np.array([s.yaw_deg for s in self._subs], dtype=np.float64))
        arr.setflags(write=False)
        return arr

    @cached_property
    def weights(self):
        arr = np.array([s.polarity * 10.0 ** (s.gain_db / 20.0)
                         for s in self._subs], dtype=np.complex128)
        arr.setflags(write=False)
        return arr

    @cached_property
    def delays(self):
        arr = np.array([s.delay_ms * 1e-3 for s in self._subs], dtype=np.float64)
        arr.setflags(write=False)
        return arr

    # TRANSFORMATIONS DE GROUPE

    def flown(self, z_bottom):
        dz = z_bottom + SUB_H / 2 - min(s.z for s in self._subs)
        return SubArray([s.moved(dz=dz) for s in self._subs])

    def mirror(self):
        return SubArray([replace(s, y=-s.y, yaw_deg=-s.yaw_deg) for s in self._subs])

    def shift(self, dx=0.0, dy=0.0, dz=0.0):
        return SubArray([s.moved(dx, dy, dz) for s in self._subs])

    def with_cardioid(self, index):
        subs = list(self._subs)
        subs[index] = subs[index].reversed()
        return SubArray(subs)

    def lr_pair(self, half_offset):
        """self decale en -half_offset, plus sa copie en +half_offset.

        Equivaut a construire deux arrays avec y_center=-half_offset puis
        y_center=+half_offset (comme l'original) : y_center entre de facon
        purement additive dans tous les constructeurs, et flown (axe z) /
        shift(dy=...) (axe y) agissent sur des axes disjoints, donc
        commutent exactement. Correct pour tout aim_deg. Bit-exact face a
        l'original seulement a aim_deg = 0 : pour un angle non nul,
        l'original calcule (y_center + o*ca) - back*sa quand cette
        translation calcule (o*ca - back*sa) + y_center, d'où une
        associativite flottante differente et un ecart residuel possible a
        l'ULP.
        """
        return self.shift(dy=-half_offset) + self.shift(dy=+half_offset)

    # CONSTRUCTEURS DE CONFIGURATIONS

    @classmethod
    def single(cls, x=0.0, y=0.0, **kw):
        """Une caisse posee au sol."""
        return cls([Sub(x, y, SUB_H / 2, **kw)])

    @classmethod
    def line(cls, n, spacing=SUB_W, x=0.0, y_center=0.0, **kw):
        """n caisses cote a cote au sol, alignees suivant y (array horizontal)."""
        ys = (np.arange(n) - (n - 1) / 2) * spacing + y_center
        return cls([Sub(x, float(yy), SUB_H / 2, **kw) for yy in ys])

    @classmethod
    def stack(cls, n, x=0.0, y=0.0, z_bottom=0.0, **kw):
        """n caisses empilees a la verticale, la premiere posee a z_bottom."""
        return cls([Sub(x, y, z_bottom + (i + 0.5) * SUB_H, **kw) for i in range(n)])

    @classmethod
    def endfire(cls, n, spacing, y=0.0, x_front=0.0, aim_deg=0.0, **kw):
        """n caisses en enfilade, retardees pour sommer dans la direction aim_deg.

        aim_deg = 0 : tir vers +x. L'axe de l'enfilade et l'orientation des caisses
        tournent ensemble. La caisse la plus en arriere joue en premier (retard
        nul) ; chaque caisse plus en avant est retardee du temps de parcours qui
        la separe, de sorte que tous les fronts d'onde partent ensemble vers
        l'avant et se soustraient vers l'arriere.

        (x_front, y) est la position de la caisse la plus avancee.
        """
        a = np.deg2rad(aim_deg)
        ca, sa = np.cos(a), np.sin(a)
        subs = []
        for i in range(n):
            back = (n - 1 - i) * spacing        # recul le long de l'axe de tir
            subs.append(Sub(x_front - back * ca, y - back * sa, SUB_H / 2,
                            yaw_deg=aim_deg,
                            delay_ms=i * spacing / C * 1e3, **kw))
        return cls(subs)

    @classmethod
    def cardioid(cls, spacing, y=0.0, x_front=0.0, **kw):
        """Paire cardioide : caisse arriere inversee et retardee de spacing / c.

        Rejection vers l'arriere (-x). Spacing usuel : quart d'onde a la frequence
        de calage.
        """
        return cls([Sub(x_front, y, SUB_H / 2, **kw),
                     Sub(x_front - spacing, y, SUB_H / 2,
                         delay_ms=spacing / C * 1e3, polarity=-1, **kw)])

    @classmethod
    def arc_physical(cls, n, radius, spacing=SUB_W, x_front=0.0, y_center=0.0, **kw):
        """Arc SUB PHYSIQUE : n caisses posees sur un arc convexe vers +x.

        Les caisses sont reparties avec un pas de `spacing` le long de l'arc ; les
        caisses exterieures reculent donc geometriquement ET pivotent vers
        l'exterieur. Le centre de courbure est derriere l'array, en x_front - radius.
        """
        d_ang = spacing / radius
        angles = (np.arange(n) - (n - 1) / 2) * d_ang
        return cls([Sub(x_front + radius * (np.cos(a) - 1.0),
                        y_center + radius * np.sin(a),
                        SUB_H / 2, yaw_deg=np.rad2deg(a), **kw)
                    for a in angles])

    @classmethod
    def arc_virtual(cls, n, radius, spacing=SUB_W, x=0.0, y_center=0.0, **kw):
        """Arc SUB VIRTUEL : n caisses alignees suivant y, courbure obtenue par delai.

        Chaque caisse est retardee du temps de parcours du recul qu'elle aurait sur
        un arc physique de meme rayon : radius - sqrt(radius^2 - dy^2).

        A la difference de l'arc physique, les caisses restent toutes face a +x :
        le delai reproduit le recul, jamais le pivotement.
        """
        subs = cls.line(n, spacing=spacing, x=x, y_center=y_center, **kw)
        out = []
        for s in subs:
            dy = s.y - y_center
            recul = radius - np.sqrt(max(radius**2 - dy**2, 0.0))
            out.append(replace(s, delay_ms=s.delay_ms + recul / C * 1e3))
        return cls(out)

    @classmethod
    def cardioid_column(cls, n=4, index=1, x=0.0, y=0.0, z_bottom=0.0):
        """Colonne de n caisses empilees, dont une passee en cardioide."""
        subs = list(cls.stack(n, x=x, y=y, z_bottom=z_bottom))
        subs[index] = subs[index].reversed()
        return cls(subs)

    @classmethod
    def cardioid_row(cls, n=4, index=1, x=0.0, y_center=0.0, spacing=SUB_W):
        """Rangee de n caisses au sol cote a cote, dont une passee en cardioide."""
        subs = list(cls.line(n, spacing=spacing, x=x, y_center=y_center))
        subs[index] = subs[index].reversed()
        return cls(subs)

    @classmethod
    def endfire_rows(cls, n_rows, n_per_row, spacing, y_center=0.0, x_front=0.0,
                     aim_deg=0.0, row_pitch=SUB_W, **kw):
        """n_rows lignes end-fire paralleles, decalees perpendiculairement au tir."""
        a = np.deg2rad(aim_deg)
        ca, sa = np.cos(a), np.sin(a)
        offs = (np.arange(n_rows) - (n_rows - 1) / 2) * row_pitch
        subs = []
        for o in offs:
            # decalage le long de la perpendiculaire a l'axe de tir : (-sa, ca)
            subs += list(cls.endfire(n_per_row, spacing, aim_deg=aim_deg,
                            x_front=x_front - float(o) * sa,
                            y=y_center + float(o) * ca, **kw))
        return cls(subs)

    @classmethod
    def pinwheel_cluster(cls, n_col=4, R=1.0, n_el=4, dz=SUB_H, h_s=8.0,
                         x_center=0.0, y_center=0.0, pinwheel_offset_deg=0.0, **kw):
        """Grappe centrale volee : n_col colonnes verticales autour d'un axe commun.

        Geometrie (vue de dessus, colonne k = 0..n_col-1) :
          theta_k = k * 360 / n_col
          position azimutale de la colonne : theta_k + pinwheel_offset_deg
          centre de la colonne : (x_center + R cos, y_center + R sin) sur ce cercle
          orientation (yaw) de toutes les caisses de la colonne : theta_k

        Avec pinwheel_offset_deg = 0 chaque colonne regarde radialement vers
        l'exterieur (disposition en croix). L'offset decale la position azimutale
        par rapport a l'axe de rayonnement : a 90 deg les colonnes regardent
        tangentiellement et s'emboitent en moulinet.

        Dans ce modele, une caisse est UN monopole (voir Monopole), sans directivite
        propre : un monopole rayonne identiquement dans toutes les directions, quel
        que soit yaw_deg. theta_k, via yaw_deg, ne fait donc PAS tourner un lobe de
        rayonnement -- il decale seulement la position de la source de SUB_D/2
        (0.4 m) devant le centre geometrique de la caisse, dans la direction
        theta_k (voir Monopole._sources). (Une version anterieure du modele,
        abandonnee, representait la caisse par plusieurs monopoles discretisant une
        face avant directive, pour laquelle cette remarque etait vraie au sens
        plein ; ce n'est plus le modele utilise ici.)

        n_el caisses par colonne, pas vertical dz, centre de la grappe a h_s
        au-dessus du sol.

        HYPOTHESE DE VALIDITE : l'omnidirectionnalite azimutale suppose que la
        distance entre colonnes adjacentes d = 2 R sin(pi / n_col) reste inferieure
        a lambda / 2 a la frequence haute de la bande. Au-dela, des lobes de reseau
        entrent dans le domaine visible. Voir check_grating_lobes().
        """
        subs = []
        # colonne centree verticalement sur h_s
        z_offsets = (np.arange(n_el) - (n_el - 1) / 2) * dz
        for k in range(n_col):
            theta = k * 360.0 / n_col
            phi = np.deg2rad(theta + pinwheel_offset_deg)
            xc = x_center + R * np.cos(phi)
            yc = y_center + R * np.sin(phi)
            for zo in z_offsets:
                subs.append(Sub(xc, yc, h_s + float(zo), yaw_deg=theta, **kw))
        return cls(subs)

    @classmethod
    def endfire_from_rear(cls, n, spacing, aim_deg, rear_x, rear_y, **kw):
        """End-fire ancre par sa caisse ARRIERE, utile pour poser un array lateral
        contre le bord d'un bloc existant sans que les deux se chevauchent."""
        subs = cls.endfire(n, spacing, aim_deg=aim_deg, **kw)
        rear = subs[0]                      # la caisse arriere = retard nul
        return cls([s.moved(dx=rear_x - rear.x, dy=rear_y - rear.y) for s in subs])

    def check_grating_lobes(self, n_col, R, f_high, label=""):
        """Avertit si l'espacement entre colonnes adjacentes depasse lambda/2.

        Retourne (ok, d, f_limite). Au-dessus de f_limite = c / (2 d), des lobes de
        reseau apparaissent dans le domaine visible et l'hypothese
        d'omnidirectionnalite azimutale tombe.
        """
        d = _column_spacing(n_col, R)
        f_limit = C / (2.0 * d)
        ok = f_high <= f_limit
        if ok:
            print(f"[OK]      {label} d = {d:.2f} m entre colonnes, lambda/2 "
                  f"atteint a {f_limit:.0f} Hz > f_max = {f_high:.0f} Hz.")
        else:
            print(f"[ATTENTION] {label} d = {d:.2f} m entre colonnes > lambda/2 "
                  f"des {f_limit:.0f} Hz, or la bande monte a {f_high:.0f} Hz.\n"
                  f"            Des lobes de reseau apparaissent en azimut au-dela "
                  f"de {f_limit:.0f} Hz : l'hypothese d'omnidirectionnalite n'est "
                  f"plus valable sur le haut de la bande.\n"
                  f"            Reduire R sous {C / (2.0 * f_high) / (2.0 * np.sin(np.pi / n_col)):.2f} m "
                  f"ou limiter la bande a {f_limit:.0f} Hz.")
        return ok, d, f_limit


def _column_spacing(n_col, R):
    """Distance entre deux colonnes adjacentes sur le cercle de rayon R."""
    return 2.0 * R * np.sin(np.pi / n_col)


def _axis(lo, hi, step):
    """Axe inclusif de lo a hi par pas de step, sans jamais depasser hi.

    np.arange(lo, hi + step/2, step) seul peut deborder au-dela de hi quand
    step est strictement compris entre (hi - lo) et 2*(hi - lo) (ex. hi=10,
    step=15 -> point a 15). On filtre donc les valeurs > hi + tolerance
    (tolerance de l'ordre de step*1e-9, pour absorber l'arrondi flottant qui
    justifiait le + step/2 au depart). Si hi n'est pas un multiple de step a
    partir de lo, l'axe s'arrete logiquement avant hi (ex. lo=0, hi=10,
    step=0.3 -> dernier point 9.9) : c'est attendu, on ne force jamais hi
    dans l'axe.
    """
    ax = np.arange(lo, hi + step / 2, step)
    return ax[ax <= hi + step * 1e-9]


@dataclass(frozen=True)
class Zone:
    """Ensemble immuable de points d'observation ou calculer la pression.

    Une zone est une surface quelconque de l'espace, decrite par trois
    tableaux aplatis X, Y, Z de meme taille. Une grille horizontale, une
    coupe verticale ou un cercle de releve ne sont que des facons
    particulieres de remplir ces trois tableaux ; le moteur de calcul n'a
    besoin que d'une seule formule, appliquee point a point sur (X, Y, Z).

    Immuabilite : seules les fabriques ci-dessous (grid_xy, slice_xz,
    slice_yz, pit, ring, points) garantissent que X, Y, Z sont proteges en
    ecriture et sans aliasing avec un tableau de l'appelant. Le constructeur
    brut `Zone(name, X, Y, Z, shape)` applique bien setflags(write=False) a
    X, Y, Z (__post_init__), mais ne les copie pas : si on lui passe une vue
    d'un tableau externe, modifier ce tableau externe modifie la Zone malgre
    le setflags (le flag ne proroge pas le tableau de base). Pour une Zone
    sans aucun risque d'aliasing, passer par les fabriques ou copier
    explicitement avant construction.
    """
    name: str
    X: np.ndarray          # (n_pts,) aplati
    Y: np.ndarray          # (n_pts,)
    Z: np.ndarray          # (n_pts,)
    shape: tuple           # forme d'origine, pour remodeler a l'affichage

    def __post_init__(self):
        for arr in (self.X, self.Y, self.Z):
            arr.setflags(write=False)

    @property
    def n_points(self):
        return self.X.size

    def reshape(self, values):
        """Redonne a `values` (aplati, n_points) la forme d'origine de la zone."""
        return values.reshape(self.shape)

    # FABRIQUES

    @classmethod
    def grid_xy(cls, xlim, ylim, step, z=Z_LISTENER, name="grid"):
        """Grille horizontale a hauteur z constante. Bornes incluses, jamais depassees."""
        x = _axis(xlim[0], xlim[1], step)
        y = _axis(ylim[0], ylim[1], step)
        XX, YY = np.meshgrid(x, y)
        ZZ = np.full_like(XX, float(z))
        return cls(name, XX.ravel(), YY.ravel(), ZZ.ravel(), XX.shape)

    @classmethod
    def slice_xz(cls, xlim, zlim, step, y=0.0, name="coupe xz"):
        """Plan vertical a y constant (coupe suivant x et z). Bornes incluses, jamais depassees."""
        x = _axis(xlim[0], xlim[1], step)
        z = _axis(zlim[0], zlim[1], step)
        XX, ZZ = np.meshgrid(x, z)
        YY = np.full_like(XX, float(y))
        return cls(name, XX.ravel(), YY.ravel(), ZZ.ravel(), XX.shape)

    @classmethod
    def slice_yz(cls, ylim, zlim, step, x=0.0, name="coupe yz"):
        """Plan vertical a x constant (coupe suivant y et z). Bornes incluses, jamais depassees."""
        y = _axis(ylim[0], ylim[1], step)
        z = _axis(zlim[0], zlim[1], step)
        YY, ZZ = np.meshgrid(y, z)
        XX = np.full_like(YY, float(x))
        return cls(name, XX.ravel(), YY.ravel(), ZZ.ravel(), YY.shape)

    @classmethod
    def pit(cls, x=(3.0, 50.0), half_width=25.0, step=0.5, z=Z_LISTENER):
        """La fosse du projet : grille horizontale entre x et +/- half_width."""
        return cls.grid_xy(x, (-half_width, half_width), step, z=z, name="fosse")

    @classmethod
    def ring(cls, radius, n_az=181, z=Z_LISTENER):
        """Cercle de rayon radius a hauteur z, pour les releves de directivite.

        Azimuts de 0 a 2*pi INCLUS (np.linspace(0, 2*pi, n_az) inclut les
        deux bornes) : le premier et le dernier point sont donc confondus a
        l'azimut 0. C'est VOULU, pour reproduire a l'identique
        `plot_azimuthal_directivity` de Codes/1ere config sub.py, qui
        construit son axe azimutal de la meme facon avec n_az=181 — une
        tache ulterieure doit reproduire cette figure. Ne pas passer en
        endpoint=False pour "corriger" la duplication : cela ferait diverger
        le resultat de l'original. Attention en aval : une moyenne azimutale
        calculee naivement sur ces points surpondere l'azimut 0 (compte deux
        fois sur n_az-1 intervalles reels).
        """
        az = np.linspace(0.0, 2.0 * np.pi, n_az)
        X = radius * np.cos(az)
        Y = radius * np.sin(az)
        Z = np.full_like(X, float(z))
        return cls(f"ring {radius:.0f} m", X, Y, Z, (n_az,))

    @classmethod
    def points(cls, xyz, name="points"):
        """Points explicites, depuis un tableau (n, 3) de colonnes x, y, z."""
        a = np.array(xyz, dtype=np.float64)
        return cls(name, a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy(),
                   (a.shape[0],))


def evaluation_mask(zone, array, margin=0.5):
    """Faux a moins de `margin` du gabarit d'une caisse (demi-largeur SUB_W/2
    en y, demi-profondeur SUB_D/2 en x), orientation yaw prise en compte.

    Note: negative margin values are accepted and shrink the footprint silently,
    potentially eliminating exclusion entirely without warning."""
    keep = np.ones(zone.n_points, dtype=bool)
    for s in array:
        a = np.deg2rad(s.yaw_deg)
        ca, sa = np.cos(a), np.sin(a)
        dx = (zone.X - s.x) * ca + (zone.Y - s.y) * sa     # repere de la caisse
        dy = -(zone.X - s.x) * sa + (zone.Y - s.y) * ca
        keep &= ~((np.abs(dx) <= SUB_D / 2 + margin)
                  & (np.abs(dy) <= SUB_W / 2 + margin))
    return keep


class DelayGain:
    """Pilotage classique : q entierement determine par delay/gain/polarite."""

    def q(self, array, model, freqs):
        """Calcule le poids complexe applique a chaque caisse a chaque frequence.

        Parameters
        ----------
        array : SubArray
            L'array de caisses.
        model : unused
            Parametre reserve pour l'interface commune avec d'autres strategies.
        freqs : array-like or scalar
            Frequences en Hz. Accepte un scalaire, une liste ou un tableau.

        Returns
        -------
        np.ndarray
            Tableau de forme (n_freq, n_box) et dtype complex128,
            contenant le poids complexe pour chaque combinaison
            (frequence, caisse).
        """
        f = np.atleast_1d(np.asarray(freqs, dtype=float))
        return (array.weights[None, :]
                * np.exp(-2j * np.pi * f[:, None]
                         * array.delays[None, :]))


def DelayGainQ(array, freqs):
    """Utilitaire raccourci : DelayGain().q(array, None, freqs)."""
    return DelayGain().q(array, None, freqs)


@dataclass(frozen=True)
class Monopole:
    """Modele acoustique : une caisse = UN monopole (plus son image de sol),
    place au centre de sa face avant.

    ground_R : coefficient de reflexion du sol (1.0 = sol rigide parfait,
        0.0 = sol parfaitement absorbant ; l'image de sol est alors annulee).
    normalization : "flat" (defaut) ou "physical".
        "flat" : poids 1.0, independant de la frequence. Un subwoofer a
            reponse plate rayonne une pression constante a debit variable
            avec la frequence ; c'est l'hypothese pertinente en large bande,
            d'ou le defaut.
        "physical" : monopole acoustique exact, poids 1j*rho*c*k*Q/(4*pi).
            Correct en mono-frequence (c'est la formule de Codes/, qui
            travaille a 80 Hz), mais 1j*rho*c*k est proportionnel a la
            frequence : a debit volumique constant Q, cette normalisation
            introduit +6 dB/octave, ce qui est faux en large bande. A
            reserver a la fidelite de portage face a Codes/.
    r_min : garde-fou anti-singularite [m] (0.0 = desactivee). Les distances
        sont plafonnees par np.maximum AVANT le calcul de l'exponentielle,
        comme dans Codes/. Avec r_min = 0.0 (defaut), un point d'observation
        confondu avec une source produit NaN et non une exception : seul un
        RuntimeWarning est emis (division par une distance nulle). Le NaN ne
        contamine que les points concernes, mais rien ne le signale ensuite
        (une carte SPL avec quelques NaN se trace sans avertissement). Poser
        r_min > 0 pour s'en premunir.
    """
    ground_R: complex = 1.0
    normalization: str = "flat"
    r_min: float = 0.0

    def _sources(self, sub):
        """Position de l'unique source de la caisse : centre de sa face
        avant, a une demi-profondeur SUB_D/2 devant le centre geometrique,
        suivant le yaw. Renvoie un tableau (1, 3)."""
        a = np.deg2rad(sub.yaw_deg)
        return np.array([[sub.x + (SUB_D / 2) * np.cos(a),
                          sub.y + (SUB_D / 2) * np.sin(a),
                          sub.z]])

    def _distances(self, sub, zone):
        """Distances source directe / image de sol vers chaque point de la
        zone, (n_points,) chacune. L'image de sol est le symetrique par
        z = 0, d'ou `zone.Z + zs`. Plafonnees par r_min AVANT l'exponentielle
        (fidelite bit-a-bit avec Codes/)."""
        xs, ys, zs = self._sources(sub)[0]
        d2 = (zone.X - xs) ** 2 + (zone.Y - ys) ** 2
        r_direct = np.sqrt(d2 + (zone.Z - zs) ** 2)
        r_image = np.sqrt(d2 + (zone.Z + zs) ** 2)
        if self.r_min > 0.0:
            r_direct = np.maximum(r_direct, self.r_min)
            r_image = np.maximum(r_image, self.r_min)
        return r_direct, r_image

    def _per_source(self, r_direct, r_image, k):
        """H(f, point) pour une seule source (directe + image de sol),
        (n_freq, n_points) complexe.

        normalization="physical" reproduit terme a terme
        1j*rho*c*k*Q*exp(-1j*k*r)/(4*pi*r) de Codes/Sub solo.py : meme
        regroupement des operations, 4*np.pi garde au denominateur plutot
        qu'en facteur precalcule.
        """
        kr_d = k[:, None] * r_direct[None, :]
        kr_i = k[:, None] * r_image[None, :]
        if self.normalization == "physical":
            numer = 1j * RHO * C * k * Q_SOURCE
            H_direct = numer[:, None] * np.exp(-1j * kr_d) / (4 * np.pi * r_direct[None, :])
            H_image = numer[:, None] * np.exp(-1j * kr_i) / (4 * np.pi * r_image[None, :])
        elif self.normalization == "flat":
            H_direct = np.exp(-1j * kr_d) / r_direct[None, :]
            H_image = np.exp(-1j * kr_i) / r_image[None, :]
        else:
            raise ValueError(f"normalization inconnue : {self.normalization!r}")
        return H_direct + self.ground_R * H_image

    def field(self, array, q, zone, freqs):
        """Champ de pression (n_freq, n_points) complexe.

        Accumulation en flux, caisse par caisse (boucle a l'exterieur,
        frequences vectorisees a l'interieur) : la matrice de transfert
        complete (n_freq, n_points, n_box) n'est jamais materialisee. Sur une
        grille d'affichage elle peserait plusieurs centaines de Mo ; c'est
        une contrainte memoire, pas de style.
        """
        f = np.atleast_1d(np.asarray(freqs, dtype=float))
        k = 2 * np.pi * f / C
        p = np.zeros((f.size, zone.n_points), dtype=np.complex128)
        for l, sub in enumerate(array):
            r_direct, r_image = self._distances(sub, zone)
            H = self._per_source(r_direct, r_image, k)
            p += q[:, l][:, None] * H
        return p

    def transfer_matrix(self, array, zone, freqs):
        """Matrice de transfert (n_freq, n_points, n_box) complexe :
        G[j, k, l] = pression au point k, frequence j, pour une excitation
        unitaire de la seule caisse l. Reservee aux zones de controle, ou le
        nombre de points est petit : pour une grille d'affichage, utiliser
        field()."""
        f = np.atleast_1d(np.asarray(freqs, dtype=float))
        k = 2 * np.pi * f / C
        n_box = len(array)
        G = np.empty((f.size, zone.n_points, n_box), dtype=np.complex128)
        for l, sub in enumerate(array):
            r_direct, r_image = self._distances(sub, zone)
            G[:, :, l] = self._per_source(r_direct, r_image, k)
        return G


def spl_from_field(p):
    """Niveau SPL (n_points,) float64 : moyenne energetique large bande du
    champ de pression complexe p (n_freq, n_points)."""
    return 20 * np.log10(np.sqrt(np.mean(np.abs(p) ** 2, axis=0)) + 1e-12)


@dataclass
class Simulation:
    """Facade unique assemblant SubArray + modele acoustique + strategie de
    pilotage + bande de frequences.

    Les defauts (Monopole(), DelayGain(), BAND) sont poses dans
    __post_init__, jamais dans la signature : un objet mutable donne comme
    valeur par defaut d'un champ de dataclass est construit une seule fois,
    a la definition de la classe, et serait alors partage par toutes les
    instances qui n'en precisent pas un autre.
    """
    array: SubArray
    model: object = None
    drive: object = None
    freqs: np.ndarray = None

    def __post_init__(self):
        if self.model is None:
            self.model = Monopole()
        if self.drive is None:
            self.drive = DelayGain()
        if self.freqs is None:
            self.freqs = BAND

    def field(self, zone):
        """Champ de pression (n_freq, n_points) complexe sur `zone`."""
        q = self.drive.q(self.array, self.model, self.freqs)
        return self.model.field(self.array, q, zone, self.freqs)

    def spl(self, zone):
        """Niveau SPL (n_points,) dB sur `zone`."""
        return spl_from_field(self.field(zone))

    def run(self, zones, plot=False, save_dir=None):
        """Calcule le SPL de chaque zone et renvoie un Result.

        plot, save_dir : acceptes mais ignores pour l'instant. Le trace
        (figures, sauvegarde eventuelle dans save_dir) est honore par une
        tache ulterieure ; ce parametre est fige des maintenant pour
        stabiliser l'interface publique.
        """
        spl = {z.name: self.spl(z) for z in zones}
        q = self.drive.q(self.array, self.model, self.freqs)
        zones_by_name = {z.name: z for z in zones}
        return Result(spl, q, zones_by_name, self.array)


def sub_footprint(sub):
    """Contour de la caisse vue de dessus, oriente suivant yaw_deg.

    Porte depuis Codes/1ere config sub.py:521-528 (sub_footprint), sans
    modification de la geometrie : memes coins relatifs (+/- SUB_D/2,
    +/- SUB_W/2), meme rotation par yaw_deg."""
    a = np.deg2rad(sub.yaw_deg)
    ca, sa = np.cos(a), np.sin(a)
    corners = [(-SUB_D / 2, -SUB_W / 2), (SUB_D / 2, -SUB_W / 2),
               (SUB_D / 2, SUB_W / 2), (-SUB_D / 2, SUB_W / 2)]
    return [(sub.x + dx * ca - dy * sa, sub.y + dx * sa + dy * ca)
            for dx, dy in corners]


def _box_bbox_xy(sub):
    """Rectangle (axes x, y) englobant l'empreinte reelle de `sub`
    (sub_footprint, deja validee), pour les traces qui ne savent dessiner que
    des paves/rectangles alignes sur les axes (bar3d en 3D, Rectangle sur une
    coupe) : (xmin, xmax, ymin, ymax).

    Portee depuis les `draw_sub_box` des 7 scripts de Codes/ (Sub solo.py,
    Horizontal array.py, Vertical array.py, End-fire.py, Horizontal
    cardioid.py, Vertical cardioid.py, Gradient end-fire 1.py), avec UNE
    DIFFERENCE DELIBEREE signalee dans le rapport de la tache 15 : ces 7
    scripts appellent tous `ax.bar3d(..., sub_w, sub_d, sub_h)` avec
    sub_w = 1.0 en etendue suivant x et sub_d = 0.8 suivant y -- soit
    l'INVERSE de la convention deja employee par `sub_footprint` dans ce
    module (et validee face a Codes/1ere config sub.py, qui a son propre
    sub_footprint strictement identique au notre) : SUB_D = 0.8 suivant x,
    SUB_W = 1.0 suivant y, a yaw = 0. Reproduire litteralement les 7 scripts
    aurait dessine, pour une meme caisse, un pave 3D en contradiction avec
    l'empreinte 2D que plot_map/plot_configs tracent deja pour cette caisse
    (sub_footprint, elle, n'est pas modifiee). On a donc choisi la coherence
    interne du module plutot que la fidelite litterale a ces 7 scripts sur
    ce point precis des dimensions de caisse.

    Prendre l'englobant de l'empreinte (plutot que SUB_D/SUB_W bruts) a aussi
    pour effet de generaliser correctement a une caisse tournee (yaw != 0,
    180) : aucun des 7 scripts n'avait a traiter ce cas, leurs scenes 3D
    n'incluant jamais de caisse pivotee."""
    corners = sub_footprint(sub)
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), max(xs), min(ys), max(ys)


def plot_3d_scene(array, z_obs=Z_LISTENER, slice_plane=None, box_size=SUB_H):
    """Scene 3D : caisses en CUBES, plan du sol, plan d'observation a
    `z_obs`, et plan de coupe optionnel. Fonction de module (comme
    `plot_configs`), alors que le plan initial de la tache 15 la nommait
    `Result.plot_3d_scene` -- ECART ASSUME et documente dans le rapport de
    la tache 15 : comme `plot_configs`, elle ne depend d'aucun Result
    particulier (juste d'un `array`), donc n'a pas a en etre une methode ;
    un seul portage contre les 7 copies quasi identiques de `draw_3d_scene`
    + `draw_sub_box` dans Codes/.

    `slice_plane` : None, ou une Zone issue de Zone.slice_xz / Zone.slice_yz
    -- seules son orientation (plan x = cste ou y = cste, lu sur les
    coordonnees de la Zone, pas sur son nom) et son etendue en z servent ici ;
    sa resolution (step) est sans effet sur le trace. Comme les 7 scripts
    (qui tracent toujours leur plan de coupe bleu dans la scene 3D), mais
    rendu optionnel : certains appels (catalogue sans coupe associee)
    n'ont pas de plan a dessiner.

    Dimensions des caisses -- DECISION DE MARTIN (ronde de correction 1,
    tache 15) : chaque caisse est un CUBE de cote `box_size` (par defaut
    SUB_H = 0.6 m), centre sur (sub.x, sub.y, sub.z). Remplace le pave
    SUB_D x SUB_W x SUB_H (englobant de `_box_bbox_xy`) utilise dans une
    version precedente : vu comme des monopoles, l'encombrement reel des
    caisses n'a pas d'effet acoustique dans ce modele (seul l'espacement
    entre monopoles en a, et les caisses d'un meme array physique sont
    considerees collees -- sauf en end-fire, ou l'espacement est regle par
    rapport a la longueur d'onde visee pour la rejection arriere, pas par la
    largeur de caisse). Le volume dessine n'est donc qu'un repere visuel, et
    un cube n'a pas d'orientation : la divergence SUB_W/SUB_D entre
    `sub_footprint` et les bar3d de Codes/ (signalee dans le rapport de la
    tache 15) devient sans objet pour ce trace.

    Pourquoi SUB_H = 0.6 m et pas SUB_W = 1.0 m par defaut : le pas vertical
    d'un empilement (SubArray.stack) est SUB_H. Des cubes de 1.0 m
    s'interpenetreraient dans un empilement (pas 0.6 m < cote 1.0 m), ce qui
    se lirait comme un bug de rendu. Avec SUB_H = 0.6 m, aucun empilement ne
    se chevauche jamais ; les arrays horizontaux (pas = SUB_W = 1.0 m)
    montrent alors un leger espace entre cubes, qui se lit correctement
    comme un espace et non comme une erreur.

    `_box_bbox_xy` reste utilisee ailleurs (contours 2D de
    `Result.plot_slice`) : seul ce rendu 3D devient cubique.

    z_obs = Z_LISTENER par defaut -- la MEME constante que le defaut de
    Zone.grid_xy/Zone.pit/Zone.ring, donc exactement le plan sur lequel la
    carte horizontale de la meme famille est calculee. Les 7 scripts de
    Codes/ affichaient 1.5 m ici alors qu'ils calculaient a 1.8 m : garder
    cet ecart aurait dessine un plan d'observation qui n'est pas celui des
    donnees. Voir Z_LISTENER."""
    if len(array):
        xs = np.array([s.x for s in array])
        ys = np.array([s.y for s in array])
        z_top = np.array([s.z + box_size / 2 for s in array])
    else:
        xs = ys = np.array([0.0])
        z_top = np.array([box_size])

    pad = 5.0  # meme marge que les 7 scripts (xlim/ylim = +/-5 m autour de la source)
    xlim = (float(xs.min()) - pad, float(xs.max()) + pad)
    ylim = (float(ys.min()) - pad, float(ys.max()) + pad)
    zlim = (0.0, max(14.0, float(z_top.max()) + 2.0))  # 14 m : meme z_max que les 7 scripts

    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(projection="3d")

    gx = np.array([[xlim[0], xlim[1]], [xlim[0], xlim[1]]])
    gy = np.array([[ylim[0], ylim[0]], [ylim[1], ylim[1]]])
    ax.plot_surface(gx, gy, np.zeros_like(gx), color="lightgray", alpha=0.25, linewidth=0)
    ax.plot_surface(gx, gy, np.full_like(gx, float(z_obs)), color="limegreen",
                    alpha=0.2, linewidth=0)

    if slice_plane is not None:
        zz = np.array([[zlim[0], zlim[0]], [zlim[1], zlim[1]]])
        if np.ptp(slice_plane.Y) == 0.0:          # plan y = cste (coupe xz)
            y0 = float(slice_plane.Y[0])
            sx = np.array([[xlim[0], xlim[1]], [xlim[0], xlim[1]]])
            sy = np.full_like(sx, y0)
        else:                                      # plan x = cste (coupe yz)
            x0 = float(slice_plane.X[0])
            sy = np.array([[ylim[0], ylim[1]], [ylim[0], ylim[1]]])
            sx = np.full_like(sy, x0)
        ax.plot_surface(sx, sy, zz, color="blue", alpha=0.2, linewidth=0)

    for sub in array:
        half = box_size / 2
        ax.bar3d(sub.x - half, sub.y - half, sub.z - half, box_size, box_size,
                 box_size, color="white", edgecolor="black", shade=False)

    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_zlabel(r"$z$ [m]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax.view_init(elev=22, azim=-58)
    return fig


def compare_ground_flown(array_factory, z_ground, z_flown, zones, model=None,
                         freqs=None, shared_reference=False):
    """Calcule et renvoie (result_ground, result_flown) : le MEME array,
    place une fois au sol (bas de la caisse la plus basse a `z_ground`) et
    une fois vole (a `z_flown`), sur les memes `zones`.

    Systematise la comparaison que les 7 scripts de Codes/ font chacun a la
    main entre leurs deux cas z_ground/z_flown -- une seule fonction contre
    sept copies quasi identiques.

    `array_factory` : appelable prenant une hauteur (float) et renvoyant un
    SubArray place a cette hauteur. Typiquement `lambda h: base.flown(h)` :
    SubArray.flown(h) est deja l'operation qui place le bas de la caisse la
    plus basse a la hauteur h (y compris h = 0, un array deja au sol restant
    au sol : voir SubArray.flown).

    `model`, `freqs` : transmis a Simulation (defauts Monopole()/BAND si
    None, comme Simulation.__post_init__). Pas de parametre `drive` : les 7
    scripts ne font varier que la geometrie (hauteur), jamais le pilotage,
    entre leurs deux cas.

    Result.spl reste l'objet renvoye par Simulation.run() -- le SPL absolu,
    STRICTEMENT NON MODIFIE : cette fonction ne touche a aucune formule et ne
    reecrit rien dans les dict `spl`. La reference de normalisation est
    calculee en PLUS et simplement deposee sur chaque Result, en attribut
    `ref` (hors contrat __init__ de Result, ajoute ici par simple affectation
    -- Result n'a pas de __slots__) : `result_ground.ref`, `result_flown.ref`.
    L'appelant trace alors `result.spl[zone.name] - result.ref`.

    REFERENCE DE NORMALISATION -- DECISION DE MARTIN (tache 15, revient sur
    la consigne initiale du plan) : par defaut (`shared_reference=False`),
    reference PAR CARTE, calculee separement pour chaque cas -- exactement
    `ref_ground` et `ref_flown` de Codes/Sub solo.py (et des 6 autres
    scripts) : le maximum du SPL sur la zone d'EVALUATION (evaluation_mask,
    donc hors voisinage des caisses) de `zones[0]`, recalcule independamment
    pour le cas au sol et pour le cas vole. C'est le comportement fidele, et
    celui attendu par defaut : chaque carte exploite alors toute la dynamique
    de l'echelle de couleur pour elle-meme, ce qui est la bonne lecture
    quand on regarde une configuration SEULE.

    `zones[0]` et non chaque zone de `zones` : comme les 7 scripts, qui
    calculent leur ref UNE fois sur la grille horizontale puis la reutilisent
    pour la coupe verticale de la MEME configuration -- l'appelant doit donc
    passer en premier, dans `zones`, la zone de reference (typiquement la
    grille d'affichage horizontale).

    `shared_reference=True` : les deux Result recoivent la MEME reference,
    max(ref_ground, ref_flown) -- a utiliser uniquement pour une planche
    sol/vole a deux panneaux partageant une echelle de couleur commune
    (via plot_configs) : sans cela, les deux cartes culminent chacune a
    0 dB et l'ecart de niveau du vol disparait du graphique, precisement ce
    qu'une telle planche doit montrer. Ne change rien a une lecture
    independante des cartes (plot_map/plot_slice, qui renormalisent de toute
    facon chacun a son propre maximum -- voir leurs docstrings : le choix de
    reference fait ici n'a donc d'effet visible que pour une figure
    multi-panneaux a echelle partagee construite directement a partir de
    `result.spl[...] - result.ref`.
    """
    array_ground = array_factory(z_ground)
    array_flown = array_factory(z_flown)
    result_ground = Simulation(array_ground, model=model, freqs=freqs).run(zones)
    result_flown = Simulation(array_flown, model=model, freqs=freqs).run(zones)

    ref_zone_name = zones[0].name
    mask_ground = evaluation_mask(zones[0], array_ground)
    mask_flown = evaluation_mask(zones[0], array_flown)
    spl_ground = result_ground.spl[ref_zone_name]
    spl_flown = result_flown.spl[ref_zone_name]
    ref_ground = float(np.max(spl_ground[mask_ground]))
    ref_flown = float(np.max(spl_flown[mask_flown]))
    if shared_reference:
        ref_ground = ref_flown = max(ref_ground, ref_flown)

    result_ground.ref = ref_ground
    result_flown.ref = ref_flown
    return result_ground, result_flown


def _slug(name):
    """Nom de fichier pour une figure : meme filtre, au caractere pres, que
    l'ancien `run_figures.py:21` (script de tracage supprime a la tache 11,
    son role etant repris par le catalogue sous __main__) -- conserve
    alphanumeriques, espaces, tirets et underscores, tronque a 60
    caracteres, puis strip(). Pas de minuscule
    forcee, pas de substitution par underscore : ce sont des noms de
    fichiers lisibles par Martin pour comparer les figures d'une version a
    l'autre, pas des identifiants normalises."""
    kept = "".join(ch if ch.isalnum() or ch in " -_" else "" for ch in name)
    return kept[:60].strip()


def plot_configs(zone, configs_meta, ncols=1, title=None,
                 cbar_label="Level relative to 1-sub maximum [dB]"):
    """Carte(s) SPL relative sur `zone`. Porte depuis `plot_configs`
    (Codes/1ere config sub.py:575-628) ; fonction de module (comme dans
    l'original), pas une methode : elle ne depend d'aucun Result particulier,
    seulement de la zone partagee et des donnees fournies par l'appelant.

    configs_meta : liste de (titre_panneau, spl_relatif, array) pour tracer
    plusieurs configurations cote a cote dans une seule figure (reproduit le
    regroupement en figures de l'original, ou chaque FIGURES[i] rassemble
    plusieurs Config dans la meme grille). spl_relatif est deja exprime en
    dB relatifs a la reference choisie par l'appelant (spl - ref), aplati
    comme zone.X/Y ; array est le SubArray de cette configuration, pour
    dessiner son empreinte au sol.

    Parametres visuels repris SANS MODIFICATION par rapport a l'original :
    levels = np.arange(-30, 22, 2), cmap="viridis", pointille pour les
    caisses volees / trait plein pour celles au sol, et la reduction de
    police du titre quand il est trop long. Seules les bornes des axes
    (xlim/ylim) et les graduations sont adaptees : elles suivaient le
    grillage global fixe de l'original (x in [-40,40], y in [-20,20]) ; ici
    elles suivent les bornes reelles de `zone`, puisque la grille d'affichage
    est etendue (tache 11, grid_xy((-40, 55), (-30, 30))).

    `cbar_label` -- AJOUTE par la ronde de correction 3 (tache 15, constat
    2) : l'intitule de la barre de couleur etait code en dur a "Level
    relative to 1-sub maximum [dB]", exact pour le catalogue principal
    (chaque spl_relatif y est reellement rapporte au maximum du cas 1 sub,
    voir `reference_level` sous __main__) mais FAUX pour toute carte dont la
    reference est une autre chose -- par exemple Result.plot_map, dont la
    reference est le maximum de la carte elle-meme. Le defaut reste le texte
    d'origine ; l'appelant passe un intitule different quand sa reference est
    differente (voir Result.plot_map).

    Les planches sol/vole sous __main__ passaient un intitule different
    jusqu'au 06/10/2026, tant qu'elles se normalisaient sur le maximum de
    chaque configuration ; elles utilisent desormais la reference commune du
    cas 1 subwoofer, donc le defaut, et toutes les cartes horizontales
    partagent une seule echelle de SPL.

    L'echelle de couleur elle-meme (`levels` ci-dessous, -30 a +20 dB par pas
    de 2) a toujours ete fixe et identique pour tous les panneaux : c'est la
    reference soustraite qui doit etre commune pour que deux cartes soient
    reellement comparables."""
    # Hauteur du plan d'observation (06/10/2026) : une carte vue de dessus ne
    # vaut que pour le plan ou elle est calculee, et une figure de rapport
    # doit le dire.
    #
    # LUE SUR LA ZONE, jamais ecrite en dur ni prise sur Z_LISTENER : si
    # l'appelant passe une zone a une autre hauteur, le sous-titre suit. Il
    # n'apparait que si la zone est bien un plan horizontal (un seul z) ;
    # sinon on ne sait pas quoi annoncer, et on se tait.
    #
    # Sous-titre separe, et non suffixe du titre : les panneaux portent deja
    # un « z » qui designe la hauteur des CAISSES (« Ground (z = 0 m) »,
    # « Flown (z = 10 m) »). Deux z de sens differents sur la meme figure se
    # confondent, d'ou le mot « observation plane », qui tranche. Le mettre
    # sur sa propre ligne evite en plus de rallonger le titre, dont la police
    # retrecit avec sa longueur (jusqu'a 8.8 pt sur la planche cardioide).
    _z_unique = np.unique(zone.Z)
    z_label = (f"observation plane z = {_z_unique[0]:g} m"
               if _z_unique.size == 1 else None)

    show_z = bool(title) and z_label is not None

    nrows = int(np.ceil(len(configs_meta) / ncols))
    # Bande reservee en haut pour le titre general, sinon il chevauche les
    # titres de panneaux de la premiere rangee : 0.6 pouce, 0.9 quand le
    # sous-titre du plan d'observation occupe une 2e ligne.
    #
    # Le supplement est AJOUTE A LA HAUTEUR DE FIGURE, pas pris sur les
    # panneaux : ceux-ci sont a aspect egal et contraints par leur largeur,
    # donc leur hauteur ne suit pas celle de leur cellule. Raccourcir la
    # cellule ne les retrecit pas, il les decale vers le bas -- et le premier
    # essai a ainsi pousse les libelles « x [m] » hors de l'image.
    header = 0.9 if show_z else 0.6
    # Bande du bas, en pouces elle aussi. Les panneaux sont a aspect egal et
    # contraints par leur LARGEUR : sur les planches a 2 colonnes, ils sont
    # plus hauts que leur cellule et debordent, si bien que le libelle
    # « x [m] » tombait hors de l'image (defaut present depuis la tache 11,
    # visible sur toutes les planches sol/vole). La marge proportionnelle
    # d'origine (6 % de la hauteur) suffisait aux figures a 3 rangees, pas aux
    # planches a 1 rangee : on garde donc la plus grande des deux.
    footer = max(0.95, 0.06 * (3.0 * nrows + 1.0))
    fig_h = 3.0 * nrows + 1.0 + (header - 0.6) + (footer - 0.06 * (3.0 * nrows + 1.0))
    fig = plt.figure(figsize=(7.2 * ncols, fig_h))
    top = 1.0 - header / fig_h
    gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[30] * ncols + [1],
                          hspace=0.32, wspace=0.12,
                          left=0.09, right=0.93, top=top,
                          bottom=footer / fig_h)
    cax = fig.add_subplot(gs[:, ncols])

    levels = np.arange(-30, 22, 2)
    cmap = "viridis"
    contour = None

    Xg = zone.reshape(zone.X)
    Yg = zone.reshape(zone.Y)

    for i, (panel_title, rel_spl, array) in enumerate(configs_meta):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        contour = ax.contourf(Xg, Yg, zone.reshape(rel_spl),
                              levels=levels, cmap=cmap, extend="both")
        for sub in array:
            # pointille = caisse en l'air (volee), plein = posee au sol
            is_flown = sub.z - SUB_H / 2 > 0.01
            ax.add_patch(Polygon(sub_footprint(sub), closed=True,
                                 facecolor="white", edgecolor="black",
                                 linewidth=1.2,
                                 linestyle="--" if is_flown else "-"))
        ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=5)
        ax.set_xlim(zone.X.min(), zone.X.max())
        ax.set_ylim(zone.Y.min(), zone.Y.max())
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")
        ax.set_xlabel("x [m]", fontsize=10)
        ax.set_ylabel("y [m]", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, linewidth=0.5, alpha=0.4)

    cbar = fig.colorbar(contour, cax=cax, ticks=np.arange(-30, 19, 6))
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    pos = cbar.ax.get_position()
    cbar.ax.set_position([pos.x0 - 0.03, pos.y0, pos.width, pos.height])

    if title:
        # on reduit la police si le titre est trop long pour la largeur de figure
        fs = min(15.0, 460.0 * ncols / (0.62 * max(len(title), 1)))
        fig.suptitle(title, fontsize=fs, fontweight="bold",
                     x=0.5, y=1.0 - 0.22 / fig_h)
        if show_z:
            fig.text(0.5, 1.0 - 0.50 / fig_h, z_label, fontsize=9.5,
                     color="0.35", ha="center", va="top")
    return fig


class Result:
    """Resultat de Simulation.run() : SPL par zone, filtres effectivement
    appliques, et les zones elles-memes (pour que les methodes plot_*
    retrouvent leurs coordonnees et puissent y appliquer evaluation_mask)."""

    def __init__(self, spl, q, zones, array):
        self.spl = spl
        self.q = q
        self.zones = zones
        self._array = array  # hors contrat public : seulement pour recalculer
                              # le masque d'evaluation dans les methodes plot_*

    def plot_map(self, zone_name, title=None):
        """Carte SPL relative a un seul panneau, pour ce Result sur
        `zone_name` : spl_relatif = self.spl[zone_name] - max(self.spl[zone_name])
        sur la zone entiere, array = self._array. Formule non modifiee.

        Enveloppe fine autour de la fonction de module `plot_configs` (voir
        sa docstring pour les parametres visuels, repris sans modification).
        Pour tracer plusieurs configurations cote a cote dans la meme
        figure -- ce que faisait l'original avec sa liste globale CONFIGS --
        appeler `plot_configs(zone, configs_meta, ncols, title)` directement :
        ce n'est pas une operation sur UN Result (qui ne porte qu'un seul
        array), donc pas une methode de cette classe. Voir le catalogue sous
        __main__ pour un exemple de cet usage groupe.

        Intitule de barre de couleur -- CORRIGE par la ronde de correction 3
        (constat 2) : la reference ici est le maximum de CETTE carte, pas
        celui du cas 1 subwoofer (c'est le defaut de `plot_configs`, faux
        dans ce cas precis) -- on passe donc explicitement un intitule qui
        le dit."""
        zone = self.zones[zone_name]
        spl = self.spl[zone_name]
        configs_meta = [(title or zone_name, spl - np.max(spl), self._array)]
        return plot_configs(zone, configs_meta, ncols=1, title=title,
                            cbar_label="Level relative to this configuration's "
                                       "maximum [dB]")

    def plot_slice(self, zone_name, title=None, ref=None):
        """Coupe verticale SPL relative, sur une Zone.slice_xz ou
        Zone.slice_yz (l'orientation est lue sur les coordonnees de la zone :
        Y constant => coupe xz, X constant => coupe yz -- jamais sur son nom,
        que l'appelant peut avoir change).

        Portee depuis `plot_vertical`, dupliquee quasi a l'identique dans les
        7 scripts de Codes/ (Sub solo.py, Horizontal array.py, Vertical
        array.py, End-fire.py, Horizontal cardioid.py, Vertical cardioid.py,
        Gradient end-fire 1.py) : un seul portage contre ces sept copies.

        REFERENCE ET PLAGE -- REVUS LE 06/10/2026, apres mesure.

        Avant : la reference etait le maximum sur la grille HORIZONTALE
        (`ref=result.ref`, l'ancrage des 7 scripts de Codes/), et l'echelle
        couvrait 20 dB, de -20 a 0, le SPL relatif etant ecrete par np.clip.
        Une coupe verticale traverse le champ proche des caisses, bien plus
        fort que le niveau a hauteur d'ecoute : la coupe sortait donc par le
        haut de l'echelle et s'ecrasait sur une seule couleur. Mesure sur les
        sept scripts : 26 a 89 % de la coupe au-dessus de 0 dB dans les cas
        voles (89 % pour la pile verticale volee, 57 % pour la cardioide
        verticale), et 33 a 73 % sous -20 dB dans les cas au sol. La figure
        ne montrait plus rien. Le module heritait du meme defaut, attenue par
        la moyenne de bande : 16 a 35 % hors echelle par le haut.

        Maintenant, deux changements :

        1. `ref=None` (defaut) : reference AUTONOME, calculee sur la coupe
           elle-meme -- maximum du SPL sur la rangee de grille la plus proche
           de Z_LISTENER, hors gabarit des caisses. C'est le point le plus
           fort a hauteur d'oreille dans le plan de coupe. La coupe ne depend
           donc plus d'une zone horizontale jumelle dans le meme Result, d'ou
           le mot « decouplee ». Mesure sur les six familles : cette
           reference s'ecarte de l'ancienne (maximum sur la grille
           horizontale) de moins de 0,5 dB dans neuf cas sur dix, 1,6 dB au
           pire -- le changement de lecture vient de la plage, pas d'elle.

           Ce n'est PAS l'ancien comportement `ref=None`, qui prenait le
           maximum sur TOUTE la coupe hors gabarit : cette valeur-la est un
           niveau de champ proche, et tout le reste de la figure tombait
           alors dans le noir.

        2. Plage portee de 20 a 50 dB, `np.arange(-30, 22, 2)` -- la MEME que
           les cartes horizontales de `plot_configs` -- avec `extend="both"`
           au lieu de l'ecretage. Mesure sur les six familles : 0,04 % des
           points sortent de [-30, +20] dB, contre 4,5 % de [-25, +15] ; les
           0,04 % restants sont le champ proche immediat des caisses volees,
           que la fleche de la barre de couleur signale honnetement au lieu
           de l'ecraser en silence.

        - `ref=<valeur>` : reference FOURNIE par l'appelant, utilisee telle
          quelle (aucun recalcul). C'est par la qu'on retrouve l'ancrage des
          7 scripts de Codes/, ou `ref_ground` et `ref_flown` sont calculees
          UNE fois sur la zone HORIZONTALE puis reutilisees sans recalcul
          pour la coupe : passer le `.ref` renvoye par `compare_ground_flown`
          reproduit ce comportement. C'est aussi par la qu'on donne une
          reference COMMUNE a deux coupes qu'on veut comparer en niveau (sol
          contre vole, par exemple) -- avec le defaut, chaque coupe est
          normalisee sur son propre maximum a hauteur d'ecoute, ce qui
          repond bien a « le niveau est-il homogene ? » mais efface l'ecart
          de niveau entre les deux cas.

        Autres parametres visuels repris SANS MODIFICATION des 7 scripts :
        cmap 'inferno', contour rouge au seuil cible -3 dB, trait noir au
        niveau du sol (z = 0) ; rectangle de caisse en bord noir d'epaisseur
        1.5 (linewidth, comme les 7 scripts) ; et le contour (rectangle
        blanc) de chaque caisse de l'array dont l'empreinte croise le plan
        de coupe, projetee sur ce plan (voir _box_bbox_xy pour l'ecart
        documente entre la convention
        SUB_W/SUB_D des bar3d/Rectangle de Codes/ et celle, deja validee, de
        sub_footprint -- ce contour 2D reste inchange par la decision du
        cube en 3D, qui ne concerne que plot_3d_scene).

        Trait vert a la hauteur d'observation : trace a Z_LISTENER, la meme
        constante que le defaut de Zone.grid_xy/Zone.pit, plutot qu'a 1.5 m
        (valeur des 7 scripts de Codes/, incoherente avec la hauteur a
        laquelle la carte horizontale de la MEME famille est reellement
        calculee)."""
        zone = self.zones[zone_name]
        spl = self.spl[zone_name]
        # Rangee de grille la plus proche de la hauteur d'ecoute. On ne
        # cherche PAS z == Z_LISTENER : le pas de la coupe (0.5 m par defaut)
        # ne tombe pas sur 1.6 m, la rangee utile est celle a 1.5 m.
        z_row = float(zone.Z[np.argmin(np.abs(zone.Z - Z_LISTENER))])
        ref_is_auto = ref is None
        if ref is None:
            keep = (zone.Z == z_row) & evaluation_mask(zone, self._array)
            if not np.any(keep):            # coupe entierement dans le
                keep = (zone.Z == z_row)    # gabarit des caisses
            ref = float(np.max(spl[keep]))
        rel = spl - ref

        xz = bool(np.ptp(zone.Y) == 0.0)   # Y constant => plan y = cste (coupe xz)
        if xz:
            Hg, Vg = zone.reshape(zone.X), zone.reshape(zone.Z)
            horiz_label = r"$x$ [m]"
            const_coord = float(zone.Y[0])
        else:
            Hg, Vg = zone.reshape(zone.Y), zone.reshape(zone.Z)
            horiz_label = r"$y$ [m]"
            const_coord = float(zone.X[0])
        rel_grid = zone.reshape(rel)

        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        levels = np.arange(-30, 22, 2)
        cf = ax.contourf(Hg, Vg, rel_grid, levels=levels, cmap="inferno",
                         extend="both")
        ax.contour(Hg, Vg, rel_grid, levels=[-3.0], colors="red",
                  linewidths=1, linestyles="solid")

        for sub in self._array:
            xmin, xmax, ymin, ymax = _box_bbox_xy(sub)
            z_bottom = max(0.0, sub.z - SUB_H / 2)
            if xz:
                lo, hi, hmin, hmax = ymin, ymax, xmin, xmax
            else:
                lo, hi, hmin, hmax = xmin, xmax, ymin, ymax
            if lo - 1e-9 <= const_coord <= hi + 1e-9:
                ax.add_patch(Rectangle((hmin, z_bottom), hmax - hmin, SUB_H,
                                       facecolor="white", edgecolor="black",
                                       linewidth=1.5))

        ax.axhline(0.0, color="black", linewidth=1)
        ax.axhline(Z_LISTENER, color="limegreen", linewidth=1, alpha=0.5)
        ax.set_xlabel(horiz_label)
        ax.set_ylabel(r"$z$ [m]")
        ax.set_xlim(float(Hg.min()), float(Hg.max()))
        ax.set_ylim(float(Vg.min()), float(Vg.max()))
        if title:
            ax.set_title(title, fontsize=12)
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(f"Level relative to the maximum at z = {z_row:g} m [dB]"
                       if ref_is_auto else "Normalised level [dB]", fontsize=10)
        cbar.ax.axhline(-3.0, color="red", linewidth=1)
        fig.tight_layout()
        return fig

    def plot_directivity(self, zone_name, freqs=(30, 50, 80), title=None):
        """Directivite azimutale : SPL par frequence sur l'anneau `zone_name`.

        Porte depuis `plot_azimuthal_directivity`
        (Codes/1ere config sub.py:531-572). `zone_name` doit designer une
        Zone construite par Zone.ring : son azimut n'est pas stocke dans la
        Zone, il est donc reconstruit par np.linspace(0, 2*pi, zone.n_points)
        -- exactement comme Zone.ring l'a construit (voir sa docstring, qui
        annonce cette reconstruction).

        A chaque frequence de `freqs`, le champ est recalcule a la frequence
        unique (Monopole + DelayGainQ, memes formules que Simulation),
        independamment de la bande avec laquelle ce Result a ete produit :
        comme l'original, qui appelait compute_spl(freqs=[f]) pour isoler
        la directivite a une frequence donnee de la bande d'etude.
        """
        zone = self.zones[zone_name]
        az = np.linspace(0.0, 2.0 * np.pi, zone.n_points)
        model = Monopole()

        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2),
                                 subplot_kw={}, constrained_layout=True)
        axes[1].remove()
        ax_pol = fig.add_subplot(1, 2, 2, projection="polar")
        ax_lin = axes[0]

        for f in freqs:
            q = DelayGainQ(self._array, [f])
            p = model.field(self._array, q, zone, [f])
            spl = spl_from_field(p)
            rel = spl - spl.mean()
            ripple = spl.max() - spl.min()
            lab = f"{f:.0f} Hz  (ondulation {ripple:.1f} dB)"
            ax_lin.plot(np.rad2deg(az), rel, linewidth=1.6, label=lab)
            ax_pol.plot(az, rel, linewidth=1.6, label=lab)

        ax_lin.set_xlabel("Azimut [deg]", fontsize=10)
        ax_lin.set_ylabel("Niveau relatif a la moyenne azimutale [dB]", fontsize=10)
        ax_lin.set_xlim(0, 360)
        ax_lin.set_xticks(np.arange(0, 361, 45))
        ax_lin.grid(True, linewidth=0.5, alpha=0.4)
        ax_lin.legend(fontsize=9)
        ax_lin.tick_params(labelsize=9)

        ax_pol.set_theta_zero_location("E")
        ax_pol.grid(True, linewidth=0.5, alpha=0.4)
        ax_pol.tick_params(labelsize=8)

        if title:
            fig.suptitle(title, fontsize=13, fontweight="bold")
        return fig

    def save(self, directory):
        """Enregistre une carte PNG par zone de ce Result dans `directory`.

        Un panneau unique par zone (plot_map sans configs_meta). Pour un
        regroupement multi-configurations comme celui du catalogue sous
        __main__, appeler plot_map(..., configs_meta=...) puis fig.savefig()
        directement : save() ne connait qu'un seul Result (un seul array).
        """
        os.makedirs(directory, exist_ok=True)
        paths = []
        for zone_name in self.spl:
            fig = self.plot_map(zone_name)
            path = os.path.join(directory, _slug(zone_name) + ".png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
        return paths


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # CATALOGUE DE CONFIGURATIONS
    # Porte depuis Codes/1ere config sub.py:366-473 (declarations) et
    # 628-636 (boucle de trace finale). Toutes les formules geometriques
    # viennent des constructeurs/transformations de SubArray deja valides
    # (aucune formule n'est reecrite ici) ; seul ce bloc, qui ne s'execute
    # jamais a l'import, assemble et trace le catalogue.
    # ---------------------------------------------------------------------
    import time
    from concurrent.futures import ThreadPoolExecutor
    from typing import Callable, Dict

    FLY_HEIGHT = 8.0        # hauteur du bas de caisse pour les configs volees [m]
    LR_OFFSET = 10.0        # demi-ecart des deux arrays L/R [m]
                            # = arrays aux extremites d'une scene de 20 m de large
    CARDIOID_INDEX = 1      # rang de la caisse cardioide dans chaque groupe de 4
    SIDE_AIM = 60.0         # angle de tir des arrays lateraux [deg]
    SIDE_ANCHOR_Y = 1.5     # position en y de la caisse arriere des arrays lateraux
    ARC_N = 8
    ARC_RADIUS = 15.0
    CLUSTER_N_COL = 4                       # nombre de colonnes
    CLUSTER_R = 1.0                         # rayon axe central -> colonne [m]
    CLUSTER_N_EL = 4                        # caisses par colonne
    CLUSTER_H_S = 8.0                       # hauteur du centre de grappe / sol [m]
    CLUSTER_OFFSET = 0.0                    # 0 = croix (tir radial), 90 = moulinet
    BAND_CORTEEL = np.linspace(30, 80, 51)  # bande d'etude 30-80 Hz (Corteel 2018)

    # FREQUENCE DE TRAVAIL DU CATALOGUE -- 80 Hz, decision de Martin du
    # 06/10/2026, pour aligner la cartographie SPL sur les scripts de Codes/,
    # qui calculent tous a `f = 80` Hz.
    #
    # Remplace la bande globale BAND (30-124 Hz, 94 frequences moyennees) dans
    # TOUT ce bloc __main__. BAND reste le defaut de Simulation, donc de
    # l'API : ce choix est celui des figures, pas du moteur.
    #
    # Ce n'est pas qu'un reglage d'affichage. A frequence unique les franges
    # d'interference sol/source sont nettes ; moyennees sur 94 frequences
    # elles s'effacent presque entierement. Les deux lectures sont valides et
    # ne repondent pas a la meme question -- 80 Hz montre le mecanisme, la
    # bande montre ce qu'un spectateur entend reellement.
    #
    # Deux figures gardent deliberement une bande, parce que leur objet EST
    # la dependance en frequence : la paire de grappes volees (BAND_CORTEEL
    # contre F_STUDY) et le releve de directivite azimutale.
    F_STUDY = np.array([80.0])
    F_STUDY.setflags(write=False)

    # --- reference et premiers essais ------------------------------------
    cardio_vertical = (SubArray.cardioid_column(4, CARDIOID_INDEX, y=-SUB_W / 2)
                       + SubArray.cardioid_column(4, CARDIOID_INDEX, y=+SUB_W / 2))
    cardio_horizontal = (SubArray.cardioid_row(4, CARDIOID_INDEX, y_center=-2 * SUB_W)
                         + SubArray.cardioid_row(4, CARDIOID_INDEX, y_center=+2 * SUB_W))
    cluster = SubArray.pinwheel_cluster(n_col=CLUSTER_N_COL, R=CLUSTER_R,
                                        n_el=CLUSTER_N_EL, dz=SUB_H, h_s=CLUSTER_H_S,
                                        pinwheel_offset_deg=CLUSTER_OFFSET)
    _cluster_tag = f"{CLUSTER_N_COL} col x {CLUSTER_N_EL} el, R = {CLUSTER_R:.1f} m"

    # CATALOGUE : un nom -> une fabrique sans argument. dict[str, Callable[[], SubArray]].
    CATALOGUE: "Dict[str, Callable[[], SubArray]]" = {
        # --- 1) reference et premiers essais -----------------------------
        "1 subwoofer": SubArray.single,
        "10 subwoofers - horizontal array": lambda: SubArray.line(10),
        "10 subwoofers - vertical stack": lambda: SubArray.stack(10),
        # --- 2) cardioide : 2 stacks de 4 accoles, 1 caisse sur 4 --------
        "Cardioid - ground-stacked, vertical (2x4)": lambda: cardio_vertical,
        "Cardioid - ground-stacked, horizontal (2x4)": lambda: cardio_horizontal,
        f"Cardioid - flown at {FLY_HEIGHT:.0f} m (2x4)":
            lambda: cardio_vertical.flown(FLY_HEIGHT),
        # --- 3) arc sub : physique (caisses sur l'arc) vs virtuel --------
        f"Arc physical - {ARC_N} subs, R = {ARC_RADIUS:.0f} m":
            lambda: SubArray.arc_physical(ARC_N, ARC_RADIUS),
        f"Arc virtual (delay only) - {ARC_N} subs, R = {ARC_RADIUS:.0f} m":
            lambda: SubArray.arc_virtual(ARC_N, ARC_RADIUS),
        # --- 4) grappe centrale volee : colonnes verticales en croix -----
        f"Flown cluster ({_cluster_tag}) - 80 Hz": lambda: cluster,
        f"Flown cluster ({_cluster_tag}) - 30-80 Hz": lambda: cluster,
    }
    # bande d'etude propre a une configuration ; absente = F_STUDY (80 Hz)
    FREQS_OVERRIDE = {
        f"Flown cluster ({_cluster_tag}) - 30-80 Hz": BAND_CORTEEL,
    }

    ENDFIRE_GROUND_NAMES = []
    for s in (1.2, 0.8):
        for n in (2, 4):
            name = f"End-fire ground - {n} rows x 4, spacing {s:.1f} m"
            CATALOGUE[name] = (lambda n=n, s=s: SubArray.endfire_rows(n, 4, s))
            ENDFIRE_GROUND_NAMES.append(name)

    ENDFIRE_FLOWN_NAMES = []
    for s in (1.2, 0.8):
        name_c = f"End-fire flown - centred, 2 rows x 4, spacing {s:.1f} m"
        CATALOGUE[name_c] = (lambda s=s:
                             SubArray.endfire_rows(2, 4, s).flown(FLY_HEIGHT))
        name_lr = (f"End-fire flown - L/R +/-{LR_OFFSET:.0f} m, "
                   f"2x(2 rows x 4), spacing {s:.1f} m")
        CATALOGUE[name_lr] = (lambda s=s:
            SubArray.endfire_rows(2, 4, s).flown(FLY_HEIGHT).lr_pair(LR_OFFSET))
        ENDFIRE_FLOWN_NAMES += [name_c, name_lr]

    # ---- Gradient end-fire -------------------------------------------------
    # Redefinie le 06/10/2026 d'apres le schema de Martin.
    #
    # Array CENTRAL : 2 lignes end-fire cote a cote, 3 caisses EN PROFONDEUR
    # chacune (6 caisses), tir a 0 deg, centre sur y = 0.
    # Arrays LATERAUX : meme forme avec UNE CAISSE DE MOINS EN PROFONDEUR
    # (2 lignes x 2 caisses = 4 caisses), tires a +/- GRADIENT_SIDE_AIM.
    # L'angle est mesure depuis l'axe x (le plan y = 0).
    #
    # Placement : chaque array lateral est construit a l'origine avec son
    # propre angle de tir, puis translate perpendiculairement a SON axe, vers
    # l'exterieur, de la demi-largeur du central plus la sienne plus un jeu.
    # On obtient l'eventail du schema -- lateraux recules en x et ecartes en
    # y -- sans chevauchement, quel que soit l'angle.
    GRADIENT_N_ROWS = 2        # lignes end-fire cote a cote
    GRADIENT_N_FRONT = 3       # caisses en profondeur, array central
    GRADIENT_N_SIDE = 2        # caisses en profondeur, arrays lateraux
    GRADIENT_GAP = 0.5         # jeu entre central et lateraux [m]

    def _gradient_endfire(spacing, side_aim_deg):
        central = SubArray.endfire_rows(GRADIENT_N_ROWS, GRADIENT_N_FRONT,
                                        spacing, y_center=0.0, aim_deg=0.0)
        half_w = GRADIENT_N_ROWS * SUB_W / 2.0      # demi-largeur d'un array
        offset = 2 * half_w + GRADIENT_GAP          # centre a centre
        sides = []
        for sign in (+1.0, -1.0):
            a = np.deg2rad(sign * side_aim_deg)
            side = SubArray.endfire_rows(GRADIENT_N_ROWS, GRADIENT_N_SIDE,
                                         spacing, y_center=0.0,
                                         aim_deg=sign * side_aim_deg)
            # normale a l'axe de tir, orientee vers l'exterieur
            sides.append(side.shift(dx=-sign * offset * np.sin(a),
                                    dy=+sign * offset * np.cos(a)))
        return central + sides[0] + sides[1]

    GRADIENT_NAMES = []
    for s in (1.2, 0.8):
        for aim in (30.0, 45.0):
            name = (f"End-fire gradient - {GRADIENT_N_FRONT} front + 2 side at "
                    f"+/-{aim:.0f} deg, spacing {s:.1f} m")
            CATALOGUE[name] = (lambda s=s, aim=aim:
                               _gradient_endfire(s, aim))
            GRADIENT_NAMES.append(name)

    BASE_NAMES = ["1 subwoofer", "10 subwoofers - horizontal array",
                 "10 subwoofers - vertical stack"]
    CARDIOID_NAMES = ["Cardioid - ground-stacked, vertical (2x4)",
                      "Cardioid - ground-stacked, horizontal (2x4)",
                      f"Cardioid - flown at {FLY_HEIGHT:.0f} m (2x4)"]
    CLUSTER_NAMES = [f"Flown cluster ({_cluster_tag}) - 80 Hz",
                     f"Flown cluster ({_cluster_tag}) - 30-80 Hz"]
    ARC_NAMES = [f"Arc physical - {ARC_N} subs, R = {ARC_RADIUS:.0f} m",
                f"Arc virtual (delay only) - {ARC_N} subs, R = {ARC_RADIUS:.0f} m"]

    # regroupement en figures : (titre, nombre de colonnes, noms de configs)
    FIGURES = [
        ("Subwoofer SPL distribution - 80 Hz", 1, BASE_NAMES),
        ("Cardioid arrays - 8 subs (2 stacks of 4, 1 box in 4 cardioid)", 1, CARDIOID_NAMES),
        ("End-fire ground-stacked - 4 boxes per row", 2, ENDFIRE_GROUND_NAMES),
        (f"End-fire flown at {FLY_HEIGHT:.0f} m - 4 boxes per row", 2, ENDFIRE_FLOWN_NAMES),
        (f"Flown cluster - {CLUSTER_N_COL} vertical columns around a common axis "
         f"(h_s = {CLUSTER_H_S:.0f} m)", 2, CLUSTER_NAMES),
        (f"Arc subwoofer array - physical vs virtual, R = {ARC_RADIUS:.0f} m", 2, ARC_NAMES),
        (f"End-fire gradient - central {GRADIENT_N_FRONT}-deep + 2 side "
         f"{GRADIENT_N_SIDE}-deep, 14 subs", 2, GRADIENT_NAMES),
    ]

    all_names = [name for _, _, names in FIGURES for name in names]
    # 22 depuis le 06/10/2026 : le gradient end-fire est decline sur DEUX angles
    # lateraux (30 et 45 deg) en plus des deux espacements, soit 4 entrees au
    # lieu de 2. Garde-fou contre une entree perdue ou dupliquee, pas une
    # contrainte de conception : mettre ce nombre a jour si le catalogue change.
    assert len(all_names) == 22, f"attendu 22 configurations, trouve {len(all_names)}"

    # controle de validite de la grappe : lobes de reseau en azimut
    cluster.check_grating_lobes(CLUSTER_N_COL, CLUSTER_R, BAND_CORTEEL[-1],
                                label="grappe, bande 30-80 Hz :")
    cluster.check_grating_lobes(CLUSTER_N_COL, CLUSTER_R, F_STUDY[-1],
                                label="grappe, 80 Hz :")

    # grille d'affichage : etendue par rapport a l'original (x in [-40,40],
    # y in [-20,20]) pour couvrir la fosse du projet (x jusqu'a 50 m,
    # y jusqu'a +/-25 m). Changement voulu de la tache 11.
    zone = Zone.grid_xy((-40, 55), (-30, 30), step=1.0)

    # Les configurations sont independantes : on les repartit sur les coeurs.
    # Threads et non processus -- numpy relache le GIL dans ses boucles
    # internes ; chaque configuration est calculee exactement comme Simulation
    # le ferait seule, seul l'ordre d'execution entre configurations change.
    n_workers = min(os.cpu_count() or 1, len(all_names))

    def _compute_one(name):
        array = CATALOGUE[name]()
        freqs = FREQS_OVERRIDE.get(name, F_STUDY)
        sim = Simulation(array, freqs=freqs)
        return name, sim.run([zone]), array

    print(f"Calcul de {len(all_names)} configurations sur {n_workers} threads...")
    _t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as _pool:
        _computed = list(_pool.map(_compute_one, all_names))
    results = {name: result for name, result, _array in _computed}
    arrays = {name: _array for name, _result, _array in _computed}
    print(f"Calcul termine en {time.time() - _t0:.1f} s")

    # REFERENCE = MAX DU CAS 1 SUB, evaluee dans la meme bande que la config
    # tracee, pour que l'echelle de couleur garde le meme sens d'une figure a
    # l'autre (max sur la zone entiere, sans evaluation_mask : comme l'original).
    _ref_cache = {}

    def reference_level(freqs):
        key = "default" if freqs is F_STUDY else tuple(np.round(np.atleast_1d(freqs), 6))
        if key not in _ref_cache:
            ref_sim = Simulation(SubArray.single(), freqs=freqs)
            _ref_cache[key] = float(np.max(ref_sim.spl(zone)))
        return _ref_cache[key]

    refs = {name: reference_level(FREQS_OVERRIDE.get(name, F_STUDY)) for name in all_names}

    HERE = os.path.dirname(os.path.abspath(__file__))              # .../src
    # Repertoire de sortie : <racine du depot>/figures par defaut, redirigeable
    # par la variable d'environnement FIGURES_DIR pour produire plusieurs jeux
    # de figures cote a cote sans ecraser le precedent. Un chemin relatif est
    # resolu depuis la racine du depot, pas depuis le repertoire courant.
    _out = os.environ.get("FIGURES_DIR", "figures")
    OUTPUT_DIR = (_out if os.path.isabs(_out)
                  else os.path.join(os.path.dirname(HERE), _out))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_figures = 0
    for title, ncols, names in FIGURES:
        configs_meta = [(name, results[name].spl[zone.name] - refs[name], arrays[name])
                        for name in names]
        fig = plot_configs(zone, configs_meta, ncols=ncols, title=title)
        fig.savefig(os.path.join(OUTPUT_DIR, f"{n_figures + 1:02d} - {_slug(title)}.png"),
                   dpi=110)
        n_figures += 1

    # Directivite azimutale de la grappe : la config est-elle vraiment omni ?
    # SEULE FIGURE QUI RESTE SUR LA BANDE (avec la paire de grappes volees) :
    # son objet est justement la dependance en frequence -- elle trace 30, 50
    # et 80 Hz cote a cote. A F_STUDY (80 Hz seul) les deux autres courbes
    # n'existeraient pas et la figure perdrait sa raison d'etre.
    ring = Zone.ring(25.0)
    cluster_result = Simulation(cluster, freqs=BAND).run([ring])
    directivity_title = (f"Flown cluster - horizontal directivity at 25 m "
                         f"({_cluster_tag})")
    fig_dir = cluster_result.plot_directivity(ring.name, freqs=(30, 50, 80),
                                              title=directivity_title)
    fig_dir.savefig(os.path.join(OUTPUT_DIR,
                                 f"{n_figures + 1:02d} - {_slug(directivity_title)}.png"),
                    dpi=110)
    n_figures += 1

    # ---------------------------------------------------------------------
    # PLANCHES SOL / VOLE -- une par famille des 7 scripts de Codes/ (Sub
    # solo, Horizontal array, Vertical array, End-fire, Horizontal cardioid,
    # Vertical cardioid, Gradient end-fire), chacune batie sur
    # compare_ground_flown + Result.plot_slice + plot_3d_scene : porte
    # depuis la comparaison z_ground/z_flown que ces 7 scripts font chacun a
    # la main (tache 15, ronde de correction 1 -- constat 2). Les 8 figures
    # ci-dessus ne sont pas touchees : ce bloc ne fait qu'ajouter des
    # fichiers, a la suite, dans le meme OUTPUT_DIR.
    #
    # DEUX FAMILLES N'ONT PAS DE PAIRE SOL/VOLE (ronde de correction 2,
    # constat 1) : `Vertical array.py` et `Vertical cardioid.py`, dans
    # Codes/, n'ont pas de `z_flown` -- elles partent de `z_start_1 = 0` et
    # s'arretent la. Ce n'est pas un oubli des auteurs : un array VERTICAL
    # est par definition un empilement depuis le sol (c'est la definition
    # meme de "ground-stacked"). Le monter en vol en fait une colonne
    # suspendue -- une configuration differente, pas la contrepartie de la
    # meme config. Produire une planche sol/vole pour ces deux familles
    # fabriquerait donc une comparaison qui n'existe pas dans Codes/. On ne
    # produit pour elles que la coupe verticale et la scene 3D du cas au sol.
    #
    # HAUTEUR DE VOL DE CES FIGURES = 10 m, PAS FLY_HEIGHT (= 8 m) (ronde de
    # correction 2, constat 1) : les 5 scripts qui comparent effectivement
    # sol/vole (Sub solo.py, Horizontal array.py, End-fire.py, Horizontal
    # cardioid.py, Gradient end-fire 1.py) utilisent tous `z_flown = 10`.
    # Puisque ce bloc illustre precisement ces scripts, on s'aligne sur leur
    # valeur plutot que sur FLY_HEIGHT = 8 m du catalogue principal
    # (utilisee plus haut pour les configs "Cardioid - flown" et "End-fire
    # flown" du catalogue, qui sont des choix de conception independants,
    # pas un portage de Codes/). Les deux valeurs sont donc volontairement
    # differentes ; ce n'est pas une incoherence.
    # ---------------------------------------------------------------------
    LEGACY_Z_FLOWN = 10.0

    def _auto_slice(array, z_top, step=0.5, margin=3.0, x_far=50.0):
        """Coupe verticale LONGITUDINALE : plan y = constante, contenant l'axe
        de tir, de l'arriere de l'array jusqu'au fond de fosse.

        CORRIGE LE 06/10/2026. La version precedente choisissait entre xz et yz
        selon le plan qui traversait le plus de caisses, ce qui donnait un plan
        x = constante pour 5 familles sur 6 -- soit une coupe PERPENDICULAIRE a
        l'axe de tir, prise a hauteur des sources. Elle ne montrait donc ni la
        propagation vers le public, ni l'interference avec l'image de sol, ni
        la difference entre une config au sol et la meme en vol : exactement ce
        qu'une coupe verticale est censee reveler.

        Le critere « traverser l'array » etait le mauvais : ce qui compte est
        de contenir la direction de propagation. Tous les arrays du catalogue
        tirent vers +x, donc le plan utile est y = constante.

        `x_far` porte la coupe jusqu'au fond de fosse (50 m par defaut) au lieu
        de s'arreter a l'encombrement de l'array : sans cela on ne voyait que
        le champ proche."""
        xs = array.positions[:, 0]
        ys = array.positions[:, 1]
        y0 = float(ys.mean())
        xlim = (float(xs.min()) - margin, x_far)
        return Zone.slice_xz(xlim, (0.0, z_top), step=step, y=y0,
                             name=f"coupe demo xz y={y0:.2f}")

    # (nom de famille, fabrique SANS ARGUMENT renvoyant l'array AU SOL,
    # a_une_paire_sol_vole) -- reutilise les arrays deja construits plus
    # haut pour le catalogue principal, un par script de Codes/.
    COMPARISON_FAMILIES = [
        ("Sub solo", SubArray.single, True),
        ("Horizontal array (10 subs)", lambda: SubArray.line(10), True),
        ("Vertical array (10 subs)", lambda: SubArray.stack(10), False),
        ("End-fire ground (2 rows x 4, 1.2 m)",
         lambda: SubArray.endfire_rows(2, 4, 1.2), True),
        ("Horizontal cardioid (2x4)", lambda: cardio_horizontal, True),
        ("Vertical cardioid (2x4)", lambda: cardio_vertical, False),
        ("Gradient end-fire (front + 2 sides, 1.2 m)",
         CATALOGUE[GRADIENT_NAMES[0]], True),
    ]

    for family_name, base_builder, has_flown_pair in COMPARISON_FAMILIES:
        base_array = base_builder()

        if has_flown_pair:
            # z_top doit couvrir le sommet du cas VOLE (ronde de correction
            # 3, constat 3) : z_flown + hauteur propre de l'array.
            z_top = LEGACY_Z_FLOWN + max((s.z + SUB_H / 2 for s in base_array),
                                         default=SUB_H) + 2.0
            slice_zone = _auto_slice(base_array, z_top)

            factory = lambda h, b=base_builder: b().flown(h)
            result_ground, result_flown = compare_ground_flown(
                factory, 0.0, LEGACY_Z_FLOWN, [zone, slice_zone])
            cases = ((result_ground, "ground"), (result_flown, "flown"))

            # Vue de dessus, 2 colonnes. REFERENCE COMMUNE (06/10/2026) : le
            # maximum du cas 1 subwoofer, la meme que les figures 01 a 07 du
            # catalogue principal. Toutes les cartes horizontales partagent
            # donc une seule echelle de SPL et sont comparables entre elles.
            #
            # Remplace la normalisation PAR CARTE heritee des 7 scripts, qui
            # ramenait chaque carte a son propre maximum : les deux panneaux
            # culminaient alors tous deux a 0 dB et l'ecart de niveau entre le
            # cas au sol et le cas vole -- precisement ce que la planche doit
            # montrer -- disparaissait du graphique.
            #
            # L'intitule de la barre de couleur redevient donc celui par
            # defaut de plot_configs, et il est desormais exact.
            ref_common = reference_level(F_STUDY)
            configs_meta = [
                ("Ground (z = 0 m)",
                 result_ground.spl[zone.name] - ref_common,
                 result_ground._array),
                (f"Flown (z = {LEGACY_Z_FLOWN:.0f} m)",
                 result_flown.spl[zone.name] - ref_common,
                 result_flown._array),
            ]
            board_title = f"{family_name} - ground vs flown"
            fig_board = plot_configs(zone, configs_meta, ncols=2,
                                     title=board_title)
            fig_board.savefig(os.path.join(
                OUTPUT_DIR, f"{n_figures + 1:02d} - {_slug(board_title)}.png"), dpi=110)
            plt.close(fig_board)
            n_figures += 1
        else:
            # Vertical array / Vertical cardioid : pas de paire sol/vole
            # (voir plus haut) -- seulement le cas au sol. z_top base sur la
            # hauteur PROPRE de l'array, SANS LEGACY_Z_FLOWN (ronde de
            # correction 3, constat 3 : avec LEGACY_Z_FLOWN, la coupe restait
            # tracee jusqu'a plus de 2x le sommet reel de ces deux familles,
            # qui ne montent jamais en vol -- l'array se retrouvait ecrase en
            # bas de l'image).
            z_top = max((s.z + SUB_H / 2 for s in base_array),
                       default=SUB_H) + 2.0
            slice_zone = _auto_slice(base_array, z_top)

            # meme reference (max hors voisinage des caisses sur la grille
            # horizontale) que compare_ground_flown calcule pour les autres
            # familles.
            result_ground = Simulation(base_array, freqs=F_STUDY).run([zone, slice_zone])
            mask = evaluation_mask(zone, base_array)
            result_ground.ref = float(np.max(result_ground.spl[zone.name][mask]))
            cases = ((result_ground, "ground"),)

        # Coupes verticales -- REFERENCE AUTONOME depuis le 06/10/2026 : on
        # ne passe plus `ref=result.ref` (le maximum sur la grille
        # HORIZONTALE, ancrage des 7 scripts de Codes/). Cette reference-la
        # est un niveau a hauteur d'ecoute applique a une coupe qui traverse
        # le champ proche des caisses : la coupe sortait par le haut de
        # l'echelle et s'ecrasait sur une seule couleur (jusqu'a 89 % de la
        # surface pour la pile verticale volee dans les scripts, 35 % ici).
        # plot_slice calcule desormais sa propre reference sur la coupe, a
        # hauteur d'ecoute, et l'echelle couvre 50 dB. Voir sa docstring.
        for result, label in cases:
            slice_title = f"{family_name} - {label} - vertical slice"
            fig_slice = result.plot_slice(slice_zone.name, title=slice_title)
            fig_slice.savefig(os.path.join(
                OUTPUT_DIR, f"{n_figures + 1:02d} - {_slug(slice_title)}.png"), dpi=110)
            plt.close(fig_slice)
            n_figures += 1

        # scenes 3D -- caisses en cubes (decision de Martin, constat 1 de la
        # ronde de correction 1).
        for result, label in cases:
            scene_title = f"{family_name} - {label} - 3D scene"
            fig_scene = plot_3d_scene(result._array, slice_plane=slice_zone)
            fig_scene.suptitle(scene_title, fontsize=11, fontweight="bold")
            fig_scene.savefig(os.path.join(
                OUTPUT_DIR, f"{n_figures + 1:02d} - {_slug(scene_title)}.png"), dpi=110)
            plt.close(fig_scene)
            n_figures += 1

    print(f"\n=== {n_figures} figures ecrites dans {OUTPUT_DIR} ===")
