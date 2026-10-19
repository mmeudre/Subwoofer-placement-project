import numpy as np

c = 343.0


def transfer_matrix(points, sources, f, reflection=1.0):
    """G[m, l] : pression en points[m] pour la caisse l (monopole unitaire),
    plus l'image par rapport au sol z = 0 pondérée par `reflection`.

    points  : (M, 3) points de controle
    sources : (L, 3) centres acoustiques
    """
    k = 2*np.pi*f/c
    points = np.asarray(points, float)[:, None, :]
    sources = np.asarray(sources, float)[None, :, :]
    image = sources*np.array([1.0, 1.0, -1.0])

    r_direct = np.maximum(np.linalg.norm(points - sources, axis=2), 0.05)
    r_image = np.maximum(np.linalg.norm(points - image, axis=2), 0.05)
    return (np.exp(-1j*k*r_direct)/r_direct
            + reflection*np.exp(-1j*k*r_image)/r_image)


def solve(G_B, p_target):
    """Poids q minimisant |p_T - G_B q|^2 : q = (G_B^H G_B)^-1 G_B^H p_T, cas de PM sans DarkZone"""
    return np.linalg.solve(G_B.conj().T @ G_B, G_B.conj().T @ p_target)


def uniform_level_target(G_B, n_iter=30):
    """Cible de module constant, de phase libre.

    On alterne : cible = module moyen avec la phase du champ courant, puis
    resolution du pressure matching classique. Le point de depart est q = 1.
    """
    q = np.ones(G_B.shape[1], dtype=complex)
    for _ in range(n_iter):
        p = G_B @ q
        p_target = np.mean(np.abs(p))*np.exp(1j*np.angle(p))
        q = solve(G_B, p_target)
    return q


# --- Indicateurs ------------------------------------------------------------

def level_spread(G_B, q):
    """Ecart L_max - L_min [dB] du SPL sur la bright zone."""
    spl = 20*np.log10(np.abs(G_B @ q))
    return spl.max() - spl.min()


def level_std(G_B, q):
    """Ecart-type du SPL [dB] sur la bright zone."""
    return np.std(20*np.log10(np.abs(G_B @ q)))


def reproduction_error(G_B, q, p_target):
    """E_rr normalisee (eq. 17)."""
    e = p_target - G_B @ q
    return np.real(e.conj() @ e)/np.real(p_target.conj() @ p_target)


def effort(G_B, q):
    """Effort [dB] (eq. 16) : energie de q rapportee a celle des poids reels
    identiques qui donneraient le meme niveau moyen sur la bright zone."""
    q_ref = np.mean(np.abs(G_B @ q))/np.mean(np.abs(G_B @ np.ones(len(q))))
    return 10*np.log10(np.real(q.conj() @ q)/(len(q)*q_ref**2))