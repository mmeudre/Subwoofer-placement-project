"""Lance la simulation, enregistre les figures dans figures/ et verifie la
non-regression des trois configurations d'origine."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util, time, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "1ere config sub.py"))
m = importlib.util.module_from_spec(spec)
t0 = time.time()
spec.loader.exec_module(m)
print("\n=== total script : %.1f s, %d figures ===" % (time.time() - t0, len(plt.get_fignums())))

for i, n in enumerate(plt.get_fignums()):
    f = plt.figure(n)
    ttl = (f._suptitle.get_text() if f._suptitle else f"figure{i+1}")
    name = "".join(ch if ch.isalnum() or ch in " -_" else "" for ch in ttl)[:60].strip()
    f.savefig(os.path.join(OUT, f"{i+1:02d} - {name}.png"), dpi=110)
print("figures ecrites dans", OUT)

# --- non-regression contre la sauvegarde d'avant toute modification ---------
BASE = r"C:\Users\marti\AppData\Local\Temp\claude\C--Users-marti-Desktop-IMDEA-PFE\b649c15e-d917-4185-b1e2-97f1da447faa\scratchpad\baseline.npz"
if os.path.exists(BASE):
    b = np.load(BASE)
    print("\nNon-regression vs script d'origine :")
    for key, name in [("one", "1 subwoofer"),
                      ("horiz", "10 subwoofers - horizontal array"),
                      ("vert", "10 subwoofers - vertical stack")]:
        d = np.max(np.abs(b[key] - m.results[name]))
        print(f"  {name:38s} max|diff| = {d:.3e} dB")

mask = (m.X >= 5) & (m.X <= 35)
print("\nZone x in [5,35] m, relatif au 1 sub de la meme bande :")
for cfg in m.CONFIGS:
    d = m.results[cfg.name][mask] - m.refs[cfg.name]
    print(f"  {cfg.name:66s} n={len(cfg.subs):2d} moy={d.mean():6.2f} sd={d.std():5.2f}")

th = np.arange(-80, 81, 10)
px = (25 * np.cos(np.deg2rad(th)))[None, :]
py = (25 * np.sin(np.deg2rad(th)))[None, :]
print("\nCouverture horizontale a 25 m, dB relatifs a l'axe :")
print("  " + " " * 46 + " ".join(f"{a:+4.0f}" for a in th))
for cfg in m.ARC + [m.BASE[1]] + m.GRADIENT + m.CLUSTER[:1]:
    s = m.compute_spl(cfg.subs, freqs=cfg.freqs, Xg=px, Yg=py)[0]
    print(f"  {cfg.name[:44]:46s} " + " ".join(f"{v - s[len(s)//2]:+4.1f}" for v in s))
