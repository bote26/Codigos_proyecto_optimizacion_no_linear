import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.integrate import solve_ivp

# ── Parámetros base (Tabla 2) ────────────────────────────────────────────────
p_base = dict(
    R0=864,
    Ge=140,
    EG0=1.44,
    SI=0.72,
    sig=43.2,
    alp=20000.0,
    rho=0.41,
    k=432.0,
    d0=0.06,
    r1=0.00084,
    r2=0.0000024,
)


# ── Funciones de equilibrio y estabilidad ────────────────────────────────────
def equilibrios(p):
    """Devuelve lista de (G*, I*, β*, tipo) para los parámetros dados."""
    puntos = []
    # P3 (trivial): β=0, I=0
    G3 = (p["R0"] + p["Ge"]) / p["EG0"]
    puntos.append((G3, 0.0, 0.0))
    # P1, P2 (no triviales): raíces de r2·G²-r1·G+d0=0
    disc = p["r1"] ** 2 - 4 * p["r2"] * p["d0"]
    if disc >= 0:
        for G in [
            (p["r1"] + np.sqrt(disc)) / (2 * p["r2"]),
            (p["r1"] - np.sqrt(disc)) / (2 * p["r2"]),
        ]:
            if G > 0:
                I = (p["R0"] + p["Ge"] - p["EG0"] * G) / (p["SI"] * G)
                b = (p["rho"] + p["k"]) * I * (p["alp"] + G**2) / (p["sig"] * G**2)
                if I > 0 and b > 0:
                    puntos.append((G, I, b))
    return sorted(puntos, key=lambda x: x[0])


def jacobiano(G, I, b, p):
    d = p["alp"] + G**2
    return np.array(
        [
            [-(p["EG0"] + p["SI"] * I), -p["SI"] * G, 0.0],
            [
                b * p["sig"] * 2 * p["alp"] * G / d**2,
                -(p["rho"] + p["k"]),
                p["sig"] * G**2 / d,
            ],
            [
                (p["r1"] - 2 * p["r2"] * G) * b,
                0.0,
                -p["d0"] + p["r1"] * G - p["r2"] * G**2,
            ],
        ]
    )


def clasificar(G, I, b, p):
    ev = np.linalg.eigvals(jacobiano(G, I, b, p))
    n_pos = sum(1 for l in ev if l.real > 1e-8)
    n_neg = sum(1 for l in ev if l.real < -1e-8)
    if n_pos == 0:
        return "estable", ev
    if n_neg == 0:
        return "inestable", ev
    return "silla", ev


def sistema(t, y, p):
    G, I, b = y
    return [
        p["R0"] + p["Ge"] - (p["EG0"] + p["SI"] * I) * G,
        b * p["sig"] * G**2 / (p["alp"] + G**2) - (p["rho"] + p["k"]) * I,
        (-p["d0"] + p["r1"] * G - p["r2"] * G**2) * b,
    ]


def f2D(G, B, p):
    rk = p["rho"] + p["k"]
    Iq = B * p["sig"] * G**2 / ((p["alp"] + G**2) * rk)
    return (
        p["R0"] + p["Ge"] - (p["EG0"] + p["SI"] * Iq) * G,
        (-p["d0"] + p["r1"] * G - p["r2"] * G**2) * B,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  F4.1  —  ANÁLISIS PARAMÉTRICO: Ge y ρ
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  F4.1 — Variación de Ge y ρ")
print("=" * 65)

Ge_vals = np.linspace(0, 500, 300)
rho_vals = np.linspace(0, 432, 300)

# ── Bifurcación respecto a Ge ─────────────────────────────────────────────────
G3_vs_Ge, I1_vs_Ge, I2_vs_Ge, b1_vs_Ge, b2_vs_Ge = [], [], [], [], []
stab1, stab2, stab3 = [], [], []

for Ge in Ge_vals:
    pp = {**p_base, "Ge": Ge}
    pts = equilibrios(pp)
    G3_vs_Ge.append((p_base["R0"] + Ge) / p_base["EG0"])
    if len(pts) == 3:
        p1, p2, p3 = sorted(pts, key=lambda x: x[0])
        I1_vs_Ge.append(p1[1])
        b1_vs_Ge.append(p1[2])
        I2_vs_Ge.append(p2[1])
        b2_vs_Ge.append(p2[2])
        s1, _ = clasificar(*p1, pp)
        s2, _ = clasificar(*p2, pp)
        s3, _ = clasificar(*p3, pp)
        stab1.append(s1)
        stab2.append(s2)
        stab3.append(s3)
    else:
        I1_vs_Ge.append(np.nan)
        b1_vs_Ge.append(np.nan)
        I2_vs_Ge.append(np.nan)
        b2_vs_Ge.append(np.nan)
        stab1.append(None)
        stab2.append(None)
        stab3.append(None)

# ── Bifurcación respecto a ρ ──────────────────────────────────────────────────
b1_vs_rho, b2_vs_rho, I1_vs_rho, I2_vs_rho = [], [], [], []
for rho in rho_vals:
    pp = {**p_base, "rho": rho}
    pts = equilibrios(pp)
    if len(pts) == 3:
        p1, p2, _ = sorted(pts, key=lambda x: x[0])
        b1_vs_rho.append(p1[2])
        b2_vs_rho.append(p2[2])
        I1_vs_rho.append(p1[1])
        I2_vs_rho.append(p2[1])
    else:
        b1_vs_rho.append(np.nan)
        b2_vs_rho.append(np.nan)
        I1_vs_rho.append(np.nan)
        I2_vs_rho.append(np.nan)

# ── FIGURA F4.1 ───────────────────────────────────────────────────────────────
fig41, axes41 = plt.subplots(2, 3, figsize=(16, 10))
fig41.suptitle(
    "F4.1 — Análisis Paramétrico: efecto de  Gₑ  y  ρ", fontsize=13, fontweight="bold"
)

# Fila 1: variando Ge
ax = axes41[0, 0]
ax.plot(Ge_vals, G3_vs_Ge, "k-", lw=2, label="G* (P₃)")
ax.axvline(140, color="gray", ls="--", lw=1.2, label="Gₑ base=140")
ax.set_xlabel("Gₑ")
ax.set_ylabel("G* de P₃")
ax.set_title("P₃: G* = (R₀+Gₑ)/E_G0  vs  Gₑ")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes41[0, 1]
ax.plot(Ge_vals, I1_vs_Ge, "b-", lw=2, label="I* en P₁ (G=100)")
ax.plot(Ge_vals, I2_vs_Ge, "r-", lw=2, label="I* en P₂ (G=250)")
ax.axvline(140, color="gray", ls="--", lw=1.2, label="Gₑ base")
ax.set_xlabel("Gₑ")
ax.set_ylabel("I*")
ax.set_title("Insulina de equilibrio vs Gₑ")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes41[0, 2]
ax.plot(Ge_vals, b1_vs_Ge, "b-", lw=2, label="β* en P₁")
ax.plot(Ge_vals, b2_vs_Ge, "r-", lw=2, label="β* en P₂")
ax.axvline(140, color="gray", ls="--", lw=1.2, label="Gₑ base")
ax.set_xlabel("Gₑ")
ax.set_ylabel("β*")
ax.set_title("Masa β de equilibrio vs Gₑ")
ax.legend()
ax.grid(True, alpha=0.3)

# Fila 2: variando ρ
ax = axes41[1, 0]
ax.plot(rho_vals, I1_vs_rho, "b-", lw=2, label="I* en P₁")
ax.plot(rho_vals, I2_vs_rho, "r-", lw=2, label="I* en P₂")
ax.axvline(0.41, color="gray", ls="--", lw=1.2, label="ρ base=0.41")
ax.set_xlabel("ρ")
ax.set_ylabel("I*")
ax.set_title("Insulina de equilibrio vs ρ")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes41[1, 1]
ax.plot(rho_vals, b1_vs_rho, "b-", lw=2, label="β* en P₁")
ax.plot(rho_vals, b2_vs_rho, "r-", lw=2, label="β* en P₂")
ax.axvline(0.41, color="gray", ls="--", lw=1.2, label="ρ base=0.41")
ax.set_xlabel("ρ")
ax.set_ylabel("β*")
ax.set_title("Masa β de equilibrio vs ρ")
ax.legend()
ax.grid(True, alpha=0.3)

# Espacio de parámetros Ge–ρ: cuántos equilibrios existen
ax = axes41[1, 2]
Ge_g, rho_g = np.meshgrid(np.linspace(0, 400, 80), np.linspace(0, 1.5, 80))
N_eq = np.zeros_like(Ge_g)
for i in range(Ge_g.shape[0]):
    for j in range(Ge_g.shape[1]):
        pp = {**p_base, "Ge": Ge_g[i, j], "rho": rho_g[i, j]}
        N_eq[i, j] = len(equilibrios(pp))
cmap_eq = mcolors.ListedColormap(["#ffcccc", "#ccffcc", "#ccccff"])
im = ax.contourf(Ge_g, rho_g, N_eq, levels=[0, 1, 2, 3], cmap=cmap_eq, alpha=0.85)
ax.contour(Ge_g, rho_g, N_eq, levels=[1.5, 2.5], colors="black", linewidths=1.2)
ax.plot(140, 0.41, "k*", ms=12, label="Parámetros base")
plt.colorbar(im, ax=ax, label="Nº equilibrios")
ax.set_xlabel("Gₑ")
ax.set_ylabel("ρ")
ax.set_title("Número de equilibrios en espacio (Gₑ, ρ)")
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("fase4_1_Ge_rho.png", dpi=150, bbox_inches="tight")
print("  Guardado: fase4_1_Ge_rho.png")

# ══════════════════════════════════════════════════════════════════════════════
#  F4.2  —  ANÁLISIS PARAMÉTRICO: r1 y r2
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  F4.2 — Variación de r₁ y r₂")
print("=" * 65)

print("""
  Condición para existencia de P₁, P₂:
    Discriminante Δ = r₁² - 4·r₂·d₀  ≥  0

  Escenarios:
    Δ > 0  →  3 equilibrios (P₁ estable, P₂ silla, P₃ estable)
    Δ = 0  →  2 equilibrios (P₁=P₂ fusionados: bifurcación silla-nodo)
    Δ < 0  →  1 equilibrio  (solo P₃ estable: estado diabético inevitable)
""")

# Curva de bifurcación: r2 = r1²/(4·d0)
r1_bif = np.linspace(0, 0.003, 300)
r2_bif = r1_bif**2 / (4 * p_base["d0"])

# ── Tres escenarios representativos ──────────────────────────────────────────
escenarios = {
    "Δ > 0  (3 equilibrios)": {**p_base, "r1": 0.00084, "r2": 0.0000024},  # caso base
    "Δ = 0  (bifurcación silla-nodo)": {
        **p_base,
        "r1": 0.00084,
        "r2": 0.00084**2 / (4 * p_base["d0"]),
    },  # en la curva
    "Δ < 0  (solo P₃, diabético)": {
        **p_base,
        "r1": 0.00050,
        "r2": 0.0000060,
    },  # encima de la curva
}

for nombre, pp in escenarios.items():
    disc = pp["r1"] ** 2 - 4 * pp["r2"] * pp["d0"]
    pts = equilibrios(pp)
    print(f"  {nombre}")
    print(f"    r₁={pp['r1']:.5f}, r₂={pp['r2']:.7f},  Δ={disc:.3e}")
    for pt in pts:
        tipo, ev = clasificar(*pt, pp)
        print(f"    P=({pt[0]:.2f}, {pt[1]:.4f}, {pt[2]:.2f})  → {tipo}")
    print()

# ── FIGURA F4.2 ───────────────────────────────────────────────────────────────
fig42, axes42 = plt.subplots(2, 3, figsize=(16, 10))
fig42.suptitle(
    "F4.2 — Análisis Paramétrico: efecto de  r₁  y  r₂", fontsize=13, fontweight="bold"
)

# ── Fila 1: espacio (r1,r2) con curva de bifurcación + retratos ──────────────
ax = axes42[0, 0]
r1_g = np.linspace(0, 0.003, 200)
r2_g = np.linspace(0, 0.000012, 200)
R1, R2 = np.meshgrid(r1_g, r2_g)
DISC = R1**2 - 4 * R2 * p_base["d0"]
ax.contourf(
    R1 * 1e3,
    R2 * 1e6,
    DISC,
    levels=[-1e-6, 0, 1e-6],
    colors=["#ffcccc", "#ccffcc"],
    alpha=0.7,
)
ax.contour(R1 * 1e3, R2 * 1e6, DISC, levels=[0], colors="black", linewidths=2)
ax.plot(r1_bif * 1e3, r2_bif * 1e6, "k-", lw=2.5, label="Curva bif.: r₂=r₁²/4d₀")
# marcar los 3 escenarios
markers = {
    "Δ > 0  (3 equilibrios)": ("o", "#1976D2", "P₁,P₂,P₃"),
    "Δ = 0  (bifurcación silla-nodo)": ("s", "#F57F17", "Bif."),
    "Δ < 0  (solo P₃, diabético)": ("^", "#C62828", "Solo P₃"),
}
for nom, pp in escenarios.items():
    m, c, lbl = markers[nom]
    ax.plot(pp["r1"] * 1e3, pp["r2"] * 1e6, m, color=c, ms=10, label=lbl, zorder=5)
ax.set_xlabel("r₁  (×10⁻³)")
ax.set_ylabel("r₂  (×10⁻⁶)")
ax.set_title("Espacio (r₁, r₂) — curva de bifurcación\nVerde: 3 eq.   Rojo: solo P₃")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Fila 1 cols 1-2: retratos de fase para cada escenario ────────────────────
titulos = [
    "Δ > 0: P₁ estable, P₂ silla, P₃ estable",
    "Δ = 0: bifurcación silla-nodo",
    "Δ < 0: solo P₃ estable (diabético)",
]
cols_esc = ["#1976D2", "#F57F17", "#C62828"]

for col_idx, (nom, pp) in enumerate(escenarios.items()):
    if col_idx == 0:
        ax = axes42[0, 1]
    elif col_idx == 1:
        ax = axes42[0, 2]
    else:
        ax = axes42[1, 0]

    # streamplot 2D
    G_v = np.linspace(10, 750, 250)
    B_v = np.linspace(0, 450, 250)
    GG, BB = np.meshgrid(G_v, B_v)
    U, V = f2D(GG, BB, pp)
    speed = np.sqrt(U**2 + V**2)
    ax.streamplot(
        G_v,
        B_v,
        U,
        V,
        density=1.8,
        color=speed,
        cmap="Blues_r",
        linewidth=0.8,
        arrowsize=0.9,
        broken_streamlines=False,
        zorder=2,
    )

    # nulclinas
    ax.axhline(0, color="darkorange", lw=1.8, ls="-", zorder=3)
    G_nc = np.linspace(1, 749, 500)
    rk = pp["rho"] + pp["k"]
    B_nc = (
        ((pp["R0"] + pp["Ge"]) / G_nc - pp["EG0"])
        * (pp["alp"] + G_nc**2)
        * rk
        / (pp["SI"] * pp["sig"] * G_nc**2)
    )
    B_nc = np.where(B_nc > 0, B_nc, np.nan)
    ax.plot(G_nc, B_nc, "r-", lw=1.8, zorder=3)

    # β-nulclinas verticales (raíces de la cuadrática)
    disc_pp = pp["r1"] ** 2 - 4 * pp["r2"] * pp["d0"]
    if disc_pp >= 0:
        for Gv in [
            (pp["r1"] + np.sqrt(max(disc_pp, 0))) / (2 * pp["r2"]),
            (pp["r1"] - np.sqrt(max(disc_pp, 0))) / (2 * pp["r2"]),
        ]:
            ax.axvline(Gv, color="darkorange", lw=1.8, ls="--", zorder=3)
    elif disc_pp == 0:
        ax.axvline(
            pp["r1"] / (2 * pp["r2"]), color="darkorange", lw=2.5, ls="-", zorder=3
        )

    # puntos de equilibrio
    estilos = {
        "estable": dict(color="white", mec="navy", mew=2, ms=9),
        "silla": dict(color="white", mec="darkred", mew=2, ms=9),
        "inestable": dict(color="red", mec="red", mew=2, ms=9),
    }
    pts = equilibrios(pp)
    for pt in pts:
        tipo, _ = clasificar(*pt, pp)
        ax.plot(pt[0], pt[2], "o", zorder=7, **estilos.get(tipo, estilos["estable"]))

    ax.set_facecolor("#f5f5f5")
    ax.set_xlim(10, 750)
    ax.set_ylim(0, 450)
    ax.set_xlabel("G  (mg/dl)")
    ax.set_ylabel("β  (mgdl/dl)")
    ax.set_title(titulos[col_idx], fontsize=9, fontweight="bold")
    ax.grid(True, color="lightgray", ls=":", lw=0.5, alpha=0.7)

# ── Diagrama de bifurcación: G* de equilibrios vs r1  (r2 fijo) ──────────────
ax = axes42[1, 1]
r1_sweep = np.linspace(0.0001, 0.003, 400)
for r1v in r1_sweep:
    pp = {**p_base, "r1": r1v}
    disc = r1v**2 - 4 * pp["r2"] * pp["d0"]
    if disc >= 0:
        G_hi = (r1v + np.sqrt(disc)) / (2 * pp["r2"])
        G_lo = (r1v - np.sqrt(disc)) / (2 * pp["r2"])
        tipo_hi, _ = clasificar(
            G_hi,
            (pp["R0"] + pp["Ge"] - pp["EG0"] * G_hi) / (pp["SI"] * G_hi),
            (pp["rho"] + pp["k"])
            * ((pp["R0"] + pp["Ge"] - pp["EG0"] * G_hi) / (pp["SI"] * G_hi))
            * (pp["alp"] + G_hi**2)
            / (pp["sig"] * G_hi**2),
            pp,
        )
        c_hi = "#1976D2" if tipo_hi == "estable" else "#C62828"
        ax.plot(r1v * 1e3, G_hi, ".", color=c_hi, ms=2, alpha=0.7)
        ax.plot(r1v * 1e3, G_lo, ".", color="#C62828", ms=2, alpha=0.7)
r1_bif_val = 2 * np.sqrt(p_base["r2"] * p_base["d0"])
ax.axvline(
    r1_bif_val * 1e3,
    color="black",
    ls="--",
    lw=2,
    label=f"r₁_bif={r1_bif_val * 1e3:.3f}×10⁻³",
)
ax.set_xlabel("r₁  (×10⁻³)")
ax.set_ylabel("G* de equilibrio")
ax.set_title("Diagrama de bifurcación: G* vs r₁\n(azul=estable, rojo=silla/inestable)")
ax.legend()
ax.grid(True, alpha=0.3)

# ── Diagrama de bifurcación: G* vs r2 (r1 fijo) ──────────────────────────────
ax = axes42[1, 2]
r2_sweep = np.linspace(1e-8, 1.2e-5, 400)
for r2v in r2_sweep:
    pp = {**p_base, "r2": r2v}
    disc = pp["r1"] ** 2 - 4 * r2v * pp["d0"]
    if disc >= 0:
        G_hi = (pp["r1"] + np.sqrt(disc)) / (2 * r2v)
        G_lo = (pp["r1"] - np.sqrt(disc)) / (2 * r2v)
        tipo_hi, _ = clasificar(
            G_hi,
            (pp["R0"] + pp["Ge"] - pp["EG0"] * G_hi) / (pp["SI"] * G_hi),
            (pp["rho"] + pp["k"])
            * ((pp["R0"] + pp["Ge"] - pp["EG0"] * G_hi) / (pp["SI"] * G_hi))
            * (pp["alp"] + G_hi**2)
            / (pp["sig"] * G_hi**2),
            pp,
        )
        c_hi = "#1976D2" if tipo_hi == "estable" else "#C62828"
        ax.plot(r2v * 1e6, G_hi, ".", color=c_hi, ms=2, alpha=0.7)
        ax.plot(r2v * 1e6, G_lo, ".", color="#C62828", ms=2, alpha=0.7)
r2_bif_val = p_base["r1"] ** 2 / (4 * p_base["d0"])
ax.axvline(
    r2_bif_val * 1e6,
    color="black",
    ls="--",
    lw=2,
    label=f"r₂_bif={r2_bif_val * 1e6:.3f}×10⁻⁶",
)
ax.set_xlabel("r₂  (×10⁻⁶)")
ax.set_ylabel("G* de equilibrio")
ax.set_title("Diagrama de bifurcación: G* vs r₂\n(azul=estable, rojo=silla/inestable)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fase4_2_r1_r2.png", dpi=150, bbox_inches="tight")
print("  Guardado: fase4_2_r1_r2.png")

plt.show()
print("\nFase 4 completada.")
