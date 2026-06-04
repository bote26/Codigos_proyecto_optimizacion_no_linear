import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ── Parámetros ──────────────────────────────────────────────────────────────
p = {
    "R0":  864,
    "Ge":  140,
    "EG0": 1.44,
    "SI":  0.72,
    "sig": 43.2,
    "alp": 20000.0,
    "rho": 0.41,
    "k":   432.0,
    "d0":  0.06,
    "r1":  0.00084,
    "r2":  0.0000024,
}

# ── Sistema completo: dI usa -(rho + k)·I ───────────────────────────────────
def sistema(t, y, p):
    G, I, beta = y
    dG    = p["R0"] + p["Ge"] - (p["EG0"] + p["SI"] * I) * G
    dI    = (beta * p["sig"] * G**2) / (p["alp"] + G**2) - (p["rho"] + p["k"]) * I
    dBeta = (-p["d0"] + p["r1"] * G - p["r2"] * G**2) * beta
    return [dG, dI, dBeta]

# ══════════════════════════════════════════════════════════════════════════════
#  F1.1 — ANÁLISIS CON G = 0
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  F1.1 — Análisis cuando G = 0")
print("=" * 60)

dG_en_G0 = p["R0"] + p["Ge"]
print(f"\n  dG/dt|_{{G=0}} = R0 + Ge = {p['R0']} + {p['Ge']} = {dG_en_G0}")
print(f"  → dG/dt = {dG_en_G0} > 0  siempre")
print("  → NO existe punto de equilibrio con G = 0")
print("    (el sistema siempre empuja G hacia valores positivos)")

rho_pk = p["rho"] + p["k"]         # = 0.41 + 432 = 432.41
print("\n  Soluciones analíticas con G = 0:")
print(f"    β(t) = β₀ · exp(-d₀·t)         con  d₀   = {p['d0']}      → β → 0  (decae)")
print(f"    I(t) = I₀ · exp(-(ρ+k)·t)      con  ρ+k  = {rho_pk:.2f}")
print(f"         = I₀ · exp(-{rho_pk:.2f}·t)  → I → 0  (decae muy rápido)")

# ══════════════════════════════════════════════════════════════════════════════
#  F1.2 — ANÁLISIS CON I = 0
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  F1.2 — Análisis cuando I = 0")
print("=" * 60)

G_star = (p["R0"] + p["Ge"]) / p["EG0"]
print(f"\n  De dG/dt = 0  →  G* = (R0 + Ge) / EG0")
print(f"                 G* = ({p['R0']} + {p['Ge']}) / {p['EG0']} = {G_star:.4f}")
print(f"\n  De dI/dt = 0 con I=0  →  β*·σ·G*² / (α + G*²) = 0  →  β* = 0")
print(f"\n  De dβ/dt = 0 con β=0  →  se satisface automáticamente ✓")
print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │  PUNTO DE EQUILIBRIO P*                 │")
print(f"  │  G* = (R0+Ge)/EG0 = {G_star:.4f}        │")
print(f"  │  I* = 0                                 │")
print(f"  │  β* = 0                                 │")
print(f"  └─────────────────────────────────────────┘")

lam1 = -p["EG0"]
lam2 = -(p["rho"] + p["k"])
lam3 = -p["d0"] + p["r1"]*G_star - p["r2"]*G_star**2
j12  = -p["SI"] * G_star
j23  = p["sig"] * G_star**2 / (p["alp"] + G_star**2)

J = np.array([
    [lam1,  j12,  0.0 ],
    [0.0,   lam2, j23 ],
    [0.0,   0.0,  lam3],
])

print(f"\n  Jacobiano en P* (matriz triangular):")
print(f"    [{J[0,0]:8.4f}  {J[0,1]:8.4f}  {J[0,2]:6.2f}]")
print(f"    [{J[1,0]:8.4f}  {J[1,1]:8.4f}  {J[1,2]:6.4f}]")
print(f"    [{J[2,0]:8.4f}  {J[2,1]:8.4f}  {J[2,2]:6.4f}]")

print("\n  Valores propios:")
for val, formula in [(lam1, f"-EG0           = {lam1:.4f}"),
                     (lam2, f"-(ρ+k)         = {lam2:.4f}"),
                     (lam3, f"-d0+r1G*-r2G*² = {lam3:.4f}")]:
    signo = "< 0  estable" if val < 0 else "> 0  INESTABLE"
    print(f"    λ = {formula}  →  {signo}")

print("\n  Clasificación de P*:")
print("  → NODO ESTABLE (las 3 direcciones son estables)")
print("  → P* es asintóticamente estable localmente")

# ══════════════════════════════════════════════════════════════════════════════
#  GRÁFICAS — todas en una sola figura con 4 subplots
# ══════════════════════════════════════════════════════════════════════════════
# β decae lento (d0=0.06), I decae muy rápido (rho+k=432.41)
# Se usan ventanas de tiempo distintas para apreciar cada curva
t_beta  = np.linspace(0, 100,   500)   # β: ventana larga  (τ = 1/0.06  ≈ 17 días)
t_I     = np.linspace(0, 0.015, 500)   # I: ventana corta  (τ = 1/432.41 ≈ 0.002)
beta0, I0 = 1.0, 1.0
beta_analitica = beta0 * np.exp(-p["d0"]  * t_beta)
I_analitica    = I0   * np.exp(-rho_pk   * t_I)     # decae a 0 muy rápido

t_span = (0, 30)
t_eval = np.linspace(0, 30, 1000)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Fase 1 — Análisis en Valores Críticos", fontsize=14)

# ── Panel 1: β(t) cuando G=0 ─────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(t_beta, beta_analitica, 'g-', linewidth=2)
ax.set_title("F1.1: β(t) cuando G = 0\n" + r"$\beta(t) = \beta_0\,e^{-d_0 t}$  →  decae a 0")
ax.set_xlabel("Tiempo")
ax.set_ylabel(r"$\beta(t)$")
ax.grid(True, alpha=0.3)

# ── Panel 2: I(t) cuando G=0 ─────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(t_I, I_analitica, 'm-', linewidth=2)
ax.set_title("F1.1: I(t) cuando G = 0\n" + r"$I(t) = I_0\,e^{-(\rho+k)t}$  →  decae a 0 muy rápido")
ax.set_xlabel("Tiempo")
ax.set_ylabel("$I(t)$")
ax.grid(True, alpha=0.3)

# ── Panel 3: G(t) analítica con I=0, β=0 ────────────────────────────────────
ax = axes[1, 0]
for G0, color in zip([0, 200, 500, 800], ['blue', 'green', 'orange', 'red']):
    G_analitica = G_star + (G0 - G_star) * np.exp(-p["EG0"] * t_eval)
    ax.plot(t_eval, G_analitica, color=color, linewidth=2, label=f"G(0)={G0}")
ax.axhline(G_star, color='k', linestyle='--', linewidth=2, label=f"G* = {G_star:.2f}")
ax.set_title(f"F1.2: G(t) con I=0, β=0\n→ converge a G* = {G_star:.2f} desde cualquier G(0)")
ax.set_xlabel("Tiempo")
ax.set_ylabel("G(t)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Panel 4: Verificación numérica con distintos β(0) ───────────────────────
ax = axes[1, 1]

def blow_up(t, y, p):
    return 1e8 - max(abs(y[i]) for i in range(3))
blow_up.terminal = True
blow_up.direction = -1

for beta0_val, color in zip([0.0, 0.5, 1.0, 2.0], ['blue', 'green', 'orange', 'red']):
    sol = solve_ivp(
        sistema, t_span, [G_star * 0.7, 0.0, beta0_val],
        t_eval=t_eval, args=(p,), events=blow_up,
        dense_output=True, rtol=1e-6, atol=1e-9
    )
    ax.plot(sol.t, sol.y[0], color=color, linewidth=2, label=f"β(0)={beta0_val}")

ax.axhline(G_star, color='k', linestyle='--', linewidth=2, label=f"G* = {G_star:.2f}")
ax.set_title("F1.2: Verificación numérica G(t)\npara distintos β(0) con I(0)=0")
ax.set_xlabel("Tiempo")
ax.set_ylabel("G(t)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fase1_analisis.png", dpi=150)
plt.show()

print("\nGráfica guardada: fase1_analisis.png")
