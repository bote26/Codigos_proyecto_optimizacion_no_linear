import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ── Parámetros del sistema (Tabla 2 del artículo) ──────────────────────────
params = {
    "R0":  864,        # Tasa neta de producción de glucosa (G=0)  [mgd/dl]
    "Ge":  140,        # Aumento de glucosa por epinefrina          [mgd/dl]
    "EG0": 1.44,       # Eficacia total de la glucosa (I=0)         [d⁻¹]
    "SI":  0.72,       # Sensibilidad total a la insulina           [mlμ/dU]
    "sig": 43.2,       # Tasa máxima de secreción de insulina       [μUm/dl]
    "alp": 20000.0,    # Punto de inflexión de la sigmoide          [mg²dl/l²]
    "rho": 0.41,       # Eficacia de epinefrina suprimiendo I (41%) [d⁻¹]
    "k":   432.0,      # Tasa de aclaramiento de I                  [d⁻¹]
    "d0":  0.06,       # Tasa de muerte natural de células β        [d⁻¹]
    "r1":  0.00084,    # Tolerancia a la glucosa de células β       [mdl/dg]
    "r2":  0.0000024,  # Inhibición de células β por glucosa alta   [mgl²/dg²]
}


# ── Sistema de EDOs (ecuaciones 1-3 del artículo) ───────────────────────────
# NOTA: la ec. (2) usa -(ρ + k)·I, NO -(ρ - k)·I
def sistema(t, y, p):
    G, I, beta = y
    dG    = p["R0"] + p["Ge"] - (p["EG0"] + p["SI"] * I) * G
    dI    = (beta * p["sig"] * G**2) / (p["alp"] + G**2) - (p["rho"] - p["k"]) * I
    dBeta = (-p["d0"] + p["r1"] * G - p["r2"] * G**2) * beta
    return [dG, dI, dBeta]



# ── Jacobiano general ────────────────────────────────────────────────────────
def jacobiano(G, I, beta, p):
    denom = p["alp"] + G**2
    J = np.array([
        [-(p["EG0"] + p["SI"] * I),
         -p["SI"] * G,
         0.0],
        [beta * p["sig"] * 2 * p["alp"] * G / denom**2,
         -(p["rho"] + p["k"]),
         p["sig"] * G**2 / denom],
        [(p["r1"] - 2 * p["r2"] * G) * beta,
         0.0,
         -p["d0"] + p["r1"] * G - p["r2"] * G**2],
    ])
    return J


# ── Puntos de equilibrio analíticos ─────────────────────────────────────────
def puntos_equilibrio(p):
    """
    P3 (trivial):  β=0, I=0  →  G* = (R0+Ge)/EG0
    P1, P2 (no triviales):  resolver r2·G²-r1·G+d0=0, luego I y β.
    """
    G3 = (p["R0"] + p["Ge"]) / p["EG0"]
    puntos = [("P3", G3, 0.0, 0.0)]

    disc = p["r1"]**2 - 4.0 * p["r2"] * p["d0"]
    if disc >= 0:
        for G in [(p["r1"] + np.sqrt(disc)) / (2 * p["r2"]),
                  (p["r1"] - np.sqrt(disc)) / (2 * p["r2"])]:
            I    = (p["R0"] + p["Ge"] - p["EG0"] * G) / (p["SI"] * G)
            beta = (p["rho"] + p["k"]) * I * (p["alp"] + G**2) / (p["sig"] * G**2)
            if I > 0 and beta > 0:
                puntos.append((None, G, I, beta))

    # Ordenar por G creciente y etiquetar P1 < P2 < P3
    puntos.sort(key=lambda x: x[1])
    etiquetados = []
    idx = 1
    for _, G, I, beta in puntos:
        if G == G3:
            etiquetados.append(("P3", G, I, beta))
        else:
            etiquetados.append((f"P{idx}", G, I, beta))
            idx += 1
    return etiquetados


# ── Límites de localización LCCI (Teorema 3, con q1=1) ───────────────────────
def limites_lcci(p, q1=1.0):
    Gmax    = (p["R0"] + p["Ge"]) / p["EG0"]
    betamax = q1 * (Gmax + (p["R0"] + p["Ge"]) / p["d0"])
    Imax    = betamax * p["sig"] * Gmax**2 / (p["alp"] * (p["rho"] + p["k"]))
    return Gmax, Imax, betamax


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTADOS ANALÍTICOS
# ══════════════════════════════════════════════════════════════════════════════
equilibrios = puntos_equilibrio(params)
Gmax, Imax, betamax = limites_lcci(params)

print("=" * 65)
print("  PUNTOS DE EQUILIBRIO  (compara con Tabla 1 del artículo)")
print("=" * 65)
for nombre, G, I, beta in equilibrios:
    print(f"\n  {nombre}: G={G:.4f},  I={I:.4f},  β={beta:.4f}")
    J  = jacobiano(G, I, beta, params)
    ev = np.linalg.eigvals(J)
    for j, lam in enumerate(ev):
        signo = "< 0  estable" if lam.real < 0 else "> 0  inestable"
        print(f"       λ{j+1} = {lam:.4f}  Re={lam.real:+.4f}  {signo}")

print("\n" + "=" * 65)
print("  LÍMITES DE LOCALIZACIÓN (LCCI, q1=1)  —  Teorema 3")
print("=" * 65)
print(f"  Gmáx  ≤ {Gmax:.2f}   mg/dl")
print(f"  Imáx  ≤ {Imax:.2f}  μUm/dl")
print(f"  βmáx  ≤ {betamax:.2f}  mgdl/dl")

# ══════════════════════════════════════════════════════════════════════════════
#  SIMULACIONES NUMÉRICAS
# ══════════════════════════════════════════════════════════════════════════════
t_span = (0, 350)
t_eval = np.linspace(0, 350, 3500)

# Condiciones iniciales del artículo
y0_P1 = [200.0, 4.5,  65.0]   # converge a P1 (Figura 1 del artículo)
y0_P3 = [500.0, 1.5,  18.0]   # converge a P3 (Figura 3 del artículo)

# Solver Radau: apropiado para sistemas rígidos (λ2 ≈ -432)
sol_P1 = solve_ivp(sistema, t_span, y0_P1, t_eval=t_eval,
                   args=(params,), method='Radau', rtol=1e-8, atol=1e-10)
sol_P3 = solve_ivp(sistema, t_span, y0_P3, t_eval=t_eval,
                   args=(params,), method='Radau', rtol=1e-8, atol=1e-10)

# Extraer valores de P1 y P3
p1 = next((e for e in equilibrios if e[0] == "P1"), None)
p3 = next((e for e in equilibrios if e[0] == "P3"), None)

# ── Figura 1 del artículo: trayectorias → P1 ─────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(sol_P1.t, sol_P1.y[0], 'b-', linewidth=2, label='G(t)')
ax1.plot(sol_P1.t, sol_P1.y[1], 'r-', linewidth=2, label='I(t)')
ax1.plot(sol_P1.t, sol_P1.y[2], 'g-', linewidth=2, label='β(t)')
if p1:
    ax1.axhline(p1[1], color='b', linestyle='--', alpha=0.6, label=f'G* = {p1[1]:.2f}')
    ax1.axhline(p1[2], color='r', linestyle='--', alpha=0.6, label=f'I* = {p1[2]:.2f}')
    ax1.axhline(p1[3], color='g', linestyle='--', alpha=0.6, label=f'β* = {p1[3]:.2f}')
ax1.set_title("Figura 1 — Convergencia al punto de equilibrio P₁\n"
              "G(0) = 200,  I(0) = 4.5,  β(0) = 65")
ax1.set_xlabel("Tiempo [días]")
ax1.set_ylabel("Variables de estado")
ax1.legend(loc='center right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 350)
plt.tight_layout()
plt.savefig("figura1_P1.png", dpi=150)

# ── Figura 2 del artículo: dominio de localización LCCI ──────────────────────
fig2, (axG, axI, axB) = plt.subplots(3, 1, figsize=(8, 9))
fig2.suptitle("Figura 2 — Dominio de Localización  K = {K₁ ∩ K₂ ∩ K₃}\n"
              "G(0) = 200,  I(0) = 4.5,  β(0) = 65", fontsize=12)

axG.plot(sol_P1.t, sol_P1.y[0], 'b-', linewidth=2, label='G(t)')
axG.axhline(Gmax, color='r', linestyle='--', linewidth=2, label=f'Gmáx = {Gmax:.2f}')
axG.set_ylabel("Glucosa [mg/dl]")
axG.legend(); axG.grid(True, alpha=0.3); axG.set_xlim(0, 350)

axI.semilogy(sol_P1.t, np.maximum(sol_P1.y[1], 1e-12), 'r-', linewidth=2, label='I(t)')
axI.axhline(Imax, color='r', linestyle='--', linewidth=2, label=f'Imáx = {Imax:.2f}')
axI.set_ylabel("Insulina [μUm/dl]")
axI.legend(); axI.grid(True, alpha=0.3); axI.set_xlim(0, 350)

axB.plot(sol_P1.t, sol_P1.y[2], 'g-', linewidth=2, label='β(t)')
axB.axhline(betamax, color='r', linestyle='--', linewidth=2, label=f'βmáx = {betamax:.2f}')
axB.set_ylabel("Células β [mgdl/dl]")
axB.set_xlabel("Tiempo [días]")
axB.legend(); axB.grid(True, alpha=0.3); axB.set_xlim(0, 350)

plt.tight_layout()
plt.savefig("figura2_lcci.png", dpi=150)

# ── Figura 3 del artículo: trayectorias → P3 ─────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(sol_P3.t, sol_P3.y[0], 'b-', linewidth=2, label='G(t)')
ax3.plot(sol_P3.t, sol_P3.y[1], 'r-', linewidth=2, label='I(t)')
ax3.plot(sol_P3.t, sol_P3.y[2], 'g-', linewidth=2, label='β(t)')
if p3:
    ax3.axhline(p3[1], color='b', linestyle='--', alpha=0.6, label=f'G* = {p3[1]:.2f}')
    ax3.axhline(p3[2], color='r', linestyle='--', alpha=0.6, label=f'I* = {p3[2]:.2f}')
    ax3.axhline(p3[3], color='g', linestyle='--', alpha=0.6, label=f'β* = {p3[3]:.2f}')
ax3.set_title("Figura 3 — Convergencia al punto de equilibrio P₃\n"
              "G(0) = 500,  I(0) = 1.5,  β(0) = 18")
ax3.set_xlabel("Tiempo [días]")
ax3.set_ylabel("Variables de estado")
ax3.legend(loc='center right')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 350)
plt.tight_layout()
plt.savefig("figura3_P3.png", dpi=150)

plt.show()
print("\nGráficas guardadas: figura1_P1.png, figura2_lcci.png, figura3_P3.png")
