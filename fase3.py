"""
Fase 3 — Análisis en el Espacio Fase
Sistema Células-β · Insulina · Glucosa en Presencia de Epinefrina

F3.1: Bosquejos locales de retratos fase alrededor de P1, P2, P3
F3.2: Simulaciones numéricas globales con comparación

Autores: Tecnológico de Monterrey, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────
# PARÁMETROS DEL SISTEMA (Tabla 2)
# ─────────────────────────────────────────
R0  = 864.0
Ge  = 140.0
EGO = 1.44
SI  = 0.72
sig = 43.2
alp = 20000.0
rho = 0.41
k   = 432.0
d0  = 0.06
r1  = 0.84e-3
r2  = 0.24e-5

# ─────────────────────────────────────────
# PUNTOS DE EQUILIBRIO
# ─────────────────────────────────────────
P1 = np.array([100.0000, 11.9444, 358.6734])
P2 = np.array([250.0000,  3.5778,  47.2715])
P3 = np.array([697.2222,  0.0000,   0.0000])

equilibria = {"$P_1$": P1, "$P_2$": P2, "$P_3$": P3}
colors_eq  = {"$P_1$": "#2ecc71", "$P_2$": "#e74c3c", "$P_3$": "#e67e22"}

# ─────────────────────────────────────────
# SISTEMA DE EDOs
# ─────────────────────────────────────────
def system(t, state):
    G, I, beta = state
    G    = max(G, 0.0)
    I    = max(I, 0.0)
    beta = max(beta, 0.0)
    dG   = R0 + Ge - (EGO + SI * I) * G
    dI   = (beta * sig * G**2) / (alp + G**2) - (rho + k) * I
    dB   = (-d0 + r1 * G - r2 * G**2) * beta
    return [dG, dI, dB]

# ─────────────────────────────────────────
# JACOBIANA SIMBÓLICA EN UN PUNTO
# ─────────────────────────────────────────
def jacobian(G, I, beta):
    J = np.zeros((3, 3))
    J[0, 0] = -(EGO + SI * I)
    J[0, 1] = -SI * G
    J[0, 2] = 0.0
    J[1, 0] = (2 * beta * sig * G * alp) / (alp + G**2)**2
    J[1, 1] = -(rho + k)
    J[1, 2] = (sig * G**2) / (alp + G**2)
    J[2, 0] = (r1 - 2 * r2 * G) * beta
    J[2, 1] = 0.0
    J[2, 2] = -d0 + r1 * G - r2 * G**2
    return J

# Valores propios en cada equilibrio
for name, P in equilibria.items():
    J  = jacobian(*P)
    ev = np.linalg.eigvals(J)
    print(f"Eigenvalores en {name}: {np.round(ev, 4)}")

# ═══════════════════════════════════════════════════════════════
# FIGURA 1 — RETRATOS FASE LOCALES (F3.1) en planos 2D
# ═══════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(3, 3, figsize=(16, 14))
fig1.suptitle(
    "Fase 3.1 — Retratos Fase Locales alrededor de los Puntos de Equilibrio\n"
    "Planos: G–I  |  G–β  |  I–β",
    fontsize=13, fontweight="bold", y=1.01
)

plane_labels = [("G (mg/dl)", "I (μU/ml)"), ("G (mg/dl)", "β (unidades)"), ("I (μU/ml)", "β (unidades)")]
plane_idx    = [(0, 1), (0, 2), (1, 2)]
eq_names     = list(equilibria.keys())
eq_points    = list(equilibria.values())
eq_colors    = list(colors_eq.values())

# Rango local alrededor de cada equilibrio (±δ proporcional)
local_scales = [
    [40, 8, 120],    # P1: δG=40, δI=8, δβ=120
    [80, 2,  30],    # P2: δG=80, δI=2, δβ=30
    [150, 0.5, 0.5], # P3: δG=150, δI=0.5, δβ=0.5
]

# Interpretaciones de estabilidad
stability_info = [
    ("Nodo Estable\n(Normoglucemia)", "#2ecc71"),
    ("Punto de Silla\n(Umbral Crítico)", "#e74c3c"),
    ("Nodo Estable\n(Diabetes Severa)", "#e67e22"),
]

T_local = 30   # tiempo de integración local
N_traj  = 14   # número de trayectorias locales por plano

for row, (P, sc, (stab_label, stab_color)) in enumerate(
        zip(eq_points, local_scales, stability_info)):

    for col, ((xi, yi), (xl, yl)) in enumerate(zip(plane_idx, plane_labels)):
        ax = axes[row, col]

        dx = sc[xi]
        dy = sc[yi]

        # Generar condiciones iniciales locales alrededor del equilibrio
        np.random.seed(42 + row * 10 + col)
        angles = np.linspace(0, 2 * np.pi, N_traj, endpoint=False)
        radii  = np.random.uniform(0.3, 1.0, N_traj)

        for angle, r in zip(angles, radii):
            ic_full = P.copy()
            ic_full[xi] = P[xi] + r * dx * np.cos(angle)
            ic_full[yi] = P[yi] + r * dy * np.sin(angle)
            # Mantener positivo
            ic_full = np.maximum(ic_full, 0.0)

            sol = solve_ivp(system, [0, T_local], ic_full,
                            max_step=0.05, dense_output=True)
            t_plot = np.linspace(0, T_local, 500)
            y_plot = sol.sol(t_plot)

            alpha_val = 0.65
            ax.plot(y_plot[xi], y_plot[yi], color=stab_color,
                    alpha=alpha_val, linewidth=0.9)
            # Flecha de dirección
            mid = len(t_plot) // 3
            ax.annotate("",
                xy=(y_plot[xi][mid+5], y_plot[yi][mid+5]),
                xytext=(y_plot[xi][mid], y_plot[yi][mid]),
                arrowprops=dict(arrowstyle="->", color=stab_color,
                                lw=0.8, alpha=0.7))

        # Punto de equilibrio
        ax.plot(P[xi], P[yi], "ko", markersize=7, zorder=5)
        ax.plot(P[xi], P[yi], "o", color=stab_color,
                markersize=5, zorder=6)

        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(
            f"{eq_names[row]}  —  {xl.split()[0]}–{yl.split()[0]}",
            fontsize=9, fontweight="bold"
        )
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

        # Etiqueta de estabilidad solo en columna 0
        if col == 0:
            ax.text(0.03, 0.96, stab_label,
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=stab_color, alpha=0.2))

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/F3_1_retratos_fase_locales.png",
            dpi=160, bbox_inches="tight")
plt.close()
print("✓ Figura 1 guardada: F3_1_retratos_fase_locales.png")


# ═══════════════════════════════════════════════════════════════
# FIGURA 2 — RETRATO FASE GLOBAL 3D + proyecciones (F3.1 global)
# ═══════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(18, 6))
fig2.suptitle(
    "Fase 3.1 — Diagrama Global: Cuencas de Atracción y Retrato Fase Completo",
    fontsize=13, fontweight="bold"
)

# Condiciones iniciales globales (diversas cuencas)
np.random.seed(0)
ICs_P1 = [  # cercanas a P1
    [150, 8, 200], [80, 15, 400], [120, 10, 300],
    [200, 5, 150], [100, 20, 500], [60, 12, 250],
]
ICs_P3 = [  # cercanas a P3 (masa β baja)
    [500,  1.5, 15], [400, 0.5, 10], [600, 2, 20],
    [300,  0.3,  5], [700, 1,   8],  [450, 1, 12],
]
ICs_all = ICs_P1 + ICs_P3
colors_traj = ["#2980b9"] * len(ICs_P1) + ["#c0392b"] * len(ICs_P3)

T_global = 350

# ── Panel izquierdo: 3D ──────────────────────────────────────
ax3d = fig2.add_subplot(131, projection="3d")

for ic, col in zip(ICs_all, colors_traj):
    sol = solve_ivp(system, [0, T_global], ic, max_step=0.5, dense_output=True)
    t_p = np.linspace(0, T_global, 1000)
    y_p = sol.sol(t_p)
    ax3d.plot(y_p[0], y_p[1], y_p[2], color=col, alpha=0.5, linewidth=0.8)
    ax3d.plot([ic[0]], [ic[1]], [ic[2]], ".", color=col, markersize=4)

for name, P, c in zip(eq_names, eq_points, eq_colors):
    ax3d.scatter(*P, color=c, s=80, zorder=10, edgecolors="k", linewidths=0.8)
    ax3d.text(P[0]+10, P[1]+0.5, P[2]+15, name, fontsize=8, color=c, fontweight="bold")

ax3d.set_xlabel("G (mg/dl)", fontsize=8, labelpad=4)
ax3d.set_ylabel("I (μU/ml)", fontsize=8, labelpad=4)
ax3d.set_zlabel("β", fontsize=8, labelpad=4)
ax3d.set_title("Espacio de Estados 3D", fontsize=10, fontweight="bold")
ax3d.tick_params(labelsize=7)
ax3d.view_init(elev=22, azim=-55)

# ── Panel central: plano G–β (más informativo biológicamente) ──
ax_gb = fig2.add_subplot(132)
for ic, col in zip(ICs_all, colors_traj):
    sol = solve_ivp(system, [0, T_global], ic, max_step=0.5, dense_output=True)
    t_p = np.linspace(0, T_global, 1000)
    y_p = sol.sol(t_p)
    ax_gb.plot(y_p[0], y_p[2], color=col, alpha=0.55, linewidth=1.0)
    ax_gb.plot(ic[0], ic[2], ".", color=col, markersize=5)
    # flecha en t≈T/3
    mid = 300
    ax_gb.annotate("",
        xy=(y_p[0][mid+8], y_p[2][mid+8]),
        xytext=(y_p[0][mid], y_p[2][mid]),
        arrowprops=dict(arrowstyle="->", color=col, lw=0.9, alpha=0.7))

for name, P, c in zip(eq_names, eq_points, eq_colors):
    ax_gb.plot(P[0], P[2], "o", color=c, markersize=9,
               markeredgecolor="k", markeredgewidth=0.7, zorder=8)
    ax_gb.text(P[0]+8, P[2]+8, name, fontsize=9, color=c, fontweight="bold")

# Separatriz aproximada (línea vertical en P2_G)
ax_gb.axvline(P2[0], color="#7f8c8d", linestyle="--", linewidth=1.0,
              alpha=0.6, label="Separatriz ≈ $G^*_2$")
ax_gb.set_xlabel("G (mg/dl)", fontsize=10)
ax_gb.set_ylabel("β (unidades)", fontsize=10)
ax_gb.set_title("Plano G – β  (Cuencas de Atracción)", fontsize=10, fontweight="bold")
ax_gb.legend(fontsize=8, loc="upper right")
ax_gb.grid(True, alpha=0.25)

# Anotaciones de cuencas
ax_gb.text(130, 320, "Cuenca de $P_1$\n(Recuperación)", fontsize=8,
           color="#2980b9", alpha=0.85, ha="center")
ax_gb.text(520, 50, "Cuenca de $P_3$\n(Diabetes)", fontsize=8,
           color="#c0392b", alpha=0.85, ha="center")

# ── Panel derecho: plano G–I ──────────────────────────────────
ax_gi = fig2.add_subplot(133)
for ic, col in zip(ICs_all, colors_traj):
    sol = solve_ivp(system, [0, T_global], ic, max_step=0.5, dense_output=True)
    t_p = np.linspace(0, T_global, 1000)
    y_p = sol.sol(t_p)
    ax_gi.plot(y_p[0], y_p[1], color=col, alpha=0.55, linewidth=1.0)
    ax_gi.plot(ic[0], ic[1], ".", color=col, markersize=5)

for name, P, c in zip(eq_names, eq_points, eq_colors):
    ax_gi.plot(P[0], P[1], "o", color=c, markersize=9,
               markeredgecolor="k", markeredgewidth=0.7, zorder=8)
    ax_gi.text(P[0]+8, P[1]+0.3, name, fontsize=9, color=c, fontweight="bold")

ax_gi.set_xlabel("G (mg/dl)", fontsize=10)
ax_gi.set_ylabel("I (μU/ml)", fontsize=10)
ax_gi.set_title("Plano G – I  (Insulina vs Glucosa)", fontsize=10, fontweight="bold")
ax_gi.grid(True, alpha=0.25)

# Leyenda global
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="#2980b9", linewidth=2, label="Trayectorias → $P_1$ (normoglucemia)"),
    Line2D([0], [0], color="#c0392b", linewidth=2, label="Trayectorias → $P_3$ (diabetes severa)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
           markersize=9, markeredgecolor="k", label="$P_1$ — Nodo Estable"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
           markersize=9, markeredgecolor="k", label="$P_2$ — Punto de Silla"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22",
           markersize=9, markeredgecolor="k", label="$P_3$ — Nodo Estable"),
]
fig2.legend(handles=legend_elements, loc="lower center",
            ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.04),
            framealpha=0.9)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/F3_1_retrato_fase_global.png",
            dpi=160, bbox_inches="tight")
plt.close()
print("✓ Figura 2 guardada: F3_1_retrato_fase_global.png")


# ═══════════════════════════════════════════════════════════════
# FIGURA 3 — SIMULACIONES NUMÉRICAS GLOBALES (F3.2)
# ═══════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9))
fig3.suptitle(
    "Fase 3.2 — Simulaciones Numéricas: Convergencia a los Puntos de Equilibrio\n"
    "Comparación de trayectorias temporales desde distintas condiciones iniciales",
    fontsize=12, fontweight="bold"
)

IC_sets = {
    r"$\rightarrow P_1$: $(G_0,I_0,\beta_0)$": {
        "ics": [
            ([200,  4.5, 65],  "#1a6bbf"),
            ([150,  8.0, 200], "#2980b9"),
            ([80,   12., 300], "#5dade2"),
            ([210,  10., 70],  "#85c1e9"),
        ],
        "eq": P1,
        "eq_name": "$P_1 = (100,\\ 11.94,\\ 358.67)$",
        "col": 0,
    },
    r"$\rightarrow P_3$: $(G_0,I_0,\beta_0)$": {
        "ics": [
            ([500, 1.5, 18],  "#a93226"),
            ([400, 0.5, 10],  "#c0392b"),
            ([600, 2.0, 20],  "#e74c3c"),
            ([350, 0.3,  5],  "#f1948a"),
        ],
        "eq": P3,
        "eq_name": "$P_3 = (697.22,\\ 0,\\ 0)$",
        "col": 1,
    },
}

var_names  = ["G (mg/dl)", "I (μU/ml)", "β (unidades)"]
var_colors = ["#2c3e50", "#8e44ad", "#16a085"]

row_labels = ["Glucosa G(t)", "Insulina I(t)", "Células β(t)"]

for set_label, info in IC_sets.items():
    col = info["col"]
    for row in range(3):
        ax = axes3[row // 2 * 2 + row % 2, col] if row < 3 else None

    # Graficar las 3 filas para este conjunto de ICs
    for var_idx in range(3):
        ax = axes3[var_idx % 2, col + (1 if var_idx == 2 else 0)] if var_idx < 2 else None

# Rehacer con estructura más clara
fig3, axes3 = plt.subplots(3, 2, figsize=(14, 11))
fig3.suptitle(
    "Fase 3.2 — Simulaciones Numéricas: Convergencia Temporal\n"
    "Izquierda: Trayectorias → $P_1$ (Normoglucemia) | Derecha: Trayectorias → $P_3$ (Diabetes Severa)",
    fontsize=11, fontweight="bold"
)

IC_toward_P1 = [
    ([200,  4.5,  65], "#1a6bbf", r"$G_0=200,\ I_0=4.5,\ \beta_0=65$"),
    ([150,  8.0, 200], "#2980b9", r"$G_0=150,\ I_0=8,\ \beta_0=200$"),
    ([80,  12.0, 300], "#5dade2", r"$G_0=80,\ I_0=12,\ \beta_0=300$"),
    ([210,  10.,  70], "#85c1e9", r"$G_0=210,\ I_0=10,\ \beta_0=70$"),
]
IC_toward_P3 = [
    ([500, 1.5, 18], "#a93226", r"$G_0=500,\ I_0=1.5,\ \beta_0=18$"),
    ([400, 0.5, 10], "#c0392b", r"$G_0=400,\ I_0=0.5,\ \beta_0=10$"),
    ([600, 2.0, 20], "#e74c3c", r"$G_0=600,\ I_0=2,\ \beta_0=20$"),
    ([350, 0.3,  5], "#f1948a", r"$G_0=350,\ I_0=0.3,\ \beta_0=5$"),
]

eq_lines_P1 = [P1[0], P1[1], P1[2]]
eq_lines_P3 = [P3[0], P3[1], P3[2]]
eq_styles   = dict(linestyle="--", linewidth=1.3, alpha=0.85)
ylabels     = ["G(t)  [mg/dl]", "I(t)  [μU/ml]", "β(t)  [unidades]"]
var_titles  = ["Glucosa", "Insulina", "Masa de Células β"]

for var_idx in range(3):
    # Columna 0: hacia P1
    ax0 = axes3[var_idx, 0]
    for ic, col, lbl in IC_toward_P1:
        sol = solve_ivp(system, [0, 350], ic, max_step=0.5, dense_output=True)
        t_p = np.linspace(0, 350, 1000)
        y_p = sol.sol(t_p)
        ax0.plot(t_p, y_p[var_idx], color=col, linewidth=1.4, label=lbl, alpha=0.85)

    ax0.axhline(eq_lines_P1[var_idx], color="#2ecc71", **eq_styles,
                label=f"$P_1^{{({['G','I','β'][var_idx]})}}={eq_lines_P1[var_idx]:.2f}$")
    ax0.set_ylabel(ylabels[var_idx], fontsize=9)
    ax0.set_title(f"{var_titles[var_idx]} — Hacia $P_1$", fontsize=9, fontweight="bold")
    ax0.grid(True, alpha=0.25)
    ax0.tick_params(labelsize=8)
    if var_idx == 0:
        ax0.legend(fontsize=6.5, loc="upper right", ncol=2)

    # Columna 1: hacia P3
    ax1 = axes3[var_idx, 1]
    for ic, col, lbl in IC_toward_P3:
        sol = solve_ivp(system, [0, 350], ic, max_step=0.5, dense_output=True)
        t_p = np.linspace(0, 350, 1000)
        y_p = sol.sol(t_p)
        ax1.plot(t_p, y_p[var_idx], color=col, linewidth=1.4, label=lbl, alpha=0.85)

    ax1.axhline(eq_lines_P3[var_idx], color="#e67e22", **eq_styles,
                label=f"$P_3^{{({['G','I','β'][var_idx]})}}={eq_lines_P3[var_idx]:.2f}$")
    ax1.set_ylabel(ylabels[var_idx], fontsize=9)
    ax1.set_title(f"{var_titles[var_idx]} — Hacia $P_3$", fontsize=9, fontweight="bold")
    ax1.grid(True, alpha=0.25)
    ax1.tick_params(labelsize=8)
    if var_idx == 0:
        ax1.legend(fontsize=6.5, loc="upper left", ncol=2)

for ax in axes3[2, :]:
    ax.set_xlabel("Tiempo (días)", fontsize=9)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/F3_2_simulaciones_convergencia.png",
            dpi=160, bbox_inches="tight")
plt.close()
print("✓ Figura 3 guardada: F3_2_simulaciones_convergencia.png")


# ═══════════════════════════════════════════════════════════════
# FIGURA 4 — CAMPO VECTORIAL 2D (F3.1 bosquejo + nullclines)
# ═══════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle(
    "Fase 3.1 — Campo Vectorial y Nullclines en el Plano G–β\n"
    "(Sección con I = I*(G) de equilibrio, β libre)",
    fontsize=12, fontweight="bold"
)

# Plano G–β con I fijada en el valor de equilibrio de cada G
G_range = np.linspace(5, 750, 26)
B_range = np.linspace(0, 500, 26)
GG, BB = np.meshgrid(G_range, B_range)

# Campo vectorial: dG/dt y dβ/dt evaluados con I = I*(G) del equilibrio
def I_eq(G):
    """I de equilibrio aproximada: I = (R0+Ge - EGO*G)/(SI*G), válida para G pequeña"""
    I_val = (R0 + Ge - EGO * G) / (SI * G + 1e-9)
    return np.maximum(I_val, 0.0)

I_GG = I_eq(GG)
dG   = R0 + Ge - (EGO + SI * I_GG) * GG
dB   = (-d0 + r1 * GG - r2 * GG**2) * BB

# Normalizar
mag  = np.sqrt(dG**2 + dB**2) + 1e-12
dGn  = dG / mag
dBn  = dB / mag

ax = axes4[0]
q = ax.quiver(GG, BB, dGn, dBn,
              np.sqrt(dGn**2 + dBn**2),
              cmap="coolwarm", alpha=0.75, scale=30)
plt.colorbar(q, ax=ax, label="Magnitud normalizada")

# Nullcline dβ/dt = 0: -d0 + r1*G - r2*G² = 0  → G ≈ 100 y G ≈ 250
ax.axvline(100,  color="#2ecc71", linestyle="--", linewidth=1.5, label="Nullcline β: G=100")
ax.axvline(250,  color="#e74c3c", linestyle="--", linewidth=1.5, label="Nullcline β: G=250")
ax.axvline(697.22, color="#e67e22", linestyle=":",  linewidth=1.5, label="$G^*_3=697.22$")

for name, P, c in zip(eq_names, eq_points, eq_colors):
    if P[2] < 510:
        ax.plot(P[0], P[2], "o", color=c, markersize=10,
                markeredgecolor="k", markeredgewidth=0.8, zorder=9)
        ax.text(P[0]+10, P[2]+10, name, fontsize=9, color=c, fontweight="bold")

ax.set_xlim(5, 750)
ax.set_ylim(0, 500)
ax.set_xlabel("G (mg/dl)", fontsize=11)
ax.set_ylabel("β (unidades)", fontsize=11)
ax.set_title("Campo Vectorial en Plano G–β", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.2)

# Panel derecho: diagrama de bifurcación informal (β* vs G*)
ax2 = axes4[1]
G_vals = np.linspace(5, 750, 500)
# Curva de equilibrio β*(G) desde ecuación de insulina + glucosa
beta_curve = np.zeros_like(G_vals)
for i, G in enumerate(G_vals):
    I_v = max((R0 + Ge - EGO * G) / (SI * G + 1e-9), 0.0)
    denom = sig * G**2
    if denom > 0:
        beta_curve[i] = (rho + k) * I_v * (alp + G**2) / denom
    else:
        beta_curve[i] = 0.0

# Región biológicamente viable (β > 0, dβ/dt > 0 en β > 0)
growth_cond = -d0 + r1 * G_vals - r2 * G_vals**2
viable = growth_cond > 0

ax2.plot(G_vals, beta_curve, color="#2c3e50", linewidth=2.0,
         label=r"$\beta^*(G)$ de equilibrio")
ax2.fill_between(G_vals, 0, beta_curve,
                 where=viable, alpha=0.15, color="#2ecc71",
                 label="Región viable ($\dot{\\beta}>0$)")
ax2.fill_between(G_vals, 0, beta_curve,
                 where=~viable, alpha=0.12, color="#e74c3c",
                 label="Región inviable ($\dot{\\beta}<0$)")

ax2.axvline(100,    color="#2ecc71", linestyle="--", linewidth=1.4, alpha=0.8)
ax2.axvline(250,    color="#e74c3c", linestyle="--", linewidth=1.4, alpha=0.8)
ax2.axvline(697.22, color="#e67e22", linestyle=":",  linewidth=1.4, alpha=0.8)

for name, P, c in zip(eq_names, eq_points, eq_colors):
    ax2.plot(P[0], P[2], "o", color=c, markersize=10,
             markeredgecolor="k", markeredgewidth=0.8, zorder=9)
    ax2.text(P[0]+8, P[2]+8, name, fontsize=9, color=c, fontweight="bold")

ax2.set_xlim(0, 750)
ax2.set_ylim(0, 500)
ax2.set_xlabel("G (mg/dl)", fontsize=11)
ax2.set_ylabel("β* (unidades)", fontsize=11)
ax2.set_title("Curva de Equilibrio β*(G) y Regiones de Viabilidad", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/F3_1_campo_vectorial_nullclines.png",
            dpi=160, bbox_inches="tight")
plt.close()

