import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# 1. ESPECIFICAÇÕES DO DIVISOR DE POTÊNCIA WILKINSON
# ==============================================================================
f0_ghz = 3.0  # Frequência central de projeto (3.0 GHz)
z0 = 50.0  # Impedância de referência dos acessos (50 Ohms)
er = 4.4  # Permissividade do substrato (FR-4)
h_mm = 1.6  # Espessura do dielétrico (mm)

c0 = 299792458.0
f0 = f0_ghz * 1e9
h = h_mm * 1e-3

# Parâmetros teóricos do Wilkinson clássico igualitário (3 dB):
# Braços de quarto de onda com Z_braco = sqrt(2) * Z0 ~ 70.71 Ohms
z_braco = np.sqrt(2.0) * z0
# Resistor de isolação entre saídas R = 2 * Z0 = 100 Ohms
r_isol = 2.0 * z0


# ==============================================================================
# 2. DIMENSIONAMENTO EM MICROFITA (HAMMERSTAD & JENSEN)
# ==============================================================================
def calc_largura_microfita(z_alvo, er_val, h_val):
  """Sintetiza a largura W para atingir a impedância Z_alvo."""
  eta0 = 119.9169832 * np.pi
  w_arr = np.linspace(0.2e-3, 12e-3, 100000)
  u = w_arr / h_val
  eps_e = (er_val + 1.0) / 2.0 + ((er_val - 1.0) / 2.0) * (
      1.0 + 12.0 / u
  ) ** (-0.5)
  z0_calc = np.where(
      u <= 1.0,
      (60.0 / np.sqrt(eps_e)) * np.log(8.0 / u + 0.25 * u),
      eta0 / (np.sqrt(eps_e) * (u + 1.393 + 0.667 * np.log(u + 1.444))),
  )
  idx = np.argmin(np.abs(z0_calc - z_alvo))
  return w_arr[idx], eps_e[idx]


# Largura das linhas de 50 Ohms (portas 1, 2 e 3)
w_50, eps_50 = calc_largura_microfita(z0, er, h)
# Largura dos dois braços de 70.71 Ohms
w_70, eps_70 = calc_largura_microfita(z_braco, er, h)

# Comprimento de quarto de onda (lambda_g / 4) dos braços em 3 GHz
lambda_g_braco = c0 / (f0 * np.sqrt(eps_70))
L_braco_mm = (lambda_g_braco / 4.0) * 1e3

print("=" * 65)
print("PROJETO DO DIVISOR WILKINSON EM MICROFITA (3.0 GHz):")
print(
    f" - Linhas de 50 Ohms:           W = {w_50*1e3:.2f} mm | eps_eff ="
    f" {eps_50:.2f}"
)
print(
    f" - Braços de 70.71 Ohms:        W = {w_70*1e3:.2f} mm | eps_eff ="
    f" {eps_70:.2f}"
)
print(
    f" - Comprimento dos Braços (L):  {L_braco_mm:.2f} mm (lambda_g / 4 @"
    f" 3 GHz)"
)
print(f" - Resistor de Isolação (R):    {r_isol:.1f} Ohms (entre portas 2 e 3)")
print("=" * 65)

# ==============================================================================
# 3. CÁLCULO ANALÍTICO DOS PARÂMETROS S (2.0 GHz a 4.0 GHz)
# ==============================================================================
freqs_ghz = np.linspace(2.0, 4.0, 1601)  # Varredura ajustada: 2 a 4 GHz
freqs_hz = freqs_ghz * 1e9

# Comprimento elétrico dos braços theta = beta * L = (pi / 2) * (f / f0)
theta = (np.pi / 2.0) * (freqs_hz / f0)

# 3.1 Reflexão na Porta 1 (S11)
zin_ramo = (
    z_braco
    * (z0 + 1j * z_braco * np.tan(theta))
    / (z_braco + 1j * z0 * np.tan(theta))
)
zin_total_porta1 = zin_ramo / 2.0
s11 = (zin_total_porta1 - z0) / (zin_total_porta1 + z0)

# 3.2 Transmissão da Porta 1 para as Portas 2 e 3 (S21 = S31)
den_trans = 3.0 * np.cos(theta) + 1j * 2.0 * np.sqrt(2.0) * np.sin(theta)
s21 = 2.0 / den_trans
s31 = s21

# 3.3 Casamento das Saídas (S22) e Isolação entre Saídas (S23)
zin2_even = (
    z_braco
    * (2.0 * z0 + 1j * z_braco * np.tan(theta))
    / (z_braco + 1j * 2.0 * z0 * np.tan(theta))
)
gamma2_even = (zin2_even - z0) / (zin2_even + z0)

z_stub_odd = 1j * z_braco * np.tan(theta)
zin2_odd = (z0 * z_stub_odd) / (z0 + z_stub_odd)
gamma2_odd = (zin2_odd - z0) / (zin2_odd + z0)

s22 = 0.5 * (gamma2_even + gamma2_odd)
s23 = 0.5 * (gamma2_even - gamma2_odd)

# Conversão para decibéis (dB)
s11_db = 20.0 * np.log10(np.clip(np.abs(s11), 1e-6, 1.0))
s21_db = 20.0 * np.log10(np.clip(np.abs(s21), 1e-6, 1.0))
s31_db = s21_db
s22_db = 20.0 * np.log10(np.clip(np.abs(s22), 1e-6, 1.0))
s23_db = 20.0 * np.log10(np.clip(np.abs(s23), 1e-6, 1.0))

# Fases em graus
fase_s21_deg = np.rad2deg(np.angle(s21))
fase_s31_deg = np.rad2deg(np.angle(s31))
delta_fase = fase_s21_deg - fase_s31_deg

df_wilkinson = pd.DataFrame({
    "Freq_GHz": freqs_ghz,
    "S11_dB": s11_db,
    "S21_dB": s21_db,
    "S31_dB": s31_db,
    "S22_dB": s22_db,
    "S23_dB": s23_db,
    "Fase_S21_deg": fase_s21_deg,
    "Fase_S31_deg": fase_s31_deg,
    "Delta_Fase_deg": delta_fase,
})

# ==============================================================================
# 4. SALVAMENTO DOS DADOS (Pasta 'Dados')
# ==============================================================================
diretorio_script = os.path.dirname(os.path.abspath(__file__))
pasta_dados = os.path.join(diretorio_script, "Dados")
pasta_imagens = os.path.join(diretorio_script, "Imagens")

os.makedirs(pasta_dados, exist_ok=True)
os.makedirs(pasta_imagens, exist_ok=True)

caminho_dados = os.path.join(pasta_dados, "wilkinson_3ghz_curvas.txt")
df_wilkinson.to_csv(caminho_dados, sep="\t", index=False, float_format="%.4f")
print(f"[OK] Curvas com 1601 pontos salvas em: {caminho_dados}")

# ==============================================================================
# 5. GERAÇÃO DOS GRÁFICOS (Pasta 'Imagens')
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# Painel 1: Magnitudes dos Parâmetros S (dB)
ax1.plot(
    df_wilkinson["Freq_GHz"],
    df_wilkinson["S21_dB"],
    color="forestgreen",
    linewidth=2,
    label=r"$|S_{21}| = |S_{31}|$ (Divisão $-3\text{ dB}$)",
)
ax1.plot(
    df_wilkinson["Freq_GHz"],
    df_wilkinson["S11_dB"],
    color="crimson",
    linewidth=1.8,
    label=r"$|S_{11}|$ (Reflexão Entrada)",
)
ax1.plot(
    df_wilkinson["Freq_GHz"],
    df_wilkinson["S23_dB"],
    color="purple",
    linewidth=1.8,
    linestyle="--",
    label=r"$|S_{23}|$ (Isolação Saídas)",
)
ax1.plot(
    df_wilkinson["Freq_GHz"],
    df_wilkinson["S22_dB"],
    color="darkorange",
    linewidth=1.5,
    linestyle=":",
    label=r"$|S_{22}|$ (Reflexão Saídas)",
)

ax1.axvline(
    f0_ghz, color="gray", linestyle=":", alpha=0.7, label=r"$f_0 = 3{,}0$ GHz"
)
ax1.axhline(-3.01, color="forestgreen", linestyle=":", alpha=0.5)
ax1.set_title(
    r"Parâmetros $S$ do Divisor Wilkinson ($2{,}0$ a $4{,}0\text{ GHz}$)",
    fontweight="bold",
    fontsize=11,
)
ax1.set_xlabel("Frequência (GHz)", fontsize=10)
ax1.set_ylabel("Magnitude (dB)", fontsize=10)
ax1.set_xlim(2.0, 4.0)
ax1.set_ylim(-60, 2)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower left", fontsize=8.5)

# Painel 2: Fases de Transmissão e Balanço de Fase
ax2.plot(
    df_wilkinson["Freq_GHz"],
    df_wilkinson["Fase_S21_deg"],
    color="navy",
    linewidth=2,
    label=r"$\angle S_{21} = \angle S_{31}$",
)
ax2.plot(
    df_wilkinson["Freq_GHz"],
    df_wilkinson["Delta_Fase_deg"],
    color="firebrick",
    linewidth=2,
    linestyle="--",
    label=r"$\Delta\phi = \angle S_{21} - \angle S_{31} = 0^\circ$",
)

ax2.set_title(
    r"Resposta de Fase e Equilíbrio ($\Delta\phi = 0^\circ$)",
    fontweight="bold",
    fontsize=11,
)
ax2.set_xlabel("Frequência (GHz)", fontsize=10)
ax2.set_ylabel(r"Fase (graus)", fontsize=10)
ax2.set_xlim(2.0, 4.0)
ax2.set_ylim(-180, 180)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper right", fontsize=9)

plt.tight_layout()

caminho_fig = os.path.join(pasta_imagens, "wilkinson_3ghz_analitico.png")
plt.savefig(caminho_fig, dpi=300)
print(f"[OK] Gráfico salvo com sucesso em: {caminho_fig}")
plt.show()