import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# 1. PARÂMETROS DO SUBSTRATO E DA LINHA DE MICROFITA
# ==============================================================================
# Substrato (FR-4 padrão)
er = 4.4  # Permissividade relativa
h_mm = 1.6  # Altura do dielétrico (mm)
t_mm = 0.035  # Espessura do cobre (1 oz = 35 um)
tan_d = 0.02  # Tangente de perdas do dielétrico
sigma_cu = 5.8e7  # Condutividade do cobre (S/m)

# Comprimento físico da linha de transmissão
L_mm = 50.0  # Comprimento da linha (50 mm)
z0_ref = 50.0  # Impedância de referência das portas (50 Ohms)

# Conversões para unidades do SI
c0 = 299792458.0  # m/s
mu0 = 4.0 * np.pi * 1e-7  # H/m
h = h_mm * 1e-3
t = t_mm * 1e-3
L = L_mm * 1e-3

# ==============================================================================
# 2. SÍNTESE DA LARGURA (W) PARA LINHA CASADA EM 50 OHMS (HAMMERSTAD & JENSEN)
# ==============================================================================
eta0 = 119.9169832 * np.pi  # ~376.73 Ohms


def calc_z0_quasi_static(W_val, er_val, h_val, t_val=0.0):
  u = W_val / h_val
  if t_val > 0.0:
    du = ((t_val / h_val) / np.pi) * np.log(
        1.0
        + (4.0 * np.e)
        / ((t_val / h_val) * (1.0 / np.tanh(np.sqrt(6.517 * u))) ** 2)
    )
    u_eff = u + du
    eps_eff = (
        (er_val + 1.0) / 2.0
        + ((er_val - 1.0) / 2.0) * (1.0 + 12.0 / u_eff) ** (-0.5)
        - ((er_val - 1.0) * t_val / (4.6 * np.sqrt(W_val * h_val)))
    )
  else:
    u_eff = u
    eps_eff = (er_val + 1.0) / 2.0 + ((er_val - 1.0) / 2.0) * (
        1.0 + 12.0 / u_eff
    ) ** (-0.5)

  if u_eff <= 1.0:
    z0 = (60.0 / np.sqrt(eps_eff)) * np.log(8.0 / u_eff + 0.25 * u_eff)
  else:
    z0 = eta0 / (
        np.sqrt(eps_eff) * (u_eff + 1.393 + 0.667 * np.log(u_eff + 1.444))
    )
  return z0, eps_eff


# Busca da largura W nominal (quasi-estática t=0) e avaliação real com t = 35 um e W = 3.08 mm
w_teste = np.linspace(0.5e-3, 10.0e-3, 100000)
z0_busca = np.array([calc_z0_quasi_static(w, er, h, 0.0)[0] for w in w_teste])
W_ideal = w_teste[np.argmin(np.abs(z0_busca - z0_ref))]
W_mm = round(W_ideal * 1e3, 2)  # 3.08 mm (dimensão física de fabricação/HFSS)
W = W_mm * 1e-3
Zc0, eps_eff0 = calc_z0_quasi_static(W, er, h, t)

print("=" * 65)
print(f"LINHA DE MICROFITA (L = {L_mm:.1f} mm):")
print(f" - Largura física (W):          {W_mm:.2f} mm")
print(f" - Impedância real (c/ t=35um): {Zc0:.2f} Ohms")
print(f" - Permissividade efetiva:      {eps_eff0:.3f}")
print("=" * 65)

# ==============================================================================
# 3. CÁLCULO ANALÍTICO DOS PARÂMETROS S11 E S12 (1.0 GHz a 5.0 GHz)
# ==============================================================================
freqs_ghz = np.linspace(1.0, 5.0, 1601)
freqs_hz = freqs_ghz * 1e9

s11_list = []
s12_list = []

for f in freqs_hz:
  omega = 2.0 * np.pi * f
  k0 = omega / c0

  # Perda no dielétrico (Np/m)
  alpha_d = (
      (k0 * er * (eps_eff0 - 1.0) * tan_d)
      / (2.0 * np.sqrt(eps_eff0) * (er - 1.0))
  )

  # Resistência de folha e perda no condutor por efeito pelicular (Np/m)
  Rs = np.sqrt(omega * mu0 / (2.0 * sigma_cu))
  alpha_c = Rs / (Zc0 * W)

  # Constante de propagação complexa gama = alfa + j*beta
  alpha_total = alpha_d + alpha_c
  beta = k0 * np.sqrt(eps_eff0)
  gamma = alpha_total + 1j * beta

  # Impedância característica complexa da linha com perdas: Zc(f) = Zc0 * (1 - j*alpha/beta)
  Zc = Zc0 * (1.0 - 1j * (alpha_total / beta))

  # Matriz ABCD da linha de transmissão de comprimento L
  A = np.cosh(gamma * L)
  B = Zc * np.sinh(gamma * L)
  C = (1.0 / Zc) * np.sinh(gamma * L)
  D = A  # Linha simétrica e recíproca

  # Conversão da matriz ABCD para Parâmetros S com referência z0_ref (50 Ohms)
  den = A + (B / z0_ref) + (C * z0_ref) + D
  s11 = (A + (B / z0_ref) - (C * z0_ref) - D) / den
  s12 = 2.0 / den  # Por reciprocidade, S12 = S21

  s11_list.append(s11)
  s12_list.append(s12)

s11_arr = np.array(s11_list)
s12_arr = np.array(s12_list)

# Magnitudes em dB e Fases em graus
s11_db = 20.0 * np.log10(np.clip(np.abs(s11_arr), 1e-6, 1.0))
s12_db = 20.0 * np.log10(np.clip(np.abs(s12_arr), 1e-6, 1.0))

fase_s11_deg = np.rad2deg(np.angle(s11_arr))
fase_s12_deg = np.rad2deg(np.unwrap(np.angle(s12_arr)))

df_curvas = pd.DataFrame({
    "Freq_GHz": freqs_ghz,
    "S11_dB": s11_db,
    "S12_dB": s12_db,
    "Fase_S11_deg": fase_s11_deg,
    "Fase_S12_deg": fase_s12_deg,
})

# ==============================================================================
# 4. SALVAMENTO DOS DADOS (Pasta 'Dados')
# ==============================================================================
diretorio_script = os.path.dirname(os.path.abspath(__file__))
pasta_raiz = os.path.dirname(diretorio_script)
pasta_dados = os.path.join(pasta_raiz, "Dados")
pasta_imagens = os.path.join(pasta_raiz, "Imagens")

os.makedirs(pasta_dados, exist_ok=True)
os.makedirs(pasta_imagens, exist_ok=True)

caminho_dados = os.path.join(pasta_dados, "linha_microfita_curvas.txt")
df_curvas.to_csv(caminho_dados, sep="\t", index=False, float_format="%.4f")
print(f"[OK] Curvas com 1601 pontos salvas em: {caminho_dados}")

# ==============================================================================
# 5. GERAÇÃO DOS GRÁFICOS (Pasta 'Imagens')
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# Painel 1: Magnitude dos Parâmetros S (|S11| e |S12| em dB)
ax1.plot(
    df_curvas["Freq_GHz"],
    df_curvas["S12_dB"],
    color="forestgreen",
    linewidth=2,
    label=r"$|S_{12}|$ (Transmissão / Inserção)",
)
ax1.plot(
    df_curvas["Freq_GHz"],
    df_curvas["S11_dB"],
    color="crimson",
    linewidth=1.8,
    linestyle="--",
    label=r"$|S_{11}|$ (Reflexão / Retorno)",
)

ax1.axvline(
    3.0, color="gray", linestyle=":", alpha=0.7, label=r"$f_0 = 3{,}0$ GHz"
)
ax1.set_title(
    r"Parâmetros $S_{11}$ e $S_{12}$ da Linha de Microfita ($L = 50\text{ mm}$)",
    fontweight="bold",
    fontsize=11,
)
ax1.set_xlabel("Frequência (GHz)", fontsize=10)
ax1.set_ylabel("Magnitude (dB)", fontsize=10)
ax1.set_ylim(-60, 2)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="center right")

# Painel 2: Fases dos Parâmetros S (Graus)
ax2.plot(
    df_curvas["Freq_GHz"],
    df_curvas["Fase_S12_deg"],
    color="navy",
    linewidth=2,
    label=r"$\angle S_{12}$ (Fase de Transmissão)",
)
ax2.set_title(
    r"Resposta de Fase $\angle S_{12}$ vs. Frequência",
    fontweight="bold",
    fontsize=11,
)
ax2.set_xlabel("Frequência (GHz)", fontsize=10)
ax2.set_ylabel(r"Fase (graus)", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper right")

plt.tight_layout()

caminho_fig = os.path.join(pasta_imagens, "linha_microfita_analitico.png")
plt.savefig(caminho_fig, dpi=300)
print(f"[OK] Gráfico salvo com sucesso em: {caminho_fig}")
plt.close(fig)
