import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# 1. ESPECIFICAÇÕES DO PROJETO DA ANTENA PATCH RETANGULAR
# ==============================================================================
f0_ghz = 3.0  # Frequência de ressonância (3 GHz)
er = 4.4  # Permissividade do substrato (FR-4)
h_mm = 1.6  # Altura do dielétrico (mm)
z0_linha = 50.0  # Impedância da linha de alimentação

c0 = 299792458.0
f0 = f0_ghz * 1e9
h = h_mm * 1e-3

# ==============================================================================
# 2. DIMENSIONAMENTO ANALÍTICO (MODELO BALANIS / TRANSMISSION LINE MODEL)
# ==============================================================================
# 2.1 Largura do Patch (W)
W = (c0 / (2.0 * f0)) * np.sqrt(2.0 / (er + 1.0))
W_mm = W * 1e3

# 2.2 Permissividade Efetiva (eps_eff)
eps_eff = (er + 1.0) / 2.0 + ((er - 1.0) / 2.0) * (1.0 + 12.0 * (h / W)) ** (
    -0.5
)

# 2.3 Extensão de borda devido ao campo de espraiamento (fringing: Delta L)
delta_L = (
    0.412
    * h
    * ((eps_eff + 0.3) * (W / h + 0.264))
    / ((eps_eff - 0.258) * (W / h + 0.8))
)

# 2.4 Comprimento Efetivo e Comprimento Físico Real (L)
L_eff = c0 / (2.0 * f0 * np.sqrt(eps_eff))
L = L_eff - 2.0 * delta_L
L_mm = L * 1e3

# 2.5 Condutância dos slots radiantes (G1) e Impedância de Entrada nas Bordas (Rin_0)
k0 = 2.0 * np.pi * f0 / c0
X = k0 * W
I1 = -2.0 + np.cos(X) + X * np.sin(X) + np.pi * X  # aproximação da integral
G1 = (1.0 / (120.0 * np.pi**2)) * (X * np.sin(X) + np.cos(X) - 1.0)
if G1 <= 0:
  G1 = (W / (120.0 * np.pi * (c0 / f0))) * (1.0 - (k0 * h) ** 2 / 24.0)

Rin_0 = 1.0 / (2.0 * G1)  # Resistência de entrada na borda

# 2.6 Posição do Inset Feed (y0) para casar em 50 ohms: Rin(y0) = Rin_0 * cos^2(pi*y0/L)
termo_cos = np.clip(np.sqrt(z0_linha / Rin_0), 0.0, 1.0)
y0 = (L / np.pi) * np.arccos(termo_cos)
y0_mm = y0 * 1e3

print("=" * 65)
print("DIMENSÕES ANALÍTICAS DO PATCH DE MICROFITA (3.0 GHz):")
print(f" - Largura do Patch (W):        {W_mm:.2f} mm")
print(f" - Comprimento do Patch (L):    {L_mm:.2f} mm")
print(f" - Extensão de Fringing (dL):   {delta_L * 1e3:.3f} mm")
print(f" - Permissividade Efetiva:      {eps_eff:.3f}")
print(f" - Impedância na borda:         {Rin_0:.1f} Ohms")
print(f" - Posição do Inset Feed (y0):  {y0_mm:.2f} mm (casamento 50 Ohms)")
print("=" * 65)

# ==============================================================================
# 3. CURVA DE RESPOSTA EM FREQUÊNCIA S11 E IMPEDÂNCIA ZIN (2.7 GHz a 3.3 GHz)
# ==============================================================================
freqs_ghz = np.linspace(2.7, 3.3, 1601)
freqs_hz = freqs_ghz * 1e9

# Modelagem por circuito ressonante RLC paralelo equivalente
Q_aprox = 35.0  # Fator de qualidade típico para FR-4 com perdas
R_res = z0_linha  # No casamento perfeito Rin = 50 ohms

delta_f = (freqs_hz - f0) / f0
Z_in = R_res / (1.0 + 2j * Q_aprox * delta_f)

# Coeficiente de reflexão Gama e S11 em dB
gamma = (Z_in - z0_linha) / (Z_in + z0_linha)
s11_db = 20.0 * np.log10(np.clip(np.abs(gamma), 1e-6, 1.0))

df_patch = pd.DataFrame({
    "Freq_GHz": freqs_ghz,
    "S11_dB": s11_db,
    "Zin_Real": np.real(Z_in),
    "Zin_Imag": np.imag(Z_in),
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

caminho_dados = os.path.join(pasta_dados, "patch_3ghz_curvas.txt")
df_patch.to_csv(caminho_dados, sep="\t", index=False, float_format="%.4f")
print(f"[OK] Curvas com 1601 pontos salvas em: {caminho_dados}")

# ==============================================================================
# 5. GERAÇÃO DOS GRÁFICOS (Pasta 'Imagens' - IDÊNTICO AO LAYOUT DO HFSS)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# Painel 1: Perda de Retorno S11 (dB)
ax1.plot(
    df_patch["Freq_GHz"],
    df_patch["S11_dB"],
    color="crimson",
    linewidth=2,
    label=r"$|S_{11}|$ Analítico",
)
ax1.axhline(
    -10.0,
    color="black",
    linestyle="--",
    alpha=0.7,
    label=r"Nível de Casamento ($-10\text{ dB}$)",
)
ax1.axvline(
    f0_ghz, color="gray", linestyle=":", alpha=0.7, label=r"$f_0 = 3{,}0$ GHz"
)

# Identificar largura de banda abaixo de -10 dB
mask_bw = df_patch["S11_dB"] <= -10.0
if np.any(mask_bw):
  f_low = df_patch.loc[mask_bw, "Freq_GHz"].iloc[0]
  f_high = df_patch.loc[mask_bw, "Freq_GHz"].iloc[-1]
  bw_mhz = (f_high - f_low) * 1e3
  ax1.fill_between(
      df_patch["Freq_GHz"],
      df_patch["S11_dB"],
      -10.0,
      where=mask_bw,
      color="pink",
      alpha=0.5,
      label=f"BW (-10 dB) = {bw_mhz:.1f} MHz",
  )

ax1.set_title(
    r"Parâmetro $S_{11}$ da Antena Patch Retangular (Analítico)",
    fontweight="bold",
    fontsize=11,
)
ax1.set_xlabel("Frequência (GHz)", fontsize=10)
ax1.set_ylabel(r"$|S_{11}|$ (dB)", fontsize=10)
ax1.set_xlim(2.7, 3.3)
ax1.set_ylim(-40, 2)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower right", fontsize=9)

# Painel 2: Impedância de Entrada Zin vs. Frequência (Igual ao do HFSS)
ax2.plot(
    df_patch["Freq_GHz"],
    df_patch["Zin_Real"],
    color="navy",
    linewidth=2,
    label=r"$\text{Re}(Z_{\text{in}})$ (Resistência)",
)
ax2.plot(
    df_patch["Freq_GHz"],
    df_patch["Zin_Imag"],
    color="darkorange",
    linewidth=1.8,
    linestyle="--",
    label=r"$\text{Im}(Z_{\text{in}})$ (Reatância)",
)

ax2.axhline(
    50.0,
    color="navy",
    linestyle=":",
    alpha=0.5,
    label=r"$R_0 = 50\,\Omega$",
)
ax2.axhline(
    0.0,
    color="darkorange",
    linestyle=":",
    alpha=0.5,
    label=r"$X_0 = 0\,\Omega$",
)
ax2.axvline(
    f0_ghz, color="gray", linestyle=":", alpha=0.7, label=r"$f_0 = 3{,}0$ GHz"
)

ax2.set_title(
    r"Impedância de Entrada $Z_{\text{in}}$ vs. Frequência (Analítico)",
    fontweight="bold",
    fontsize=11,
)
ax2.set_xlabel("Frequência (GHz)", fontsize=10)
ax2.set_ylabel(r"Impedância ($\Omega$)", fontsize=10)
ax2.set_xlim(2.7, 3.3)
ax2.set_ylim(-50, 100)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper right", fontsize=8.5)

plt.tight_layout()
caminho_fig = os.path.join(pasta_imagens, "patch_3ghz_analitico.png")
plt.savefig(caminho_fig, dpi=300)
print(f"[OK] Gráfico salvo com sucesso em: {caminho_fig}")
plt.close(fig)
