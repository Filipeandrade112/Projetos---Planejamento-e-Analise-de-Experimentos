import os
import matplotlib.pyplot as plt
import numpy as np

# Resolução automática da pasta raiz do projeto
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(diretorio_atual).lower() in [
    "analiticos",
    "hfss",
    "tidy3d",
]:
  pasta_raiz = os.path.dirname(diretorio_atual)
else:
  pasta_raiz = diretorio_atual

pasta_dados = os.path.join(pasta_raiz, "Dados")
pasta_imagens = os.path.join(pasta_raiz, "Imagens")
os.makedirs(pasta_dados, exist_ok=True)
os.makedirs(pasta_imagens, exist_ok=True)

# Parâmetros redimensionados para f0 = 3.0 GHz
f0_ghz = 3.0
f0 = f0_ghz * 1e9  # 3.0 GHz
Z0 = 50.0  # 50 Ohms
Z0_p = 50.0  # Ramos shunt
Z0_s = Z0 / np.sqrt(2.0)  # Ramos série (~35.355 Ohms)

# Faixa de análise: 2.0 GHz a 4.0 GHz com 1601 pontos
num_points = 1601
freqs = np.linspace(2.0e9, 4.0e9, num_points)
freqs_ghz = freqs / 1e9

s11_db = np.zeros(num_points)
s21_db = np.zeros(num_points)
s31_db = np.zeros(num_points)
s41_db = np.zeros(num_points)
p21_deg = np.zeros(num_points)
p31_deg = np.zeros(num_points)
dp_deg = np.zeros(num_points)

Y0_p = 1.0 / Z0_p
Y0_s = 1.0 / Z0_s

for idx, f in enumerate(freqs):
  theta = (np.pi / 2.0) * (f / f0)

  # Matriz série
  M_ser = np.array(
      [
          [np.cos(theta), 1j * Z0_s * np.sin(theta)],
          [1j * Y0_s * np.sin(theta), np.cos(theta)],
      ],
      dtype=complex,
  )

  # Matriz shunt modo par e ímpar
  tan_half = np.tan(theta / 2.0)
  M_shunt_e = np.array([[1.0, 0.0], [1j * Y0_p * tan_half, 1.0]], dtype=complex)
  M_shunt_o = np.array(
      [[1.0, 0.0], [-1j * Y0_p * tan_half, 1.0]], dtype=complex
  )

  M_e = M_shunt_e @ M_ser @ M_shunt_e
  M_o = M_shunt_o @ M_ser @ M_shunt_o

  A_e, B_e, C_e, D_e = M_e[0, 0], M_e[0, 1], M_e[1, 0], M_e[1, 1]
  A_o, B_o, C_o, D_o = M_o[0, 0], M_o[0, 1], M_o[1, 0], M_o[1, 1]

  den_e = A_e + (B_e / Z0) + (C_e * Z0) + D_e
  den_o = A_o + (B_o / Z0) + (C_o * Z0) + D_o

  gamma_e = (A_e + (B_e / Z0) - (C_e * Z0) - D_e) / den_e
  gamma_o = (A_o + (B_o / Z0) - (C_o * Z0) - D_o) / den_o

  trans_e = 2.0 / den_e
  trans_o = 2.0 / den_o

  s11 = 0.5 * (gamma_e + gamma_o)
  s21 = 0.5 * (trans_e + trans_o)
  s31 = 0.5 * (trans_e - trans_o)
  s41 = 0.5 * (gamma_e - gamma_o)

  s11_db[idx] = 20.0 * np.log10(np.clip(np.abs(s11), 1e-12, None))
  s21_db[idx] = 20.0 * np.log10(np.clip(np.abs(s21), 1e-12, None))
  s31_db[idx] = 20.0 * np.log10(np.clip(np.abs(s31), 1e-12, None))
  s41_db[idx] = 20.0 * np.log10(np.clip(np.abs(s41), 1e-12, None))

  p21 = np.angle(s21, deg=True)
  p31 = np.angle(s31, deg=True)
  p21_deg[idx] = p21
  p31_deg[idx] = p31
  dp_deg[idx] = (p21 - p31) % 360.0

# 1. Salva arquivo TXT em Dados
out_txt = os.path.join(pasta_dados, "branchline_3ghz_curvas.txt")
header = (
    "# Acoplador Branch-Line Analitico (f0 = 3.0 GHz, Z0 = 50 Ohms)\n#"
    " Freq(GHz)\t|S11|(dB)\t|S21|(dB)\t|S31|(dB)\t|S41|(dB)\tFase_S21(deg)\tFase_S31(deg)\tDelta_Fase(deg)"
)
data_mat = np.column_stack(
    [freqs_ghz, s11_db, s21_db, s31_db, s41_db, p21_deg, p31_deg, dp_deg]
)
np.savetxt(
    out_txt, data_mat, fmt="%.6e", delimiter="\t", header=header, comments=""
)
print(f"[OK] Curvas do Branchline (Analítico) salvas em: {out_txt}")

# 2. Gera e salva gráfico em Imagens (mesmo padrão visual de 2 painéis do HFSS e Tidy3D)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# Painel 1: Magnitudes dos Parâmetros S
ax1.plot(
    freqs_ghz,
    s21_db,
    color="forestgreen",
    linewidth=2,
    label=r"$|S_{21}|$ (Transmissão / Through)",
)
ax1.plot(
    freqs_ghz,
    s31_db,
    color="darkorange",
    linewidth=2,
    label=r"$|S_{31}|$ (Acoplada / Coupled)",
)
ax1.plot(
    freqs_ghz,
    s11_db,
    color="crimson",
    linewidth=1.8,
    linestyle="--",
    label=r"$|S_{11}|$ (Reflexão / Retorno)",
)
ax1.plot(
    freqs_ghz,
    s41_db,
    color="purple",
    linewidth=1.8,
    linestyle=":",
    label=r"$|S_{41}|$ (Isolação)",
)

ax1.axvline(
    f0_ghz, color="gray", linestyle=":", alpha=0.7, label=r"$f_0 = 3{,}0$ GHz"
)
ax1.axhline(-3.01, color="forestgreen", linestyle=":", alpha=0.5)

ax1.set_title(
    r"Parâmetros $S$ do Acoplador Branchline 3 GHz (Analítico)",
    fontweight="bold",
    fontsize=11,
)
ax1.set_xlabel("Frequência (GHz)", fontsize=10)
ax1.set_ylabel("Magnitude (dB)", fontsize=10)
ax1.set_xlim(2.0, 4.0)
ax1.set_ylim(-50, 2)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower left", fontsize=8.5)

# Painel 2: Diferença de Fase em Quadratura (Delta_phi = 90 graus)
ax2.plot(
    freqs_ghz,
    dp_deg,
    color="navy",
    linewidth=2,
    label=r"$\Delta\phi = \angle S_{21} - \angle S_{31}$",
)
ax2.axhline(
    90.0,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=r"Quadratura Ideal ($90^\circ$)",
)
ax2.axvline(
    f0_ghz, color="gray", linestyle=":", alpha=0.7, label=r"$f_0 = 3{,}0$ GHz"
)

ax2.set_title(
    r"Diferença de Fase entre as Saídas ($\Delta\phi$ Analítico)",
    fontweight="bold",
    fontsize=11,
)
ax2.set_xlabel("Frequência (GHz)", fontsize=10)
ax2.set_ylabel(r"Diferença de Fase (graus)", fontsize=10)
ax2.set_xlim(2.0, 4.0)
ax2.set_ylim(80, 150)
ax2.set_yticks(np.arange(80, 151, 10))
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper left", fontsize=9)

plt.tight_layout()
out_img = os.path.join(pasta_imagens, "branchline_3ghz_analitico.png")
fig.savefig(out_img, dpi=300)
print(f"[OK] Gráfico do Branchline (Analítico) salvo em: {out_img}")
plt.show()
