import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# 1. PARÂMETROS DA CAVIDADE RESSONANTE RETANGULAR
# ==============================================================================
# Dimensões da cavidade em milímetros (mm):
# a = largura (eixo x), b = altura (eixo y), d = comprimento (eixo z)
# (Ajuste os valores abaixo se a sua cavidade tiver dimensões específicas)
a_mm = 50.0  # mm
b_mm = 25.0  # mm
d_mm = 70.0  # mm

# Propriedades do meio dielétrico interno (vácuo/ar):
er = 1.0  # Permissividade relativa
ur = 1.0  # Permeabilidade relativa

# Conversão para unidades do SI (metros) e velocidade da luz
a = a_mm * 1e-3
b = b_mm * 1e-3
d = d_mm * 1e-3
c0 = 299792458.0  # m/s
v = c0 / np.sqrt(er * ur)

# ==============================================================================
# 2. CÁLCULO ANALÍTICO DAS FREQUÊNCIAS DE RESSONÂNCIA (TE_mnp e TM_mnp)
# ==============================================================================
# Equação geral: f_mnp = (v / 2) * sqrt( (m/a)^2 + (n/b)^2 + (p/d)^2 )
# Condições de existência dos modos em cavidade retangular:
#   - Modo TE_mnp: m, n >= 0 (no máximo um zero: m+n >= 1) e p >= 1
#   - Modo TM_mnp: m >= 1, n >= 1 e p >= 0

modos_lista = []
max_index = 6  # Varredura de índices m, n, p

# 2.1 Varredura de Modos TE
for m in range(max_index):
  for n in range(max_index):
    if m == 0 and n == 0:
      continue  # Não existe TE_00p
    for p in range(1, max_index):  # p >= 1
      f_hz = (v / 2.0) * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2)
      f_ghz = f_hz / 1e9
      modos_lista.append({
          "Tipo": "TE",
          "m": m,
          "n": n,
          "p": p,
          "Modo": f"TE_{m}{n}{p}",
          "Freq_GHz": f_ghz,
      })

# 2.2 Varredura de Modos TM
for m in range(1, max_index):  # m >= 1
  for n in range(1, max_index):  # n >= 1
    for p in range(max_index):  # p >= 0
      f_hz = (v / 2.0) * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2)
      f_ghz = f_hz / 1e9
      modos_lista.append({
          "Tipo": "TM",
          "m": m,
          "n": n,
          "p": p,
          "Modo": f"TM_{m}{n}{p}",
          "Freq_GHz": f_ghz,
      })

# Criar DataFrame, ordenar por frequência ascendente e pegar os 30 primeiros modos
df_modos = pd.DataFrame(modos_lista)
df_30 = (
    df_modos.sort_values(by="Freq_GHz")
    .head(30)
    .reset_index(drop=True)
)
df_30.insert(0, "Ordem", range(1, 31))

# ==============================================================================
# 3. SALVAMENTO DOS DADOS (Pasta 'Dados')
# ==============================================================================
diretorio_script = os.path.dirname(os.path.abspath(__file__))
pasta_dados = os.path.join(diretorio_script, "Dados")
pasta_imagens = os.path.join(diretorio_script, "Imagens")

os.makedirs(pasta_dados, exist_ok=True)
os.makedirs(pasta_imagens, exist_ok=True)

caminho_txt = os.path.join(pasta_dados, "cavidade_modos_30.txt")
df_30.to_csv(caminho_txt, sep="\t", index=False, float_format="%.4f")
print(f"[OK] 30 modos salvos com sucesso em: {caminho_txt}")

# ==============================================================================
# 4. GERAÇÃO DO GRÁFICO DO ESPECTRO (Pasta 'Imagens')
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)

# Separar cores para modos TE e TM
df_te = df_30[df_30["Tipo"] == "TE"]
df_tm = df_30[df_30["Tipo"] == "TM"]

# Linhas verticais do espectro
ax.vlines(
    df_te["Freq_GHz"],
    ymin=0,
    ymax=1.0,
    colors="royalblue",
    linewidth=1.8,
    label="Modos TE",
    alpha=0.85,
)
ax.vlines(
    df_tm["Freq_GHz"],
    ymin=0,
    ymax=1.0,
    colors="crimson",
    linewidth=1.8,
    linestyles="dashed",
    label="Modos TM",
    alpha=0.85,
)

# Anotação com destaque nos 5 primeiros modos (incluindo o modo fundamental)
for i in range(min(5, len(df_30))):
  row = df_30.iloc[i]
  cor_pt = "royalblue" if row["Tipo"] == "TE" else "crimson"
  ax.plot(row["Freq_GHz"], 1.0, marker="o", color=cor_pt, markersize=6)
  ax.annotate(
      f"{row['Modo']}\n{row['Freq_GHz']:.3f} GHz",
      xy=(row["Freq_GHz"], 1.0),
      xytext=(0, 12),
      textcoords="offset points",
      ha="center",
      fontsize=8,
      fontweight="bold",
      color=cor_pt,
  )

# Identificação do modo fundamental
modo_fund = df_30.iloc[0]
ax.text(
    0.02,
    0.88,
    (
        f"Modo Fundamental: {modo_fund['Modo']} @ {modo_fund['Freq_GHz']:.4f}"
        f" GHz\nDimensões: a = {a_mm} mm, b = {b_mm} mm, d = {d_mm} mm"
    ),
    transform=ax.transAxes,
    fontsize=9,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="lightyellow",
        edgecolor="goldenrod",
        alpha=0.9,
    ),
)

# Configurações dos eixos (usando raw strings r"" para evitar conflito de escape)
ax.set_title(
    r"Espectro de Ressonância da Cavidade Retangular ($N = 30$ Modos)",
    fontsize=12,
    fontweight="bold",
)
ax.set_xlabel("Frequência de Ressonância (GHz)", fontsize=10)
ax.set_yticks([])  # Remove escala de amplitude arbitrária
ax.set_ylim(0, 1.35)
ax.grid(axis="x", linestyle="--", alpha=0.5)
ax.legend(loc="upper right", framealpha=0.95)

plt.tight_layout()

# Salvar e exibir o gráfico
caminho_fig = os.path.join(pasta_imagens, "cavidade_30_modos_analitico.png")
plt.savefig(caminho_fig, dpi=300)
print(f"[OK] Gráfico salvo com sucesso em: {caminho_fig}")
plt.show()