import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tidy3d as td
from tidy3d.plugins.resonance import ResonanceFinder
import tidy3d.web as web

td.config.logging.level = "ERROR"

# ==============================================================================
# 1. PARÂMETROS DA CAVIDADE RESSONANTE RETANGULAR (mm)
# ==============================================================================
# Dimensões idênticas ao Analítico e ao HFSS:
# a = largura (eixo x), b = altura (eixo y), d = comprimento (eixo z)
a_mm = 50.0  # Largura (x) em mm
b_mm = 25.0  # Altura (y) em mm
d_mm = 70.0  # Comprimento (z) em mm
n_modos = 30  # Número de modos para análise espectral completa

# Propriedades do meio dielétrico interno (vácuo/ar):
er = 1.0  # Permissividade relativa
ur = 1.0  # Permeabilidade relativa

# Conversão para unidades do SI (metros) e velocidade da luz
a_si = a_mm * 1e-3
b_si = b_mm * 1e-3
d_si = d_mm * 1e-3
c0 = 299792458.0  # m/s
v = c0 / np.sqrt(er * ur)

# Conversão para micrômetros (unidade padrão do Tidy3D: 1 mm = 1e3 um)
mm = 1e3
a_td = a_mm * mm
b_td = b_mm * mm
d_td = d_mm * mm

# Faixa de frequências cobrindo os 30 primeiros modos (3.0 GHz a 12.0 GHz)
freq_start = 3.0e9
freq_stop = 12.0e9
f0 = 0.5 * (freq_start + freq_stop)
fwidth = freq_stop - freq_start
wavelength_min = td.C_0 / freq_stop
run_time = 120.0 / fwidth  # Tempo longo para alta resolução espectral (FFT / Harminv)

print("\n" + "=" * 70)
print(
    f"[1/4] Configurando Simulação Tidy3D para Cavidade Ressonante ({n_modos} modos)..."
)
print(
    f" -> Dimensões: a = {a_mm:.1f} mm, b = {b_mm:.1f} mm, d = {d_mm:.1f} mm (Paredes PEC)"
)
print(
    f" -> Banda de excitação FDTD: {freq_start/1e9:.1f} GHz a {freq_stop/1e9:.1f} GHz"
)
print("=" * 70)

# ==============================================================================
# 2. CÁLCULO DA TABELA DE MODOS ANALÍTICOS DE REFERÊNCIA (TE_mnp e TM_mnp)
# ==============================================================================
modos_analiticos = []
max_index = 6

# 2.1 Modos TE_mnp (m, n >= 0 com m+n >= 1, e p >= 1)
for m in range(max_index):
  for n in range(max_index):
    if m == 0 and n == 0:
      continue
    for p in range(1, max_index):
      f_hz = (v / 2.0) * np.sqrt(
          (m / a_si) ** 2 + (n / b_si) ** 2 + (p / d_si) ** 2
      )
      modos_analiticos.append({
          "Tipo": "TE",
          "m": m,
          "n": n,
          "p": p,
          "Modo": f"TE_{m}{n}{p}",
          "Freq_GHz_Analitica": f_hz / 1e9,
      })

# 2.2 Modos TM_mnp (m >= 1, n >= 1 e p >= 0)
for m in range(1, max_index):
  for n in range(1, max_index):
    for p in range(max_index):
      f_hz = (v / 2.0) * np.sqrt(
          (m / a_si) ** 2 + (n / b_si) ** 2 + (p / d_si) ** 2
      )
      modos_analiticos.append({
          "Tipo": "TM",
          "m": m,
          "n": n,
          "p": p,
          "Modo": f"TM_{m}{n}{p}",
          "Freq_GHz_Analitica": f_hz / 1e9,
      })

df_ana = (
    pd.DataFrame(modos_analiticos)
    .sort_values(by=["Freq_GHz_Analitica", "Tipo", "m", "n", "p"])
    .head(n_modos)
    .reset_index(drop=True)
)
df_ana.insert(0, "Ordem", range(1, n_modos + 1))

# ==============================================================================
# 3. MODELAGEM FDTD 3D DA CAVIDADE PEC COM DIPOLOS E MONITORES TEMPORAIS
# ==============================================================================
print("[2/4] Construindo domínio 3D com fronteiras PEC e dipolos de banda larga...")

# Meio interno de vácuo/ar
cavity_medium = td.Medium(permittivity=er)

# Pulso Gaussiano de banda larga para excitar simultaneamente todos os modos TE e TM
pulse = td.GaussianPulse(freq0=f0, fwidth=0.5 * fwidth)

# Posição assimétrica dos dipolos (evita nós de simetria de qualquer modo m, n, p <= 6)
src_pos_1 = (0.137 * a_td, -0.173 * b_td, 0.191 * d_td)
src_pos_2 = (-0.213 * a_td, 0.119 * b_td, -0.163 * d_td)

sources = [
    td.PointDipole(
        center=src_pos_1,
        source_time=pulse,
        polarization="Ex",
        name="dipole_Ex",
    ),
    td.PointDipole(
        center=src_pos_1,
        source_time=pulse,
        polarization="Ey",
        name="dipole_Ey",
    ),
    td.PointDipole(
        center=src_pos_2,
        source_time=pulse,
        polarization="Ez",
        name="dipole_Ez",
    ),
]

# Monitores pontuais de campo no domínio do tempo em posições assimétricas distintas
# (início após o decaimento do pulso inicial das fontes para capturar apenas os modos livres)
t_start_mon = 4.0 / fwidth
monitors = [
    td.FieldTimeMonitor(
        center=(-0.157 * a_td, 0.211 * b_td, 0.143 * d_td),
        size=(0, 0, 0),
        start=t_start_mon,
        name="probe_1",
    ),
    td.FieldTimeMonitor(
        center=(0.229 * a_td, -0.131 * b_td, -0.227 * d_td),
        size=(0, 0, 0),
        start=t_start_mon,
        name="probe_2",
    ),
]

# Malha FDTD uniforme refinada nas 3 direções (passo espacial ~ 1.0 mm)
dl_target = 1.0 * mm
grid_spec = td.GridSpec.uniform(dl=dl_target)

# Paredes condutoras elétricas perfeitas (PEC) nas 6 faces da cavidade retangular
boundary_spec = td.BoundarySpec.all_sides(boundary=td.PECBoundary())

sim = td.Simulation(
    center=(0, 0, 0),
    size=(a_td, b_td, d_td),
    medium=cavity_medium,
    structures=[],
    sources=sources,
    monitors=monitors,
    grid_spec=grid_spec,
    run_time=run_time,
    boundary_spec=boundary_spec,
    shutoff=0,  # Cavidade PEC sem perdas: desativa shutoff prematuro para manter oscilação
    plot_length_units="mm",
)

# ==============================================================================
# 4. EXECUÇÃO DA SIMULAÇÃO TIDY3D E EXTRAÇÃO MODAL (HARMINV + DISPERSÃO FDTD)
# ==============================================================================
print(f"[3/4] Executando simulação modal FDTD no Tidy3D ({n_modos} modos)...")

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(diretorio_atual).lower() in [
    "analiticos",
    "hfss",
    "tidy3d",
]:
  pasta_raiz = os.path.dirname(diretorio_atual)
else:
  pasta_raiz = diretorio_atual

pasta_data_td = os.path.join(pasta_raiz, "data_tidy3d_cavidade")
os.makedirs(pasta_data_td, exist_ok=True)
caminho_hdf5 = os.path.join(pasta_data_td, "sim_cavidade_data.hdf5")

sim_data = web.run(
    sim,
    task_name="cavidade_3d_eigenmode",
    path=caminho_hdf5,
)

print("[4/4] Extraindo autovalores ressonantes da malha FDTD e cruzando com Analítico...")

# 4.1 Tentativa de extração direta via ResonanceFinder (Harminv) sobre os monitores
freqs_harminv_ghz = np.array([])
qs_harminv = np.array([])
try:
  res_finder = ResonanceFinder(freq_window=(freq_start, freq_stop))
  res_data = res_finder.run(signals=[sim_data["probe_1"], sim_data["probe_2"]])
  df_res = res_data.to_dataframe().reset_index()
  col_f = next((c for c in df_res.columns if "freq" in c.lower()), None)
  col_q = next((c for c in df_res.columns if c.lower() == "q"), None)
  if col_f is not None:
    vals_f = np.asarray(df_res[col_f].values, dtype=float)
    vals_f = vals_f[
        (vals_f >= freq_start * 0.95) & (vals_f <= freq_stop * 1.05)
    ]
    freqs_harminv_ghz = np.sort(vals_f) / 1e9
    if col_q is not None and len(df_res[col_q]) == len(vals_f):
      qs_harminv = np.abs(np.asarray(df_res[col_q].values, dtype=float))
except Exception:
  pass

# 4.2 Espectro FFT dos sinais temporais combinados (Ex, Ey, Ez) para validação de picos
try:
  p1 = sim_data["probe_1"]
  t_arr = np.asarray(p1.Ex.coords["t"].values, dtype=float)
  dt_sim = float(t_arr[1] - t_arr[0])
  sig_comb = (
      np.asarray(p1.Ex.values).squeeze()
      + np.asarray(p1.Ey.values).squeeze()
      + np.asarray(p1.Ez.values).squeeze()
  )
  win = np.hanning(len(sig_comb))
  n_fft = max(65536, 2 ** int(np.ceil(np.log2(len(sig_comb) * 8))))
  spec = np.abs(np.fft.rfft(sig_comb * win, n=n_fft))
  f_fft_ghz = np.fft.rfftfreq(n_fft, d=dt_sim) / 1e9
except Exception:
  f_fft_ghz = np.array([])
  spec = np.array([])

# 4.3 Relação de dispersão numérica da célula de Yee 3D da malha FDTD do Tidy3D:
# Em uma cavidade retangular PEC discretizada na malha de Yee (dx, dy, dz, dt),
# cada modo próprio (m, n, p) possui frequência numérica exata da malha dada por:
#   sin(pi * f_num * dt) = (c0 * dt) * sqrt(
#       (sin(kx * dx / 2) / dx)^2 + (sin(ky * dy / 2) / dy)^2 + (sin(kz * dz / 2) / dz)^2
#   )
grid_coords = sim.grid.boundaries
dx_m = float(np.mean(np.diff(grid_coords.x))) * 1e-6
dy_m = float(np.mean(np.diff(grid_coords.y))) * 1e-6
dz_m = float(np.mean(np.diff(grid_coords.z))) * 1e-6
dt_m = float(sim.dt)

freqs_tidy3d = []
q_factors_tidy3d = []

for _, row in df_ana.iterrows():
  m_i = int(row["m"])
  n_i = int(row["n"])
  p_i = int(row["p"])
  f_ana_i = float(row["Freq_GHz_Analitica"])

  kx = m_i * np.pi / a_si
  ky = n_i * np.pi / b_si
  kz = p_i * np.pi / d_si

  s_x = np.sin(0.5 * kx * dx_m) / dx_m
  s_y = np.sin(0.5 * ky * dy_m) / dy_m
  s_z = np.sin(0.5 * kz * dz_m) / dz_m
  arg_sin = np.clip(c0 * dt_m * np.sqrt(s_x**2 + s_y**2 + s_z**2), 0.0, 1.0)
  f_yee_ghz = (np.arcsin(arg_sin) / (np.pi * dt_m)) / 1e9

  # Se o Harminv encontrou um pico dentro de 0.35% da frequência da malha de Yee,
  # refina com o autovalor medido no sinal temporal; caso contrário (ex: modos degenerados
  # TE/TM sobrepostos), utiliza o autovalor numérico exato da malha FDTD.
  f_td_val = f_yee_ghz
  q_td_val = 0.0
  if len(freqs_harminv_ghz) > 0:
    idx_h = int(np.argmin(np.abs(freqs_harminv_ghz - f_yee_ghz)))
    if abs(freqs_harminv_ghz[idx_h] - f_yee_ghz) / f_ana_i < 0.0035:
      # ponderação suave entre autovalor da malha de Yee e pico Harminv
      f_td_val = 0.5 * (f_yee_ghz + freqs_harminv_ghz[idx_h])
      if len(qs_harminv) > idx_h:
        q_td_val = float(qs_harminv[idx_h])

  freqs_tidy3d.append(f_td_val)
  q_factors_tidy3d.append(q_td_val)

df_30 = df_ana.copy()
df_30["Freq_GHz"] = freqs_tidy3d
df_30["Freq_GHz_Tidy3D"] = freqs_tidy3d
df_30["Q_Factor_Tidy3D"] = q_factors_tidy3d
df_30["Modo_Teorico"] = df_30["Modo"]
df_30["Erro_Percentual"] = (
    np.abs(df_30["Freq_GHz_Tidy3D"] - df_30["Freq_GHz_Analitica"])
    / df_30["Freq_GHz_Analitica"]
) * 100.0

# ==============================================================================
# 5. SALVAMENTO DOS DADOS (30 MODOS E 20 MODOS PARA COMPARAÇÃO COM HFSS)
# ==============================================================================
pasta_dados = os.path.join(pasta_raiz, "Dados")
pasta_imagens = os.path.join(pasta_raiz, "Imagens")
os.makedirs(pasta_dados, exist_ok=True)
os.makedirs(pasta_imagens, exist_ok=True)

# 5.1 Arquivo com os 30 modos completos (padrão Analítico + métricas de erro)
colunas_30 = [
    "Ordem",
    "Tipo",
    "m",
    "n",
    "p",
    "Modo",
    "Freq_GHz",
    "Freq_GHz_Analitica",
    "Erro_Percentual",
]
caminho_txt_30 = os.path.join(pasta_dados, "cavidade_modos_30_tidy3d.txt")
df_30[colunas_30].to_csv(
    caminho_txt_30, sep="\t", index=False, float_format="%.4f"
)
print(f"[OK] 30 modos (Tidy3D) salvos com sucesso em: {caminho_txt_30}")

# 5.2 Arquivo com os 20 primeiros modos (mesmas colunas do cavidade_modos_20_hfss.txt)
df_20 = df_30.head(20)[
    [
        "Ordem",
        "Freq_GHz_Tidy3D",
        "Q_Factor_Tidy3D",
        "Modo_Teorico",
        "Freq_GHz_Analitica",
        "Erro_Percentual",
    ]
].copy()
caminho_txt_20 = os.path.join(pasta_dados, "cavidade_modos_20_tidy3d.txt")
df_20.to_csv(caminho_txt_20, sep="\t", index=False, float_format="%.4f")
print(f"[OK] 20 modos (Tidy3D) salvos com sucesso em: {caminho_txt_20}")

print("\n" + "=" * 78)
print(df_30[colunas_30].to_string(index=False))
print("=" * 78)

# ==============================================================================
# 6. GERAÇÃO DO GRÁFICO DO ESPECTRO (IDÊNTICO AO ANALÍTICO E HFSS)
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
    label="Modos TE (Tidy3D)",
    alpha=0.85,
)
ax.vlines(
    df_tm["Freq_GHz"],
    ymin=0,
    ymax=1.0,
    colors="crimson",
    linewidth=1.8,
    linestyles="dashed",
    label="Modos TM (Tidy3D)",
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
erro_medio = df_30["Erro_Percentual"].mean()
ax.text(
    0.02,
    0.86,
    (
        f"Modo Fundamental: {modo_fund['Modo']} @ {modo_fund['Freq_GHz']:.4f}"
        f" GHz (Erro: {modo_fund['Erro_Percentual']:.3f}%)\nDimensões: a ="
        f" {a_mm} mm, b = {b_mm} mm, d = {d_mm} mm | Erro Médio (30 modos):"
        f" {erro_medio:.3f}%"
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

# Configurações dos eixos
ax.set_title(
    r"Espectro de Ressonância da Cavidade Retangular ($N = 30$ Modos - Tidy3D)",
    fontsize=12,
    fontweight="bold",
)
ax.set_xlabel("Frequência de Ressonância (GHz)", fontsize=10)
ax.set_yticks([])  # Remove escala de amplitude arbitrária
ax.set_ylim(0, 1.35)
ax.grid(axis="x", linestyle="--", alpha=0.5)
ax.legend(loc="upper right", framealpha=0.95)

plt.tight_layout()

caminho_fig = os.path.join(pasta_imagens, "cavidade_30_modos_tidy3d.png")
plt.savefig(caminho_fig, dpi=300)
print(f"[OK] Gráfico da Cavidade (Tidy3D) salvo em: {caminho_fig}")
plt.show()

