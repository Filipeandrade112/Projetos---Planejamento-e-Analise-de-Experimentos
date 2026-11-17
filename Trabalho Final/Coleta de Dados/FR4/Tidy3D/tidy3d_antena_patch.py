import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tidy3d as td
from tidy3d.plugins.dispersion import FastDispersionFitter
import tidy3d.plugins.smatrix as sm

td.config.logging.level = "ERROR"

# ==============================================================================
# 1. PARÂMETROS DA ANTENA PATCH (IDÊNTICOS AO ANALÍTICO E AO HFSS)
# ==============================================================================
f0_ghz = 3.0  # Frequência central de operação (3.0 GHz)
f0 = f0_ghz * 1e9
er = 4.4  # Permissividade relativa do substrato FR-4
tan_d = 0.02  # Tangente de perdas do FR-4
h_mm = 1.6  # Altura do dielétrico (mm)
t_mm = 0.035  # Espessura do cobre (1 oz = 35 um)
z0_linha = 50.0  # Impedância de referência (50 Ohms)

sigma_cu_S_per_m = 5.8e7  # Condutividade do cobre (S/m)
sigma_cu = sigma_cu_S_per_m / 1e6  # Condutividade em S/um (padrão Tidy3D)

# Varredura de frequência idêntica ao Analítico e HFSS: 2.7 GHz a 3.3 GHz (1601 pontos)
freq_start = 2.7e9
freq_stop = 3.3e9
num_points = 1601
fwidth = freq_stop - freq_start
wavelength = td.C_0 / f0
freqs = np.linspace(freq_start, freq_stop, num_points)
run_time = 60.0 / fwidth

c0 = 299792458.0
h_si = h_mm * 1e-3

# Cálculo de referência do modelo de linha de transmissão (Balanis):
W_ana = (c0 / (2.0 * f0)) * np.sqrt(2.0 / (er + 1.0))
eps_eff_patch = (er + 1.0) / 2.0 + ((er - 1.0) / 2.0) * (
    1.0 + 12.0 * (h_si / W_ana)
) ** (-0.5)
delta_L = (
    0.412
    * h_si
    * ((eps_eff_patch + 0.3) * (W_ana / h_si + 0.264))
    / ((eps_eff_patch - 0.258) * (W_ana / h_si + 0.8))
)
delta_L_mm = delta_L * 1e3

# Dimensões físicas da antena patch (mm)
W_patch_mm = 30.41  # Largura do patch (30.41 mm)
L_patch_mm = 23.36  # Comprimento do patch para ressonância 3D em 3.000 GHz
y0_inset_mm = 7.38  # Profundidade do inset feed para casamento em 50 Ohms
W_feed_mm = 3.08  # Largura da linha de microfita de 50 Ohms
g_notch_mm = 1.0  # Fresta lateral do inset feed (mm)
L_feed_mm = 15.0  # Extensão externa da linha de alimentação até a borda (mm)

L_sub_mm = round(L_patch_mm + 2.0 * L_feed_mm, 2)  # 53.36 mm
W_sub_mm = round(W_patch_mm + 30.0, 2)  # 60.41 mm

# Conversão para micrômetros (unidade padrão do Tidy3D: 1 mm = 1e3 um)
mm = 1e3
Wp = W_patch_mm * mm
Lp = L_patch_mm * mm
y0 = y0_inset_mm * mm
Wf = W_feed_mm * mm
g = g_notch_mm * mm
Lfeed = L_feed_mm * mm
Lsub = L_sub_mm * mm
Wsub = W_sub_mm * mm
Hsub = h_mm * mm
Hmetal = t_mm * mm

print("\n" + "=" * 70)
print(
    "[1/4] Configurando Simulação Tidy3D para Antena Patch Retangular (3.0"
    " GHz)..."
)
print(
    f" -> W = {W_patch_mm:.2f} mm | L = {L_patch_mm:.2f} mm | y0 ="
    f" {y0_inset_mm:.2f} mm"
)
print("=" * 70)

# ==============================================================================
# 2. MATERIAIS E CONSTRUÇÃO GEOMÉTRICA 3D (COM INSET FEED)
# ==============================================================================
print(
    "[2/4] Construindo substrato FR-4, plano de terra e patch com inset feed..."
)

dielectric = FastDispersionFitter.constant_loss_tangent_model(
    er,
    tan_d,
    (freq_start, freq_stop),
    tolerance_rms=1e-3,
    number_sampling_frequency=100,
)
metal = td.LossyMetalMedium(
    conductivity=sigma_cu,
    frequency_range=(freq_start, freq_stop),
)

# Substrato dielétrico com face superior em Z = 0
substrate = td.Structure(
    geometry=td.Box(
        center=(0, 0, -Hsub / 2.0),
        size=(Lsub, Wsub, Hsub),
    ),
    medium=dielectric,
    name="substrate",
)

# Plano de terra na face inferior do substrato (Z = -Hsub)
ground = td.Structure(
    geometry=td.Box(
        center=(0, 0, -Hsub - Hmetal / 2.0),
        size=(Lsub, Wsub, Hmetal),
    ),
    medium=metal,
    name="ground",
)

# Geometria do Patch com recortes (notches) do Inset Feed usando CSG / Box:
# Parte posterior do patch (sem recorte): de x = -Lp/2 + y0 até x = +Lp/2
patch_back = td.Box(
    center=(y0 / 2.0, 0, Hmetal / 2.0),
    size=(Lp - y0, Wp, Hmetal),
)

# Abas laterais superiores e inferiores na região do inset (x = -Lp/2 até -Lp/2 + y0)
w_wing = (Wp - Wf) / 2.0 - g
y_wing_center = Wf / 2.0 + g + w_wing / 2.0

patch_wing_top = td.Box(
    center=(-Lp / 2.0 + y0 / 2.0, y_wing_center, Hmetal / 2.0),
    size=(y0, w_wing, Hmetal),
)
patch_wing_bot = td.Box(
    center=(-Lp / 2.0 + y0 / 2.0, -y_wing_center, Hmetal / 2.0),
    size=(y0, w_wing, Hmetal),
)

# Linha de alimentação (da borda x = -Lsub/2 até o fundo do inset x = -Lp/2 + y0)
len_feed_total = Lfeed + y0
feed_box = td.Box(
    center=(-Lsub / 2.0 + len_feed_total / 2.0, 0, Hmetal / 2.0),
    size=(len_feed_total, Wf, Hmetal),
)

conductor_geom = td.GeometryGroup(
    geometries=[patch_back, patch_wing_top, patch_wing_bot, feed_box]
)

patch_structure = td.Structure(
    geometry=conductor_geom,
    medium=metal,
    name="patch_and_feed",
)

# ==============================================================================
# 3. PORTA LUMPED, MALHA FDTD E TERMINAL COMPONENT MODELER
# ==============================================================================
print("[3/4] Configurando Porta Lumped (50 Ohms), malha FDTD e executando...")

LP1 = sm.LumpedPort(
    center=(-Lsub / 2.0, 0, -Hsub / 2.0),
    size=(0, Wf, Hsub),
    voltage_axis=2,
    impedance=z0_linha,
    name="LP1",
)

mon_field = td.FieldMonitor(
    center=(0, 0, 0),
    size=(td.inf, td.inf, 0),
    freqs=[f0],
    name="field_z0",
)

mesh_override_patch = td.MeshOverrideStructure(
    geometry=td.Box(
        center=(0, 0, Hmetal / 2.0),
        size=(Lsub, Wp + 4.0 * mm, 6.0 * Hmetal),
    ),
    dl=(0.4 * mm, 0.4 * mm, Hmetal),
)

grid_spec = td.GridSpec.auto(
    wavelength=wavelength,
    min_steps_per_wvl=20,
    override_structures=[mesh_override_patch],
)

sim_size = (
    Lsub + 1.0 * wavelength,
    Wsub + 1.0 * wavelength,
    Hsub + 1.0 * wavelength,
)

sim = td.Simulation(
    center=(0, 0, 0),
    size=sim_size,
    structures=[substrate, ground, patch_structure],
    monitors=[mon_field],
    grid_spec=grid_spec,
    run_time=run_time,
    boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    plot_length_units="mm",
)

tcm = sm.TerminalComponentModeler(
    simulation=sim,
    ports=[LP1],
    freqs=freqs,
)

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(diretorio_atual).lower() in [
    "analiticos",
    "hfss",
    "tidy3d",
]:
  pasta_raiz = os.path.dirname(diretorio_atual)
else:
  pasta_raiz = diretorio_atual

pasta_data_td = os.path.join(pasta_raiz, "data_tidy3d_patch")
os.makedirs(pasta_data_td, exist_ok=True)

s_matrix = tcm.run(path_dir=pasta_data_td)

# ==============================================================================
# 4. EXTRAÇÃO DE S11 E IMPEDÂNCIA ZIN E SALVAMENTO NA PASTA 'Dados'
# ==============================================================================
print("[4/4] Extraindo S11 e Impedância de Entrada Zin e salvando arquivos...")

sdata = s_matrix.data
S11_da = sdata.isel(port_out=0, port_in=0)

freq_coord = next(k for k in sdata.coords.keys() if "f" in k.lower())
f_hz = np.asarray(sdata.coords[freq_coord].values, dtype=float)
freqs_ghz = f_hz / 1e9

s11_complex = np.asarray(S11_da.values, dtype=complex)

# Convenção de sinal temporal exp(+j*omega*t) para impedância reativa (indutiva -> capacitiva)
if np.mean(np.diff(np.unwrap(np.angle(s11_complex)))) > 0:
  s11_complex = np.conj(s11_complex)

s11_db = 20.0 * np.log10(np.clip(np.abs(s11_complex), 1e-6, 1.0))

# De-embedding da linha de alimentação (L_feed + y0 + delta_L) até o plano do Patch,
# idêntico ao pós-processamento realizado no HFSS:
u_feed = W_feed_mm / h_mm
eps_eff_feed = (er + 1.0) / 2.0 + ((er - 1.0) / 2.0) * (
    1.0 + 12.0 / u_feed
) ** (-0.5)
d_deembed_m = (L_feed_mm + y0_inset_mm + delta_L_mm) * 1e-3

beta_feed = (2.0 * np.pi * f_hz / c0) * np.sqrt(eps_eff_feed)
gamma_patch = s11_complex * np.exp(2j * beta_feed * d_deembed_m)
z_patch = z0_linha * (1.0 + gamma_patch) / (1.0 - gamma_patch)

re_z = np.real(z_patch)
im_z = np.imag(z_patch)

df_patch_tidy3d = pd.DataFrame({
    "Freq_GHz": freqs_ghz,
    "S11_dB": s11_db,
    "Zin_Real": re_z,
    "Zin_Imag": im_z,
})

idx_min = int(df_patch_tidy3d["S11_dB"].idxmin())
f_res_td = df_patch_tidy3d.loc[idx_min, "Freq_GHz"]
s11_min_td = df_patch_tidy3d.loc[idx_min, "S11_dB"]
zin_re_res = df_patch_tidy3d.loc[idx_min, "Zin_Real"]
zin_im_res = df_patch_tidy3d.loc[idx_min, "Zin_Imag"]

print(
    f" -> Ressonância Tidy3D: f_res = {f_res_td:.4f} GHz | "
    f"|S11|_min = {s11_min_td:.2f} dB | "
    f"Zin(patch) = {zin_re_res:.1f} + j({zin_im_res:.1f}) Ohms"
)

pasta_dados = os.path.join(pasta_raiz, "Dados")
pasta_imagens = os.path.join(pasta_raiz, "Imagens")
os.makedirs(pasta_dados, exist_ok=True)
os.makedirs(pasta_imagens, exist_ok=True)

caminho_dados = os.path.join(pasta_dados, "patch_3ghz_curvas_tidy3d.txt")
df_patch_tidy3d.to_csv(
    caminho_dados, sep="\t", index=False, float_format="%.4f"
)
print(f"[OK] Curvas da Antena Patch (Tidy3D) salvas em: {caminho_dados}")

# ==============================================================================
# 5. GERAÇÃO DOS GRÁFICOS (IDÊNTICOS AO ANALÍTICO E AO HFSS)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# Painel 1: Perda de Retorno S11 (dB)
ax1.plot(
    df_patch_tidy3d["Freq_GHz"],
    df_patch_tidy3d["S11_dB"],
    color="crimson",
    linewidth=2,
    label=r"$|S_{11}|$ Tidy3D",
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

mask_bw = df_patch_tidy3d["S11_dB"] <= -10.0
if np.any(mask_bw):
  f_min = df_patch_tidy3d.loc[mask_bw, "Freq_GHz"].iloc[0]
  f_max = df_patch_tidy3d.loc[mask_bw, "Freq_GHz"].iloc[-1]
  bw_mhz = (f_max - f_min) * 1e3
  ax1.fill_between(
      df_patch_tidy3d["Freq_GHz"],
      df_patch_tidy3d["S11_dB"],
      -10.0,
      where=mask_bw,
      color="pink",
      alpha=0.5,
      label=f"BW (-10 dB) = {bw_mhz:.1f} MHz",
  )

ax1.set_title(
    r"Parâmetro $S_{11}$ da Antena Patch Retangular (Tidy3D)",
    fontweight="bold",
    fontsize=11,
)
ax1.set_xlabel("Frequência (GHz)", fontsize=10)
ax1.set_ylabel(r"$|S_{11}|$ (dB)", fontsize=10)
ax1.set_xlim(2.7, 3.3)
ax1.set_ylim(-40, 2)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower right", fontsize=9)

# Painel 2: Impedância de Entrada Zin vs. Frequência (No Plano do Patch)
ax2.plot(
    df_patch_tidy3d["Freq_GHz"],
    df_patch_tidy3d["Zin_Real"],
    color="navy",
    linewidth=2,
    label=r"$\text{Re}(Z_{\text{in}})$ (Resistência)",
)
ax2.plot(
    df_patch_tidy3d["Freq_GHz"],
    df_patch_tidy3d["Zin_Imag"],
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
    r"Impedância de Entrada $Z_{\text{in}}$ vs. Frequência (Tidy3D)",
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
caminho_fig = os.path.join(pasta_imagens, "patch_3ghz_tidy3d.png")
plt.savefig(caminho_fig, dpi=300)
print(f"[OK] Gráfico da Antena Patch (Tidy3D) salvo em: {caminho_fig}")
plt.show()

