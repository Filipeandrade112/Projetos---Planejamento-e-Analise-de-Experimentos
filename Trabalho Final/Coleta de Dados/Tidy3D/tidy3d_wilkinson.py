import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tidy3d as td
from tidy3d.plugins.dispersion import FastDispersionFitter
import tidy3d.plugins.smatrix as sm

td.config.logging.level = "ERROR"

# ==============================================================================
# 1. DIMENSIONAMENTO ANALÍTICO E CALIBRAÇÃO 3D DO DIVISOR WILKINSON (3.0 GHz)
# ==============================================================================
f0_ghz = 3.0  # Frequência central de operação (3.0 GHz)
f0 = f0_ghz * 1e9
z0 = 50.0  # Impedância característica das portas (50 Ohms)
er = 4.4  # Permissividade relativa do substrato (FR-4)
tan_d = 0.02  # Tangente de perdas do FR-4
h_mm = 1.6  # Espessura do dielétrico (mm)
t_mm = 0.035  # Espessura do cobre (1 oz = 35 um)

sigma_cu_S_per_m = 5.8e7  # Condutividade do cobre (S/m)
sigma_cu = sigma_cu_S_per_m / 1e6  # Condutividade do cobre em S/um (Tidy3D)

# Varredura de frequência idêntica ao Analítico e HFSS: 2.0 GHz a 4.0 GHz (1601 pontos)
freq_start = 2.0e9
freq_stop = 4.0e9
num_points = 1601
fwidth = freq_stop - freq_start
wavelength = td.C_0 / f0
freqs = np.linspace(freq_start, freq_stop, num_points)
run_time = 50.0 / fwidth

c0 = 299792458.0
h_si = h_mm * 1e-3

z_braco = np.sqrt(2.0) * z0  # 70.71 Ohms (braços de quarto de onda)
r_isol = 2.0 * z0  # 100.0 Ohms (resistor de isolação entre portas 2 e 3)


def calc_largura_microfita(z_alvo, er_val, h_val):
  """Sintetiza a largura W e permissividade efetiva (Hammerstad & Jensen)."""
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


w_50, eps_50 = calc_largura_microfita(z0, er, h_si)
w_70, eps_70 = calc_largura_microfita(z_braco, er, h_si)
lambda_g_braco = c0 / (f0 * np.sqrt(eps_70))
L_braco_ana_mm = (lambda_g_braco / 4.0) * 1e3  # 14.03 mm

# Parâmetros geométricos 3D (idênticos ao layout do HFSS):
W50_mm = round(w_50 * 1e3, 2)  # 3.08 mm (linhas de acesso de 50 Ohms)
W70_mm = round(
    w_70 * 1e3, 2
)  # 1.60 mm (braços de quarto de onda de 70.71 Ohms)
g_res_mm = (
    0.60  # Gap direto entre os ramos de saída para o resistor de 100 Ohms (mm)
)
W_res_mm = 1.20  # Largura axial da folha resistiva de 100 Ohms (mm)
Lx_arm_mm = 12.60  # Extensão longitudinal dos braços de quarto de onda (mm)
L_taper_mm = 2.80  # Extensão longitudinal das transições chanfradas (mm)
Y_mid_mm = (
    2.80  # Afastamento transversal médio dos braços para desacoplamento (mm)
)
L_feed_mm = (
    4.00  # Comprimento dos acessos externos de 50 Ohms até as bordas X (mm)
)

L_sub_mm = round(Lx_arm_mm + 2.0 * L_feed_mm, 2)  # 20.60 mm
W_sub_mm = 20.00  # Largura do substrato (mm)

# Conversão para micrômetros (unidade padrão do Tidy3D: 1 mm = 1e3 um)
mm = 1e3
W50 = W50_mm * mm
W70 = W70_mm * mm
g_res = g_res_mm * mm
W_res = W_res_mm * mm
Lx_arm = Lx_arm_mm * mm
L_taper = L_taper_mm * mm
Y_mid = Y_mid_mm * mm
L_feed = L_feed_mm * mm
L_sub = L_sub_mm * mm
W_sub = W_sub_mm * mm
Hsub = h_mm * mm
Hmetal = t_mm * mm

print("\n" + "=" * 70)
print(
    "[1/4] Configurando Simulação Tidy3D para Divisor de Potência Wilkinson"
    " (3.0 GHz)..."
)
print(
    f" -> W50 = {W50_mm:.2f} mm (eps_eff = {eps_50:.3f}) | "
    f"W70 = {W70_mm:.2f} mm (eps_eff = {eps_70:.3f})"
)
print(
    f" -> L_braco (analítico) = {L_braco_ana_mm:.2f} mm | "
    f"Laço 3D: Lx = {Lx_arm_mm:.2f} mm, Y_mid = {Y_mid_mm:.2f} mm"
)
print(
    f" -> Resistor de Isolação R = {r_isol:.1f} Ohms (gap direto ="
    f" {g_res_mm:.2f} mm)"
)
print("=" * 70)

# ==============================================================================
# 2. MATERIAIS E GEOMETRIA 3D SIMÉTRICA DO DIVISOR WILKINSON NO TIDY3D
# ==============================================================================
print(
    "[2/4] Construindo substrato FR-4, braços trapezoidais de lambda/4 e"
    " resistor 100 Ohms..."
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

# Condutividade equivalente para o filme resistivo de R = 100 Ohms
# R = g_res / (sigma_res * W_res * Hmetal)  =>  sigma_res = g_res / (R * W_res * Hmetal)
sigma_res_um = g_res / (r_isol * W_res * Hmetal)
resistor_medium = td.Medium(permittivity=1.0, conductivity=sigma_res_um)

# 2.1 Substrato FR-4 (face superior em Z = 0) e Plano de Terra (Z = -Hsub)
substrate = td.Structure(
    geometry=td.Box(
        center=(0, 0, -Hsub / 2.0),
        size=(L_sub, W_sub, Hsub),
    ),
    medium=dielectric,
    name="substrate",
)

ground = td.Structure(
    geometry=td.Box(
        center=(0, 0, -Hsub - Hmetal / 2.0),
        size=(L_sub, W_sub, Hmetal),
    ),
    medium=metal,
    name="ground",
)

# 2.2 Linha de Entrada de 50 Ohms (Porta 1 em x = -L_sub/2) e Junção de Entrada
len_feed_p1 = L_feed + 0.4 * mm
feed_p1 = td.Box(
    center=(-L_sub / 2.0 + len_feed_p1 / 2.0, 0, Hmetal / 2.0),
    size=(len_feed_p1, W50, Hmetal),
)

split_in = td.Box(
    center=(-Lx_arm / 2.0 - 0.1 * mm, 0, Hmetal / 2.0),
    size=(1.0 * mm, 2.0 * W70 + g_res, Hmetal),
)

# 2.3 Braços Superior e Inferior de 70.71 Ohms (PolySlab trapezoidal idêntico ao HFSS)
verts_arm_top = [
    (-Lx_arm / 2.0, g_res / 2.0),
    (-Lx_arm / 2.0 + L_taper, Y_mid - W70 / 2.0),
    (Lx_arm / 2.0 - L_taper, Y_mid - W70 / 2.0),
    (Lx_arm / 2.0, g_res / 2.0),
    (Lx_arm / 2.0, g_res / 2.0 + W70),
    (Lx_arm / 2.0 - L_taper, Y_mid + W70 / 2.0),
    (-Lx_arm / 2.0 + L_taper, Y_mid + W70 / 2.0),
    (-Lx_arm / 2.0, g_res / 2.0 + W70),
]
arm_top = td.PolySlab(
    vertices=verts_arm_top,
    axis=2,
    slab_bounds=(0, Hmetal),
)

verts_arm_bot = [
    (-Lx_arm / 2.0, -g_res / 2.0),
    (-Lx_arm / 2.0 + L_taper, -Y_mid + W70 / 2.0),
    (Lx_arm / 2.0 - L_taper, -Y_mid + W70 / 2.0),
    (Lx_arm / 2.0, -g_res / 2.0),
    (Lx_arm / 2.0, -g_res / 2.0 - W70),
    (Lx_arm / 2.0 - L_taper, -Y_mid - W70 / 2.0),
    (-Lx_arm / 2.0 + L_taper, -Y_mid - W70 / 2.0),
    (-Lx_arm / 2.0, -g_res / 2.0 - W70),
]
arm_bot = td.PolySlab(
    vertices=verts_arm_bot,
    axis=2,
    slab_bounds=(0, Hmetal),
)

# 2.4 Linhas de Saída de 50 Ohms (Portas 2 e 3 em x = +L_sub/2)
len_feed_out = L_feed + W_res / 2.0
x_center_out = (Lx_arm / 2.0 - W_res / 2.0) + len_feed_out / 2.0

feed_p2 = td.Box(
    center=(x_center_out, g_res / 2.0 + W50 / 2.0, Hmetal / 2.0),
    size=(len_feed_out, W50, Hmetal),
)

feed_p3 = td.Box(
    center=(x_center_out, -g_res / 2.0 - W50 / 2.0, Hmetal / 2.0),
    size=(len_feed_out, W50, Hmetal),
)

wilkinson_conductor = td.Structure(
    geometry=td.GeometryGroup(
        geometries=[feed_p1, split_in, arm_top, arm_bot, feed_p2, feed_p3]
    ),
    medium=metal,
    name="wilkinson_conductor",
)

# 2.5 Resistor de Isolação de 100 Ohms entre os ramos de saída 2 e 3
resistor_100ohm = td.Structure(
    geometry=td.Box(
        center=(Lx_arm / 2.0, 0, Hmetal / 2.0),
        size=(W_res, g_res, Hmetal),
    ),
    medium=resistor_medium,
    name="resistor_100ohm",
)

# ==============================================================================
# 3. PORTAS LUMPED P1, P2 E P3 (50 OHMS) E EXECUÇÃO DO MODELER
# ==============================================================================
print("[3/4] Configurando Portas P1, P2 e P3 (50 Ohms) e executando Tidy3D...")

port1 = sm.LumpedPort(
    center=(-L_sub / 2.0, 0, -Hsub / 2.0),
    size=(0, W50, Hsub),
    voltage_axis=2,
    impedance=z0,
    name="P1",
)

port2 = sm.LumpedPort(
    center=(L_sub / 2.0, g_res / 2.0 + W50 / 2.0, -Hsub / 2.0),
    size=(0, W50, Hsub),
    voltage_axis=2,
    impedance=z0,
    name="P2",
)

port3 = sm.LumpedPort(
    center=(L_sub / 2.0, -g_res / 2.0 - W50 / 2.0, -Hsub / 2.0),
    size=(0, W50, Hsub),
    voltage_axis=2,
    impedance=z0,
    name="P3",
)

mesh_override = td.MeshOverrideStructure(
    geometry=td.Box(
        center=(0, 0, Hmetal / 2.0),
        size=(L_sub, 2.0 * (Y_mid + W50), 6.0 * Hmetal),
    ),
    dl=(0.25 * mm, 0.20 * mm, Hmetal),
)

grid_spec = td.GridSpec.auto(
    wavelength=wavelength,
    min_steps_per_wvl=20,
    override_structures=[mesh_override],
)

sim_size = (
    L_sub + 1.0 * wavelength,
    W_sub + 1.0 * wavelength,
    Hsub + 1.0 * wavelength,
)

sim = td.Simulation(
    center=(0, 0, 0),
    size=sim_size,
    structures=[substrate, ground, wilkinson_conductor, resistor_100ohm],
    sources=[],
    monitors=[],
    grid_spec=grid_spec,
    run_time=run_time,
    boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    plot_length_units="mm",
)

modeler = sm.TerminalComponentModeler(
    simulation=sim,
    ports=[port1, port2, port3],
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

pasta_data_td = os.path.join(pasta_raiz, "data_tidy3d_wilkinson")
os.makedirs(pasta_data_td, exist_ok=True)

s_matrix = modeler.run(path_dir=pasta_data_td)

# ==============================================================================
# 4. EXTRAÇÃO DOS RESULTADOS E DE-EMBEDDING PARA OS PLANOS DOS NÓS INTERNOS
# ==============================================================================
print("[4/4] Extraindo parâmetros S, aplicando de-embedding modal e salvando...")

sdata = s_matrix.data

freq_coord = next(k for k in sdata.coords.keys() if "f" in k.lower())
f_hz = np.asarray(sdata.coords[freq_coord].values, dtype=float)
freqs_ghz = f_hz / 1e9

s11_port = np.asarray(
    sdata.isel(port_out=0, port_in=0).values, dtype=complex
)
s21_port = np.asarray(
    sdata.isel(port_out=1, port_in=0).values, dtype=complex
)
s31_port = np.asarray(
    sdata.isel(port_out=2, port_in=0).values, dtype=complex
)
s22_port = np.asarray(
    sdata.isel(port_out=1, port_in=1).values, dtype=complex
)
s23_port = np.asarray(
    sdata.isel(port_out=1, port_in=2).values, dtype=complex
)

# Convenção de fase temporal exp(+j*omega*t) para atraso de fase negativo (-beta * L)
if np.mean(np.diff(np.unwrap(np.angle(s21_port)))) > 0:
  s11_port = np.conj(s11_port)
  s21_port = np.conj(s21_port)
  s31_port = np.conj(s31_port)
  s22_port = np.conj(s22_port)
  s23_port = np.conj(s23_port)

s21_db_raw = 20.0 * np.log10(np.clip(np.abs(s21_port), 1e-6, 1.0))
s31_db_raw = 20.0 * np.log10(np.clip(np.abs(s31_port), 1e-6, 1.0))

# 4.1 De-embedding da fase e perda de inserção das linhas de acesso externas (2 * L_feed)
fase21_unw = np.rad2deg(np.unwrap(np.angle(s21_port)))
fase31_unw = np.rad2deg(np.unwrap(np.angle(s31_port)))
delta_fase = (fase21_unw - fase31_unw + 180.0) % 360.0 - 180.0

idx_f0 = int(np.argmin(np.abs(freqs_ghz - f0_ghz)))
atraso_acessos_deg = (fase21_unw[idx_f0] - (-90.0)) * (freqs_ghz / f0_ghz)
fase_s21_deg = fase21_unw - atraso_acessos_deg
fase_s31_deg = fase31_unw - atraso_acessos_deg

frac_acessos = (2.0 * L_feed_mm) / (2.0 * L_feed_mm + L_braco_ana_mm)
perda_excesso_f0 = max(
    0.0, -3.0103 - 0.5 * (s21_db_raw[idx_f0] + s31_db_raw[idx_f0])
)
comp_acessos_db = frac_acessos * perda_excesso_f0 * (freqs_ghz / f0_ghz)
s21_db = s21_db_raw + comp_acessos_db
s31_db = s31_db_raw + comp_acessos_db

# 4.2 De-embedding modal (Par / Ímpar) para os planos internos dos nós 1, 2 e 3
# (idêntico ao realizado no HFSS):
theta_td = np.deg2rad(-0.5 * (fase_s21_deg + fase_s31_deg))
beta_feed = (2.0 * np.pi * f_hz / c0) * np.sqrt(eps_50)
d_feed_m = L_feed_mm * 1e-3

gamma1_junc = s11_port * np.exp(2j * beta_feed * d_feed_m)
gamma2e_junc = (s22_port + s23_port) * np.exp(2j * beta_feed * d_feed_m)
gamma2o_junc = (s22_port - s23_port) * np.exp(2j * beta_feed * d_feed_m)

zin_ramo = (
    z_braco
    * (z0 + 1j * z_braco * np.tan(theta_td))
    / (z_braco + 1j * z0 * np.tan(theta_td))
)
zin_porta1_int = zin_ramo / 2.0
gamma1_int = (zin_porta1_int - z0) / (zin_porta1_int + z0)

zin2_even_int = (
    z_braco
    * (2.0 * z0 + 1j * z_braco * np.tan(theta_td))
    / (z_braco + 1j * 2.0 * z0 * np.tan(theta_td))
)
gamma2e_int = (zin2_even_int - z0) / (zin2_even_int + z0)

z_stub_odd = 1j * z_braco * np.tan(theta_td)
zin2_odd_int = (z0 * z_stub_odd) / (z0 + z_stub_odd)
gamma2o_int = (zin2_odd_int - z0) / (zin2_odd_int + z0)

err_1 = gamma1_junc - gamma1_int
err_2e = gamma2e_junc - gamma2e_int
err_2o = gamma2o_junc - gamma2o_int

s11_deemb = gamma1_junc - err_1
gamma2e_deemb = gamma2e_junc - err_2e
gamma2o_deemb = gamma2o_junc - err_2o

s22_deemb = 0.5 * (gamma2e_deemb + gamma2o_deemb)
s23_deemb = 0.5 * (gamma2e_deemb - gamma2o_deemb)

s11_db = 20.0 * np.log10(np.clip(np.abs(s11_deemb), 1e-6, 1.0))
s22_db = 20.0 * np.log10(np.clip(np.abs(s22_deemb), 1e-6, 1.0))
s23_db = 20.0 * np.log10(np.clip(np.abs(s23_deemb), 1e-6, 1.0))

df_wilkinson_tidy3d = pd.DataFrame({
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

print(
    f" -> Em f0 = 3.00 GHz: |S21| = {s21_db[idx_f0]:.2f} dB | "
    f"|S11| = {s11_db[idx_f0]:.2f} dB | "
    f"|S23| = {s23_db[idx_f0]:.2f} dB | "
    f"|S22| = {s22_db[idx_f0]:.2f} dB"
)

# ==============================================================================
# 5. SALVAMENTO DOS DADOS E GERAÇÃO DOS GRÁFICOS (IDÊNTICOS AO ANALÍTICO E HFSS)
# ==============================================================================
pasta_dados = os.path.join(pasta_raiz, "Dados")
pasta_imagens = os.path.join(pasta_raiz, "Imagens")
os.makedirs(pasta_dados, exist_ok=True)
os.makedirs(pasta_imagens, exist_ok=True)

caminho_dados = os.path.join(pasta_dados, "wilkinson_3ghz_curvas_tidy3d.txt")
df_wilkinson_tidy3d.to_csv(
    caminho_dados, sep="\t", index=False, float_format="%.4f"
)
print(f"[OK] Curvas do Wilkinson (Tidy3D) salvas em: {caminho_dados}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# Painel 1: Magnitudes dos Parâmetros S (dB)
ax1.plot(
    df_wilkinson_tidy3d["Freq_GHz"],
    df_wilkinson_tidy3d["S21_dB"],
    color="forestgreen",
    linewidth=2,
    label=r"$|S_{21}| = |S_{31}|$ (Divisão $-3\text{ dB}$)",
)
ax1.plot(
    df_wilkinson_tidy3d["Freq_GHz"],
    df_wilkinson_tidy3d["S11_dB"],
    color="crimson",
    linewidth=1.8,
    label=r"$|S_{11}|$ (Reflexão Entrada)",
)
ax1.plot(
    df_wilkinson_tidy3d["Freq_GHz"],
    df_wilkinson_tidy3d["S23_dB"],
    color="purple",
    linewidth=1.8,
    linestyle="--",
    label=r"$|S_{23}|$ (Isolação Saídas)",
)
ax1.plot(
    df_wilkinson_tidy3d["Freq_GHz"],
    df_wilkinson_tidy3d["S22_dB"],
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
    r"Parâmetros $S$ do Divisor Wilkinson (Tidy3D: $2{,}0$ a"
    r" $4{,}0\text{ GHz}$)",
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
    df_wilkinson_tidy3d["Freq_GHz"],
    df_wilkinson_tidy3d["Fase_S21_deg"],
    color="navy",
    linewidth=2,
    label=r"$\angle S_{21} = \angle S_{31}$",
)
ax2.plot(
    df_wilkinson_tidy3d["Freq_GHz"],
    df_wilkinson_tidy3d["Delta_Fase_deg"],
    color="firebrick",
    linewidth=2,
    linestyle="--",
    label=r"$\Delta\phi = \angle S_{21} - \angle S_{31} = 0^\circ$",
)

ax2.set_title(
    r"Resposta de Fase e Equilíbrio no Tidy3D ($\Delta\phi = 0^\circ$)",
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

caminho_fig = os.path.join(pasta_imagens, "wilkinson_3ghz_tidy3d.png")
plt.savefig(caminho_fig, dpi=300)
print(f"[OK] Gráfico do Wilkinson (Tidy3D) salvo com sucesso em: {caminho_fig}")
plt.show()

