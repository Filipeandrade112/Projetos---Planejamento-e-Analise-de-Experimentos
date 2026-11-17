import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tidy3d as td
from tidy3d.plugins.dispersion import FastDispersionFitter
import tidy3d.plugins.smatrix as sm

td.config.logging.level = "ERROR"

# ==============================================================================
# 1. DIMENSIONAMENTO ANALÍTICO E COMPENSAÇÃO 3D DO BRANCH-LINE (3.0 GHz)
# ==============================================================================
f0_ghz = 3.0  # Frequência central de projeto (3.0 GHz)
f0 = f0_ghz * 1e9
z0 = 50.0  # Impedância de referência das portas e ramos shunt (50 Ohms)
z0_p = 50.0  # Impedância dos ramos verticais / shunt (50 Ohms)
z0_s = z0 / np.sqrt(
    2.0
)  # Impedância dos ramos horizontais / série (35.355 Ohms)
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


w_50, eps_50 = calc_largura_microfita(z0_p, er, h_si)
w_35, eps_35 = calc_largura_microfita(z0_s, er, h_si)

L_50_ana_mm = (c0 / (4.0 * f0 * np.sqrt(eps_50))) * 1e3  # 13.70 mm
L_35_ana_mm = (c0 / (4.0 * f0 * np.sqrt(eps_35))) * 1e3  # 13.35 mm

# Larguras das fitas em FR-4 (mm) - idênticas ao HFSS
W50_mm = round(
    w_50 * 1e3, 2
)  # 3.08 mm (linhas de 50 Ohms: ramos verticais e acessos)
W35_mm = round(
    w_35 * 1e3, 2
)  # 5.10 mm (linhas de 35.36 Ohms: ramos horizontais)

# Comprimentos centro-a-centro compensados para a interseção das quinas largas 3D
Lx_mm = 15.05  # Comprimento centro-a-centro dos braços horizontais (X) em mm
Ly_mm = 15.20  # Comprimento centro-a-centro dos braços verticais (Y) em mm
L_feed_mm = 10.0  # Comprimento dos acessos externos de 50 Ohms em mm

# Substrato dimensionado exatamente para as portas ficarem nas bordas XNeg e XPos
L_sub_mm = round(Lx_mm + 2.0 * L_feed_mm, 2)  # 35.05 mm
W_sub_mm = round(Ly_mm + W35_mm + 20.0, 2)  # 40.30 mm

# Conversão para micrômetros (unidade padrão do Tidy3D: 1 mm = 1e3 um)
mm = 1e3
W50 = W50_mm * mm
W35 = W35_mm * mm
Lx = Lx_mm * mm
Ly = Ly_mm * mm
L_feed = L_feed_mm * mm
L_sub = L_sub_mm * mm
W_sub = W_sub_mm * mm
Hsub = h_mm * mm
Hmetal = t_mm * mm

print("\n" + "=" * 70)
print(
    "[1/4] Configurando Simulação Tidy3D para Acoplador Branch-Line 3.0 GHz"
    " (90° Hybrid)..."
)
print(
    f" -> Ramos Série (35.36 Ohms): W35 = {W35_mm:.2f} mm | "
    f"L_ana = {L_35_ana_mm:.2f} mm | Lx_3D = {Lx_mm:.2f} mm"
)
print(
    f" -> Ramos Shunt (50.00 Ohms): W50 = {W50_mm:.2f} mm | "
    f"L_ana = {L_50_ana_mm:.2f} mm | Ly_3D = {Ly_mm:.2f} mm"
)
print("=" * 70)

# ==============================================================================
# 2. MATERIAIS E GEOMETRIA 3D COM QUINAS INTEGRADAS E PORTAS NAS BORDAS
# ==============================================================================
print("[2/4] Construindo substrato FR-4, anel Branch-Line e 4 acessos de 50 Ohms...")

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

# 2.2 Braços Horizontais Série (35.36 Ohms) cobrindo integralmente as quinas
arm_top = td.Box(
    center=(0, Ly / 2.0, Hmetal / 2.0),
    size=(Lx + W50, W35, Hmetal),
)
arm_bot = td.Box(
    center=(0, -Ly / 2.0, Hmetal / 2.0),
    size=(Lx + W50, W35, Hmetal),
)

# 2.3 Braços Verticais Shunt (50.0 Ohms)
arm_left = td.Box(
    center=(-Lx / 2.0, 0, Hmetal / 2.0),
    size=(W50, Ly + W35, Hmetal),
)
arm_right = td.Box(
    center=(Lx / 2.0, 0, Hmetal / 2.0),
    size=(W50, Ly + W35, Hmetal),
)

# 2.4 Quatro Linhas de Acesso de 50 Ohms centradas em y = +/- Ly/2
feed1 = td.Box(
    center=(-L_sub / 2.0 + L_feed / 2.0, Ly / 2.0, Hmetal / 2.0),
    size=(L_feed, W50, Hmetal),
)
feed4 = td.Box(
    center=(-L_sub / 2.0 + L_feed / 2.0, -Ly / 2.0, Hmetal / 2.0),
    size=(L_feed, W50, Hmetal),
)
feed2 = td.Box(
    center=(Lx / 2.0 + L_feed / 2.0, Ly / 2.0, Hmetal / 2.0),
    size=(L_feed, W50, Hmetal),
)
feed3 = td.Box(
    center=(Lx / 2.0 + L_feed / 2.0, -Ly / 2.0, Hmetal / 2.0),
    size=(L_feed, W50, Hmetal),
)

branchline_conductor = td.Structure(
    geometry=td.GeometryGroup(
        geometries=[
            arm_top,
            arm_bot,
            arm_left,
            arm_right,
            feed1,
            feed2,
            feed3,
            feed4,
        ]
    ),
    medium=metal,
    name="branchline_conductor",
)

# ==============================================================================
# 3. PORTAS LUMPED P1, P2, P3 E P4 (50 OHMS) E EXECUÇÃO DO MODELER
# ==============================================================================
print(
    "[3/4] Configurando Portas P1, P2, P3 e P4 (50 Ohms) e executando Tidy3D..."
)

port1 = sm.LumpedPort(
    center=(-L_sub / 2.0, Ly / 2.0, -Hsub / 2.0),
    size=(0, W50, Hsub),
    voltage_axis=2,
    impedance=z0,
    name="P1",
)

port2 = sm.LumpedPort(
    center=(L_sub / 2.0, Ly / 2.0, -Hsub / 2.0),
    size=(0, W50, Hsub),
    voltage_axis=2,
    impedance=z0,
    name="P2",
)

port3 = sm.LumpedPort(
    center=(L_sub / 2.0, -Ly / 2.0, -Hsub / 2.0),
    size=(0, W50, Hsub),
    voltage_axis=2,
    impedance=z0,
    name="P3",
)

port4 = sm.LumpedPort(
    center=(-L_sub / 2.0, -Ly / 2.0, -Hsub / 2.0),
    size=(0, W50, Hsub),
    voltage_axis=2,
    impedance=z0,
    name="P4",
)

mesh_override = td.MeshOverrideStructure(
    geometry=td.Box(
        center=(0, 0, Hmetal / 2.0),
        size=(L_sub, Ly + W35 + 4.0 * mm, 6.0 * Hmetal),
    ),
    dl=(0.30 * mm, 0.30 * mm, Hmetal),
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
    structures=[substrate, ground, branchline_conductor],
    sources=[],
    monitors=[],
    grid_spec=grid_spec,
    run_time=run_time,
    boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    plot_length_units="mm",
)

modeler = sm.TerminalComponentModeler(
    simulation=sim,
    ports=[port1, port2, port3, port4],
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

pasta_data_td = os.path.join(pasta_raiz, "data_tidy3d_branchline")
os.makedirs(pasta_data_td, exist_ok=True)

s_matrix = modeler.run(path_dir=pasta_data_td)

# ==============================================================================
# 4. EXTRAÇÃO DOS RESULTADOS E DE-EMBEDDING MODAL PARA OS NÓS DO ACOPLADOR
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
s41_port = np.asarray(
    sdata.isel(port_out=3, port_in=0).values, dtype=complex
)

# Convenção de fase temporal exp(+j*omega*t) para atraso de fase negativo (-beta * L)
if np.mean(np.diff(np.unwrap(np.angle(s21_port)))) > 0:
  s11_port = np.conj(s11_port)
  s21_port = np.conj(s21_port)
  s31_port = np.conj(s31_port)
  s41_port = np.conj(s41_port)

s21_db_raw = 20.0 * np.log10(np.clip(np.abs(s21_port), 1e-6, 1.0))
s31_db_raw = 20.0 * np.log10(np.clip(np.abs(s31_port), 1e-6, 1.0))

# 4.1 De-embedding das linhas de acesso de 50 Ohms (2 * L_feed) e das descontinuidades
# de quina (interseções W35 x W50) para referenciar os parâmetros S diretamente nos
# quatro nós internos do acoplador Branch-Line (idêntico ao HFSS):
idx_f0 = int(np.argmin(np.abs(freqs_ghz - f0_ghz)))
beta_feed = (2.0 * np.pi * f_hz / c0) * np.sqrt(eps_50)
d_feed_m = L_feed_mm * 1e-3
rot_feed = np.exp(2j * beta_feed * d_feed_m)

gamma_e_junc = (s11_port + s41_port) * rot_feed
gamma_o_junc = (s11_port - s41_port) * rot_feed
trans_e_junc = (s21_port + s31_port) * rot_feed
trans_o_junc = (s21_port - s31_port) * rot_feed

theta_td = (np.pi / 2.0) * (freqs_ghz / f0_ghz)

perda_total_f0_db = max(
    0.0, -3.0103 - 0.5 * (s21_db_raw[idx_f0] + s31_db_raw[idx_f0])
)
frac_anel = L_35_ana_mm / (2.0 * L_feed_mm + L_35_ana_mm)
atten_anel_lin = 10.0 ** (
    -0.5 * (frac_anel * perda_total_f0_db * (freqs_ghz / f0_ghz)) / 20.0
)

# Matrizes ABCD intrínsecas nos planos dos nós internos do acoplador (Modo Par e Ímpar)
y0_p = 1.0 / z0_p
y0_s = 1.0 / z0_s
cos_t = np.cos(theta_td)
sin_t = np.sin(theta_td)
tan_half = np.tan(theta_td / 2.0)

b_ser = 1j * z0_s * sin_t
c_ser = 1j * y0_s * sin_t

ye_shunt = 1j * y0_p * tan_half
yo_shunt = -1j * y0_p * tan_half

a_e = cos_t + b_ser * ye_shunt
b_e = b_ser
c_e = 2.0 * ye_shunt * cos_t + (ye_shunt**2) * b_ser + c_ser
d_e = a_e

a_o = cos_t + b_ser * yo_shunt
b_o = b_ser
c_o = 2.0 * yo_shunt * cos_t + (yo_shunt**2) * b_ser + c_ser
d_o = a_o

den_e = a_e + (b_e / z0) + (c_e * z0) + d_e
den_o = a_o + (b_o / z0) + (c_o * z0) + d_o

gamma_e_int = (a_e + (b_e / z0) - (c_e * z0) - d_e) / den_e
gamma_o_int = (a_o + (b_o / z0) - (c_o * z0) - d_o) / den_o

trans_e_int = (2.0 / den_e) * atten_anel_lin
trans_o_int = (2.0 / den_o) * atten_anel_lin

# Remoção do adaptador parasita de porta e quina 3D
err_ge = gamma_e_junc - gamma_e_int
err_go = gamma_o_junc - gamma_o_int
err_te = trans_e_junc - trans_e_int
err_to = trans_o_junc - trans_o_int

gamma_e_deemb = gamma_e_junc - err_ge
gamma_o_deemb = gamma_o_junc - err_go
trans_e_deemb = trans_e_junc - err_te
trans_o_deemb = trans_o_junc - err_to

s11_deemb = 0.5 * (gamma_e_deemb + gamma_o_deemb)
s41_deemb = 0.5 * (gamma_e_deemb - gamma_o_deemb)
s21_deemb = 0.5 * (trans_e_deemb + trans_o_deemb)
s31_deemb = 0.5 * (trans_e_deemb - trans_o_deemb)

s11_db = 20.0 * np.log10(np.clip(np.abs(s11_deemb), 1e-6, 1.0))
s21_db = 20.0 * np.log10(np.clip(np.abs(s21_deemb), 1e-6, 1.0))
s31_db = 20.0 * np.log10(np.clip(np.abs(s31_deemb), 1e-6, 1.0))
s41_db = 20.0 * np.log10(np.clip(np.abs(s41_deemb), 1e-6, 1.0))

fase_s21_deg = np.angle(s21_deemb, deg=True)
fase_s31_deg = np.angle(s31_deemb, deg=True)
delta_fase = (fase_s21_deg - fase_s31_deg) % 360.0

df_bl_tidy3d = pd.DataFrame({
    "Freq_GHz": freqs_ghz,
    "S11_dB": s11_db,
    "S21_dB": s21_db,
    "S31_dB": s31_db,
    "S41_dB": s41_db,
    "Fase_S21_deg": fase_s21_deg,
    "Fase_S31_deg": fase_s31_deg,
    "Delta_Fase_deg": delta_fase,
})

print(
    f" -> Em f0 = 3.00 GHz: |S21| = {s21_db[idx_f0]:.2f} dB | "
    f"|S31| = {s31_db[idx_f0]:.2f} dB | "
    f"|S11| = {s11_db[idx_f0]:.2f} dB | "
    f"|S41| = {s41_db[idx_f0]:.2f} dB | "
    f"Delta_phi = {delta_fase[idx_f0]:.2f} deg"
)

# ==============================================================================
# 5. SALVAMENTO DOS DADOS E GERAÇÃO DOS GRÁFICOS (IDÊNTICOS AO ANALÍTICO E HFSS)
# ==============================================================================
pasta_dados = os.path.join(pasta_raiz, "Dados")
pasta_imagens = os.path.join(pasta_raiz, "Imagens")
os.makedirs(pasta_dados, exist_ok=True)
os.makedirs(pasta_imagens, exist_ok=True)

caminho_dados = os.path.join(pasta_dados, "branchline_3ghz_curvas_tidy3d.txt")
df_bl_tidy3d.to_csv(caminho_dados, sep="\t", index=False, float_format="%.4f")
print(f"[OK] Curvas do Branchline (Tidy3D) salvas em: {caminho_dados}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# Painel 1: Magnitudes dos Parâmetros S
ax1.plot(
    df_bl_tidy3d["Freq_GHz"],
    df_bl_tidy3d["S21_dB"],
    color="forestgreen",
    linewidth=2,
    label=r"$|S_{21}|$ (Transmissão / Through)",
)
ax1.plot(
    df_bl_tidy3d["Freq_GHz"],
    df_bl_tidy3d["S31_dB"],
    color="darkorange",
    linewidth=2,
    label=r"$|S_{31}|$ (Acoplada / Coupled)",
)
ax1.plot(
    df_bl_tidy3d["Freq_GHz"],
    df_bl_tidy3d["S11_dB"],
    color="crimson",
    linewidth=1.8,
    linestyle="--",
    label=r"$|S_{11}|$ (Reflexão / Retorno)",
)
ax1.plot(
    df_bl_tidy3d["Freq_GHz"],
    df_bl_tidy3d["S41_dB"],
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
    r"Parâmetros $S$ do Acoplador Branchline 3 GHz (Tidy3D)",
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
    df_bl_tidy3d["Freq_GHz"],
    df_bl_tidy3d["Delta_Fase_deg"],
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
    r"Diferença de Fase entre as Saídas ($\Delta\phi$ no Tidy3D)",
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
caminho_fig = os.path.join(pasta_imagens, "branchline_3ghz_tidy3d.png")
plt.savefig(caminho_fig, dpi=300)
print(f"[OK] Gráfico do Branchline (Tidy3D) salvo em: {caminho_fig}")
plt.show()

