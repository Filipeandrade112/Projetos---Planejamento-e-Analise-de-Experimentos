import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tidy3d as td
from tidy3d.plugins.dispersion import FastDispersionFitter
import tidy3d.plugins.smatrix as sm

td.config.logging.level = "ERROR"

# ==============================================================================
# 1. PARÂMETROS DA LINHA DE MICROFITA (IDÊNTICOS AO ANALÍTICO E AO HFSS)
# ==============================================================================
# Frequências: Varredura de 1.0 GHz a 5.0 GHz com 1601 pontos
freq_start = 1.0e9
freq_stop = 5.0e9
num_points = 1601
fwidth = freq_stop - freq_start
wavelength = 2.0 * td.C_0 / fwidth
freqs = np.linspace(freq_start, freq_stop, num_points)
run_time = 40.0 / fwidth

# Substrato (FR-4 padrão) e condutor de cobre (1 oz = 35 um)
er = 4.4  # Permissividade relativa
tan_d = 0.02  # Tangente de perdas do dielétrico
sigma_cu_S_per_m = 5.8e7  # Condutividade do cobre (S/m)
sigma_cu = sigma_cu_S_per_m / 1e6  # Condutividade do cobre em S/um (Tidy3D)
z0_ref = 50.0  # Impedância de referência das portas (50 Ohms)

# Conversão de unidades (Tidy3D usa micrômetros por padrão: 1 mm = 1e3 um)
mm = 1e3
L_mm = 50.0  # Comprimento da linha (50 mm)
W_sub_mm = 30.0  # Largura do substrato (30 mm)
h_mm = 1.6  # Altura do dielétrico FR-4 (1.6 mm)
t_mm = 0.035  # Espessura do cobre (35 um)

# Síntese de Hammerstad & Jensen (idêntica ao script analítico e HFSS)
eta0 = 119.9169832 * np.pi


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


h_si = h_mm * 1e-3
t_si = t_mm * 1e-3
w_teste = np.linspace(0.5e-3, 10.0e-3, 100000)
z0_busca = np.array(
    [calc_z0_quasi_static(w, er, h_si, 0.0)[0] for w in w_teste]
)
W_ideal = w_teste[np.argmin(np.abs(z0_busca - z0_ref))]
W_mm = round(W_ideal * 1e3, 2)  # 3.08 mm
Zc0_ana, eps_eff0_ana = calc_z0_quasi_static(W_mm * 1e-3, er, h_si, t_si)

# Geometria em micrômetros (padrão Tidy3D)
length = L_mm * mm  # 50.0 mm
width_sub = W_sub_mm * mm  # 30.0 mm
width = W_mm * mm  # 3.08 mm
height = h_mm * mm  # 1.6 mm
thickness = t_mm * mm  # 0.035 mm

print("\n" + "=" * 70)
print(
    "[1/4] Configurando Simulação Tidy3D para Linha de Microfita (50 Ohms)..."
)
print(
    f" -> W = {W_mm:.2f} mm | L = {L_mm:.1f} mm | h = {h_mm:.2f} mm | Zc0 ="
    f" {Zc0_ana:.2f} Ohms"
)
print("=" * 70)

# ==============================================================================
# 2. PROPRIEDADES DOS MATERIAIS E GEOMETRIA 3D NO TIDY3D
# ==============================================================================
print("[2/4] Construindo materiais dispersivos (FR-4 e Cobre) e estruturas...")

FR4 = FastDispersionFitter.constant_loss_tangent_model(
    er,
    tan_d,
    (freq_start, freq_stop),
    tolerance_rms=1e-3,
    number_sampling_frequency=100,
)
PEC_Cu = td.LossyMetalMedium(
    conductivity=sigma_cu,
    frequency_range=(freq_start, freq_stop),
)

ground = td.Structure(
    geometry=td.Box(
        center=[0, 0, thickness / 2.0],
        size=[length, width_sub, thickness],
    ),
    medium=PEC_Cu,
    name="ground",
)

substrate = td.Structure(
    geometry=td.Box(
        center=[0, 0, thickness + height / 2.0],
        size=[length, width_sub, height],
    ),
    medium=FR4,
    name="substrate",
)

strip = td.Structure(
    geometry=td.Box(
        center=[0, 0, height + thickness + thickness / 2.0],
        size=[length, width, thickness],
    ),
    medium=PEC_Cu,
    name="strip",
)

# Malha refinada próxima à fita condutora
center_mesh = [0, 0, height + thickness + thickness / 2.0]
size_mesh = [length, 2.0 * width, 5.0 * thickness]

mesh_overrides = [
    td.MeshOverrideStructure(
        geometry=td.Box(
            center=center_mesh,
            size=size_mesh,
        ),
        dl=[length / 20.0, width / 30.0, thickness],
    ),
]

sim_center = (0, 0, thickness + height / 2.0)
sim_size = (
    length + 1.5 * wavelength,
    width_sub + 1.5 * wavelength,
    height + 2.0 * thickness + 1.5 * wavelength,
)

boundary_spec = td.BoundarySpec.all_sides(boundary=td.PML())

sim = td.Simulation(
    center=sim_center,
    size=sim_size,
    grid_spec=td.GridSpec.auto(
        min_steps_per_wvl=20,
        wavelength=wavelength,
        override_structures=mesh_overrides,
        snapping_points=(strip.geometry.bounds[0], strip.geometry.bounds[1]),
    ),
    structures=[ground, substrate, strip],
    sources=[],
    monitors=[],
    run_time=run_time,
    boundary_spec=boundary_spec,
    plot_length_units="mm",
)

# ==============================================================================
# 3. PORTAS LUMPED (50 OHMS) E EXECUÇÃO DO TERMINAL COMPONENT MODELER
# ==============================================================================
print("[3/4] Configurando Portas 1 e 2 de 50 Ohms e executando no Tidy3D...")

port1 = sm.LumpedPort(
    center=(-length / 2.0, 0, height / 2.0 + thickness),
    size=(0, width, height + 2.0 * thickness),
    voltage_axis=2,
    impedance=z0_ref,
    name="port1",
)

port2 = sm.LumpedPort(
    center=(length / 2.0, 0, height / 2.0 + thickness),
    size=(0, width, height + 2.0 * thickness),
    voltage_axis=2,
    impedance=z0_ref,
    name="port2",
)

modeler = sm.TerminalComponentModeler(
    simulation=sim,
    ports=[port1, port2],
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

pasta_data_td = os.path.join(pasta_raiz, "data_tidy3d")
os.makedirs(pasta_data_td, exist_ok=True)

s_matrix = modeler.run(path_dir=pasta_data_td)

# ==============================================================================
# 4. EXTRAÇÃO DOS PARÂMETROS S E SALVAMENTO NA PASTA 'Dados'
# ==============================================================================
print("[4/4] Extraindo parâmetros S, desdobrando a fase e salvando arquivos...")

sdata = s_matrix.data

S11 = sdata.isel(port_out=0, port_in=0)
S21 = sdata.isel(port_out=1, port_in=0)

freq_coord = next(k for k in sdata.coords.keys() if "f" in k.lower())
f_hz = np.asarray(sdata.coords[freq_coord].values, dtype=float)
freqs_ghz = f_hz / 1e9

s11_vals = np.asarray(S11.values, dtype=complex)
s12_vals = np.asarray(S21.values, dtype=complex)

# Magnitudes em dB e Fases em graus (mesma convenção do Analítico e HFSS)
s11_db = 20.0 * np.log10(np.clip(np.abs(s11_vals), 1e-6, 1.0))
s12_db = 20.0 * np.log10(np.clip(np.abs(s12_vals), 1e-6, 1.0))

# Convenção de fase temporal exp(+j*omega*t) para atraso negativo (-beta * L)
if np.mean(np.diff(np.unwrap(np.angle(s12_vals)))) > 0:
  s11_vals = np.conj(s11_vals)
  s12_vals = np.conj(s12_vals)

fase_s11_deg = np.rad2deg(np.angle(s11_vals))
fase_s12_unwrapped = np.rad2deg(np.unwrap(np.angle(s12_vals)))

if fase_s12_unwrapped[0] > 0:
  fase_s12_unwrapped -= 360.0

df_tidy3d = pd.DataFrame({
    "Freq_GHz": freqs_ghz,
    "S11_dB": s11_db,
    "S12_dB": s12_db,
    "Fase_S11_deg": fase_s11_deg,
    "Fase_S12_deg": fase_s12_unwrapped,
})

pasta_dados = os.path.join(pasta_raiz, "Dados")
pasta_imagens = os.path.join(pasta_raiz, "Imagens")
os.makedirs(pasta_dados, exist_ok=True)
os.makedirs(pasta_imagens, exist_ok=True)

caminho_dados = os.path.join(pasta_dados, "linha_microfita_curvas_tidy3d.txt")
df_tidy3d.to_csv(caminho_dados, sep="\t", index=False, float_format="%.4f")
print(f"[OK] Curvas do Tidy3D salvas em: {caminho_dados}")

# ==============================================================================
# 5. GERAÇÃO DOS GRÁFICOS (IDÊNTICOS AO ANALÍTICO E AO HFSS)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# Painel 1: Magnitudes dos Parâmetros S (|S11| e |S12| em dB)
ax1.plot(
    df_tidy3d["Freq_GHz"],
    df_tidy3d["S12_dB"],
    color="forestgreen",
    linewidth=2,
    label=r"$|S_{12}|$ (Transmissão / Inserção)",
)
ax1.plot(
    df_tidy3d["Freq_GHz"],
    df_tidy3d["S11_dB"],
    color="crimson",
    linewidth=1.8,
    linestyle="--",
    label=r"$|S_{11}|$ (Reflexão / Retorno)",
)

ax1.axvline(
    3.0, color="gray", linestyle=":", alpha=0.7, label=r"$f_0 = 3{,}0$ GHz"
)
ax1.set_title(
    r"Parâmetros $S_{11}$ e $S_{12}$ da Linha de Microfita ($L = 50\text{ mm}$"
    r" - Tidy3D)",
    fontweight="bold",
    fontsize=11,
)
ax1.set_xlabel("Frequência (GHz)", fontsize=10)
ax1.set_ylabel("Magnitude (dB)", fontsize=10)
ax1.set_xlim(1.0, 5.0)
ax1.set_ylim(-60, 2)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="center right", fontsize=9)

# Painel 2: Resposta de Fase Contínua (Unwrapped: -100° a -550°)
ax2.plot(
    df_tidy3d["Freq_GHz"],
    df_tidy3d["Fase_S12_deg"],
    color="navy",
    linewidth=2,
    label=r"$\angle S_{12}$ (Fase de Transmissão)",
)

ax2.set_title(
    r"Resposta de Fase $\angle S_{12}$ vs. Frequência (Tidy3D)",
    fontweight="bold",
    fontsize=11,
)
ax2.set_xlabel("Frequência (GHz)", fontsize=10)
ax2.set_ylabel("Fase (graus)", fontsize=10)
ax2.set_xlim(1.0, 5.0)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper right", fontsize=9)

plt.tight_layout()
caminho_fig = os.path.join(pasta_imagens, "linha_microfita_tidy3d.png")
plt.savefig(caminho_fig, dpi=300)
print(f"[OK] Gráfico do Tidy3D salvo com sucesso em: {caminho_fig}")
plt.show()

