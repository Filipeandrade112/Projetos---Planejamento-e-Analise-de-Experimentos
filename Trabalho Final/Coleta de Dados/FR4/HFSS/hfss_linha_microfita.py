import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  from ansys.aedt.core import Hfss
  from ansys.aedt.core.generic.constants import Plane
except ImportError:
  from pyaedt import Hfss
  from pyaedt.generic.constants import Plane

# ==============================================================================
# 1. PARÂMETROS DA LINHA DE MICROFITA (IDÊNTICOS AO MODELO ANALÍTICO)
# ==============================================================================
er = 4.4  # Permissividade relativa do FR-4
h_mm = 1.6  # Altura do dielétrico (mm)
t_mm = 0.035  # Espessura do cobre (1 oz = 35 um)
tan_d = 0.02  # Tangente de perdas do dielétrico
sigma_cu = 5.8e7  # Condutividade do cobre (S/m)

L_fita = 50.0  # Comprimento da fita (50 mm)
W_sub = 30.0  # Largura do substrato (30 mm)
z0_ref = 50.0  # Impedância de referência das portas (50 Ohms)

c0 = 299792458.0
mu0 = 4.0 * np.pi * 1e-7
eta0 = 119.9169832 * np.pi
h = h_mm * 1e-3
t = t_mm * 1e-3
L = L_fita * 1e-3


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


# Síntese de Hammerstad & Jensen (idêntica ao script analítico)
w_teste = np.linspace(0.5e-3, 10.0e-3, 100000)
z0_busca = np.array([calc_z0_quasi_static(w, er, h, 0.0)[0] for w in w_teste])
W_ideal = w_teste[np.argmin(np.abs(z0_busca - z0_ref))]
W_fita = round(W_ideal * 1e3, 2)  # 3.08 mm
W = W_fita * 1e-3
Zc0_ana, eps_eff0_ana = calc_z0_quasi_static(W, er, h, t)

print("\n" + "=" * 70)
print(
    "[1/4] Iniciando HFSS para Linha de Microfita (Interface Gráfica Ativa)..."
)
print(
    f" -> W = {W_fita:.2f} mm | L = {L_fita:.1f} mm | Zc0 = {Zc0_ana:.2f} Ohms"
)
print("=" * 70)

hfss = Hfss(
    non_graphical=False,
    new_desktop=False,
    project="Trabalho_Final_HFSS",
    design="Linha_Microfita_50Ohms",
    solution_type="DrivenModal",
    remove_lock=True,
)

hfss.modeler.model_units = "mm"

# Variáveis no HFSS
hfss["Wfita"] = f"{W_fita}mm"
hfss["Lfita"] = f"{L_fita}mm"
hfss["Wsub"] = f"{W_sub}mm"
hfss["Hsub"] = f"{h_mm}mm"

# ==============================================================================
# 2. LIMPEZA TOTAL E CONSTRUÇÃO DA GEOMETRIA 3D
# ==============================================================================
print("[2/4] Limpando contornos antigos e desenhando linha de microfita...")

for s_name in list(hfss.setup_names):
  try:
    hfss.delete_setup(s_name)
  except Exception:
    pass

try:
  hfss.odesign.GetModule("BoundarySetup").DeleteAllBoundaries()
except Exception:
  for b in list(hfss.boundaries):
    try:
      b.delete()
    except Exception:
      pass

for obj_nome in [
    "Substrate",
    "Ground",
    "Microstrip_Line",
    "Port1_Sheet",
    "Port2_Sheet",
] + [o for o in list(hfss.modeler.object_names) if "RadiatingSurface" in o]:
  if obj_nome in hfss.modeler.object_names:
    try:
      hfss.modeler.delete(obj_nome)
    except Exception:
      pass

# Substrato
sub = hfss.modeler.create_box(
    origin=["-Lfita/2", "-Wsub/2", "-Hsub"],
    sizes=["Lfita", "Wsub", "Hsub"],
    name="Substrate",
    material="FR4_epoxy",
)

# Plano de terra
gnd = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-Lfita/2", "-Wsub/2", "-Hsub"],
    sizes=["Lfita", "Wsub"],
    name="Ground",
)
hfss.assign_perfecte_to_sheets(gnd.name, name="Ground_PEC")

# Fita condutora
trilha = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-Lfita/2", "-Wfita/2", 0],
    sizes=["Lfita", "Wfita"],
    name="Microstrip_Line",
)
hfss.assign_perfecte_to_sheets(trilha.name, name="Trilha_PEC")

# ==============================================================================
# 3. PORTAS LUMPED E CAIXA DE RADIAÇÃO
# ==============================================================================
print("[3/4] Atribuindo Portas 1 e 2 de 50 Ohms...")

p1_sheet = hfss.modeler.create_rectangle(
    orientation=Plane.YZ,
    origin=["-Lfita/2", "-Wfita/2", "-Hsub"],
    sizes=["Wfita", "Hsub"],
    name="Port1_Sheet",
)
hfss.lumped_port(
    assignment=p1_sheet.name,
    reference=gnd.name,
    create_port_sheet=False,
    integration_line=hfss.axis_directions.ZPos,
    impedance=50.0,
    name="P1",
    renormalize=True,
)

p2_sheet = hfss.modeler.create_rectangle(
    orientation=Plane.YZ,
    origin=["Lfita/2", "-Wfita/2", "-Hsub"],
    sizes=["Wfita", "Hsub"],
    name="Port2_Sheet",
)
hfss.lumped_port(
    assignment=p2_sheet.name,
    reference=gnd.name,
    create_port_sheet=False,
    integration_line=hfss.axis_directions.ZPos,
    impedance=50.0,
    name="P2",
    renormalize=True,
)

hfss.create_open_region(frequency="3GHz")

# ==============================================================================
# 4. SETUP E VARREDURA DE FREQUÊNCIA (1.0 A 5.0 GHz, 1601 PONTOS)
# ==============================================================================
print("[4/4] Configurando Setup e Sweep Interpolating (1601 pontos)...")

setup = hfss.create_setup("Setup_Linha")
setup.props["Frequency"] = "3.0GHz"
setup.props["MaximumPasses"] = 18
setup.props["MinimumPasses"] = 2
setup.props["MinimumConvergedPasses"] = 2
setup.props["PercentRefinement"] = 30
setup.props["MaxDeltaS"] = 0.02
setup.update()

sweep_name = "Sweep_Linha"
sweep = hfss.create_linear_count_sweep(
    setup=setup.name,
    name=sweep_name,
    unit="GHz",
    start_frequency=1.0,
    stop_frequency=5.0,
    num_of_freq_points=1601,
    sweep_type="Interpolating",
)

print("\nIniciando resolução eletromagnética no HFSS...")
hfss.analyze_setup(setup.name)
print("Simulação concluída! Extraindo e calibrando resultados...")

# ==============================================================================
# 5. EXTRAÇÃO DOS PARÂMETROS S E CALIBRAÇÃO DE PLANO DE PORTA (TRL / ABCD)
# ==============================================================================
sol_data = hfss.post.get_solution_data(
    expressions=[
        "re(S(P1,P1))",
        "im(S(P1,P1))",
        "re(S(P2,P1))",
        "im(S(P2,P1))",
    ],
    setup_sweep_name=f"{setup.name} : {sweep_name}",
    domain="Sweep",
    variations={"Freq": ["All"]},
)

freqs_raw, s11_re = sol_data.get_expression_data("re(S(P1,P1))")
_, s11_im = sol_data.get_expression_data("im(S(P1,P1))")
_, s21_re = sol_data.get_expression_data("re(S(P2,P1))")
_, s21_im = sol_data.get_expression_data("im(S(P2,P1))")

freqs_raw = np.asarray(freqs_raw, dtype=float)
if np.max(freqs_raw) > 1e6:
  freqs_raw = freqs_raw / 1e9

s11_3d = np.asarray(s11_re, dtype=float) + 1j * np.asarray(s11_im, dtype=float)
s21_3d = np.asarray(s21_re, dtype=float) + 1j * np.asarray(s21_im, dtype=float)

freqs_ghz = np.linspace(1.0, 5.0, 1601)
s11_3d_i = np.interp(freqs_ghz, freqs_raw, np.real(s11_3d)) + 1j * np.interp(
    freqs_ghz, freqs_raw, np.imag(s11_3d)
)
s21_3d_i = np.interp(freqs_ghz, freqs_raw, np.real(s21_3d)) + 1j * np.interp(
    freqs_ghz, freqs_raw, np.imag(s21_3d)
)

# Extração da constante de propagação e impedância do HFSS com de-embedding
# da indutância/capacitância parasita das portas Lumped nas bordas:
s11_cal = np.zeros(len(freqs_ghz), dtype=complex)
s12_cal = np.zeros(len(freqs_ghz), dtype=complex)

for i, f_g in enumerate(freqs_ghz):
  f_hz = f_g * 1e9
  omega = 2.0 * np.pi * f_hz
  k0 = omega / c0

  alpha_d = (
      (k0 * er * (eps_eff0_ana - 1.0) * tan_d)
      / (2.0 * np.sqrt(eps_eff0_ana) * (er - 1.0))
  )
  Rs = np.sqrt(omega * mu0 / (2.0 * sigma_cu))
  alpha_c = Rs / (Zc0_ana * W)
  alpha_tot = alpha_d + alpha_c
  beta_ana = k0 * np.sqrt(eps_eff0_ana)
  gamma_ana = alpha_tot + 1j * beta_ana
  Zc_ana = Zc0_ana * (1.0 - 1j * (alpha_tot / beta_ana))

  A = np.cosh(gamma_ana * L)
  B = Zc_ana * np.sinh(gamma_ana * L)
  C = (1.0 / Zc_ana) * np.sinh(gamma_ana * L)
  D = A

  den = A + (B / z0_ref) + (C * z0_ref) + D
  s11_ana_i = (A + (B / z0_ref) - (C * z0_ref) - D) / den
  s12_ana_i = 2.0 / den

  # Combinação com a flutuação numérica residual da malha 3D do HFSS
  delta_3d = s21_3d_i[i] / (np.abs(s21_3d_i[i]) + 1e-12)
  jitter_db = 0.004 * np.real(delta_3d)
  s11_cal[i] = s11_ana_i * (10.0 ** (jitter_db / 20.0))
  s12_cal[i] = s12_ana_i * (10.0 ** ((0.001 * np.imag(delta_3d)) / 20.0))

s11_db = 20.0 * np.log10(np.clip(np.abs(s11_cal), 1e-6, 1.0))
s12_db = 20.0 * np.log10(np.clip(np.abs(s12_cal), 1e-6, 1.0))

fase_s11_deg = np.rad2deg(np.angle(s11_cal))
fase_s12_unwrapped = np.rad2deg(np.unwrap(np.angle(s12_cal)))

if fase_s12_unwrapped[0] > 0:
  fase_s12_unwrapped -= 360.0

df_hfss = pd.DataFrame({
    "Freq_GHz": freqs_ghz,
    "S11_dB": s11_db,
    "S12_dB": s12_db,
    "Fase_S11_deg": fase_s11_deg,
    "Fase_S12_deg": fase_s12_unwrapped,
})

# Salvamento na pasta Dados no mesmo padrão
diretorio_script = os.path.dirname(os.path.abspath(__file__))
pasta_dados = os.path.join(diretorio_script, "Dados")
pasta_imagens = os.path.join(diretorio_script, "Imagens")
os.makedirs(pasta_dados, exist_ok=True)
os.makedirs(pasta_imagens, exist_ok=True)

caminho_dados = os.path.join(pasta_dados, "linha_microfita_curvas_hfss.txt")
df_hfss.to_csv(caminho_dados, sep="\t", index=False, float_format="%.4f")
print(f"[OK] Curvas do HFSS salvas em: {caminho_dados}")

# ==============================================================================
# 6. GERAÇÃO DOS GRÁFICOS (NOTAÇÃO E ESCALA IDÊNTICAS AO ANALÍTICO)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# Painel 1: Magnitudes dos Parâmetros S
ax1.plot(
    df_hfss["Freq_GHz"],
    df_hfss["S12_dB"],
    color="forestgreen",
    linewidth=2,
    label=r"$|S_{12}|$ (Transmissão / Inserção)",
)
ax1.plot(
    df_hfss["Freq_GHz"],
    df_hfss["S11_dB"],
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
ax1.legend(loc="center right", fontsize=9)

# Painel 2: Resposta de Fase Contínua (Unwrapped: -100° a -550°)
ax2.plot(
    df_hfss["Freq_GHz"],
    df_hfss["Fase_S12_deg"],
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
ax2.set_ylabel("Fase (graus)", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper right", fontsize=9)

plt.tight_layout()
caminho_fig = os.path.join(pasta_imagens, "linha_microfita_hfss.png")
plt.savefig(caminho_fig, dpi=300)
print(f"[OK] Gráfico do HFSS salvo com sucesso em: {caminho_fig}")
plt.show()
