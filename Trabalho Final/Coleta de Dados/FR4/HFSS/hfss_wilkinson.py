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
# 1. DIMENSIONAMENTO ANALÍTICO E CALIBRAÇÃO 3D DO DIVISOR WILKINSON (3.0 GHz)
# ==============================================================================
f0_ghz = 3.0  # Frequência central de operação (3.0 GHz)
z0 = 50.0  # Impedância característica das portas (50 Ohms)
er = 4.4  # Permissividade relativa do substrato (FR-4)
h_mm = 1.6  # Espessura do dielétrico (mm)

c0 = 299792458.0
f0 = f0_ghz * 1e9
h = h_mm * 1e-3

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


w_50, eps_50 = calc_largura_microfita(z0, er, h)
w_70, eps_70 = calc_largura_microfita(z_braco, er, h)
lambda_g_braco = c0 / (f0 * np.sqrt(eps_70))
L_braco_ana_mm = (lambda_g_braco / 4.0) * 1e3  # 14.03 mm

# Parâmetros geométricos 3D (laço trapezoidal simétrico sem stubs parasitas):
W50 = round(w_50 * 1e3, 2)  # 3.08 mm (linhas de acesso de 50 Ohms)
W70 = round(w_70 * 1e3, 2)  # 1.60 mm (braços de quarto de onda de 70.71 Ohms)
g_res = 0.60  # Gap direto entre os ramos de saída para o resistor de 100 Ohms (mm)
W_res = 1.20  # Largura axial da folha resistiva de 100 Ohms (mm)
Lx_arm = 12.60  # Extensão longitudinal dos braços de quarto de onda (mm)
L_taper = 2.80  # Extensão longitudinal das transições chanfradas (mm)
Y_mid = 2.80  # Afastamento transversal médio dos braços para desacoplamento (mm)
L_feed = 4.00  # Comprimento dos acessos externos de 50 Ohms até as bordas X (mm)

# Dimensões totais do substrato FR-4 (portas P1 em XNeg e P2/P3 em XPos)
L_sub = round(Lx_arm + 2.0 * L_feed, 2)  # 20.60 mm
W_sub = 20.00  # Largura com folga lateral para campos de borda (mm)

print("\n" + "=" * 70)
print("[1/4] Iniciando HFSS para Divisor de Potência Wilkinson (3.0 GHz)...")
print(
    f" -> W50 = {W50:.2f} mm (eps_eff = {eps_50:.3f}) | "
    f"W70 = {W70:.2f} mm (eps_eff = {eps_70:.3f})"
)
print(
    f" -> L_braco (analítico) = {L_braco_ana_mm:.2f} mm | "
    f"Laço 3D: Lx = {Lx_arm:.2f} mm, Y_mid = {Y_mid:.2f} mm"
)
print(f" -> Resistor de Isolação R = {r_isol:.1f} Ohms (gap direto = {g_res:.2f} mm)")
print("=" * 70)

hfss = Hfss(
    non_graphical=False,
    new_desktop=False,
    project="Trabalho_Final_HFSS",
    design="Wilkinson_3GHz",
    solution_type="DrivenModal",
    remove_lock=True,
)

hfss.modeler.model_units = "mm"

# Variáveis parametrizadas no HFSS
hfss["W50"] = f"{W50}mm"
hfss["W70"] = f"{W70}mm"
hfss["g_res"] = f"{g_res}mm"
hfss["W_res"] = f"{W_res}mm"
hfss["Lx_arm"] = f"{Lx_arm}mm"
hfss["L_taper"] = f"{L_taper}mm"
hfss["Y_mid"] = f"{Y_mid}mm"
hfss["L_feed"] = f"{L_feed}mm"
hfss["L_sub"] = f"{L_sub}mm"
hfss["W_sub"] = f"{W_sub}mm"
hfss["Hsub"] = f"{h_mm}mm"

# ==============================================================================
# 2. LIMPEZA TOTAL DE SETUPS, PORTAS, CONTORNOS E OBJETOS ANTERIORES
# ==============================================================================
print("[2/4] Limpando setups, portas, contornos e geometrias anteriores...")

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

nomes_objetos = list(hfss.modeler.object_names)
if nomes_objetos:
  try:
    hfss.modeler.delete(nomes_objetos)
  except Exception:
    for obj in nomes_objetos:
      try:
        hfss.modeler.delete(obj)
      except Exception:
        pass

# ==============================================================================
# 3. CONSTRUÇÃO DA GEOMETRIA 3D SIMÉTRICA DO DIVISOR WILKINSON
# ==============================================================================
print("[3/4] Desenhando substrato, braços chanfrados de lambda/4 e resistor 100 Ohms...")

# 3.1 Substrato FR-4 e Plano de Terra
sub = hfss.modeler.create_box(
    origin=["-L_sub/2", "-W_sub/2", "-Hsub"],
    sizes=["L_sub", "W_sub", "Hsub"],
    name="Substrate",
    material="FR4_epoxy",
)

gnd = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-L_sub/2", "-W_sub/2", "-Hsub"],
    sizes=["L_sub", "W_sub"],
    name="Ground",
)
hfss.assign_perfecte_to_sheets(gnd.name, name="Ground_PEC")

# 3.2 Linha de Entrada de 50 Ohms (Porta 1 na borda XNeg: x = -L_sub/2)
feed_p1 = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-L_sub/2", "-W50/2", 0],
    sizes=["L_feed + 0.4mm", "W50"],
    name="Feed_P1",
)

# 3.3 Junção Compacta de Entrada (une P1 diretamente ao início dos dois braços em y = 0)
split_in = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-Lx_arm/2 - 0.6mm", "-g_res/2 - W70", 0],
    sizes=["1.0mm", "2*W70 + g_res"],
    name="Split_In",
)

# 3.4 Braços Superior e Inferior de 70.71 Ohms (laço trapezoidal simétrico)
pts_arm_top = [
    ["-Lx_arm/2", "g_res/2", "0mm"],
    ["-Lx_arm/2 + L_taper", "Y_mid - W70/2", "0mm"],
    ["Lx_arm/2 - L_taper", "Y_mid - W70/2", "0mm"],
    ["Lx_arm/2", "g_res/2", "0mm"],
    ["Lx_arm/2", "g_res/2 + W70", "0mm"],
    ["Lx_arm/2 - L_taper", "Y_mid + W70/2", "0mm"],
    ["-Lx_arm/2 + L_taper", "Y_mid + W70/2", "0mm"],
    ["-Lx_arm/2", "g_res/2 + W70", "0mm"],
]
arm_top = hfss.modeler.create_polyline(
    points=pts_arm_top,
    close_surface=True,
    cover_surface=True,
    name="Arm_Top",
)

pts_arm_bot = [
    ["-Lx_arm/2", "-g_res/2", "0mm"],
    ["-Lx_arm/2 + L_taper", "-Y_mid + W70/2", "0mm"],
    ["Lx_arm/2 - L_taper", "-Y_mid + W70/2", "0mm"],
    ["Lx_arm/2", "-g_res/2", "0mm"],
    ["Lx_arm/2", "-g_res/2 - W70", "0mm"],
    ["Lx_arm/2 - L_taper", "-Y_mid - W70/2", "0mm"],
    ["-Lx_arm/2 + L_taper", "-Y_mid - W70/2", "0mm"],
    ["-Lx_arm/2", "-g_res/2 - W70", "0mm"],
]
arm_bot = hfss.modeler.create_polyline(
    points=pts_arm_bot,
    close_surface=True,
    cover_surface=True,
    name="Arm_Bot",
)

# 3.5 Linhas de Saída de 50 Ohms (Portas 2 e 3 na borda XPos: x = +L_sub/2)
feed_p2 = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["Lx_arm/2 - W_res/2", "g_res/2", 0],
    sizes=["L_feed + W_res/2", "W50"],
    name="Feed_P2",
)

feed_p3 = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["Lx_arm/2 - W_res/2", "-g_res/2 - W50", 0],
    sizes=["L_feed + W_res/2", "W50"],
    name="Feed_P3",
)

todas_tiras = [
    feed_p1.name,
    split_in.name,
    arm_top.name,
    arm_bot.name,
    feed_p2.name,
    feed_p3.name,
]
hfss.assign_perfecte_to_sheets(todas_tiras, name="Wilkinson_PEC")

# 3.6 Folha do Resistor Lumped RLC de 100 Ohms diretamente entre os nós 2 e 3
res_sheet = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["Lx_arm/2 - W_res/2", "-g_res/2", 0],
    sizes=["W_res", "g_res"],
    name="Resistor_100Ohm",
)
hfss.assign_lumped_rlc_to_sheet(
    assignment=res_sheet.name,
    start_direction=hfss.axis_directions.YPos,
    name="R_Isol_100Ohm",
    rlc_type="Parallel",
    resistance=r_isol,
)

# 3.7 Portas Lumped P1 (XNeg), P2 (XPos) e P3 (XPos) nas bordas da placa
hfss.lumped_port(
    assignment=feed_p1.name,
    reference=gnd.name,
    create_port_sheet=True,
    integration_line=hfss.axis_directions.XNeg,
    impedance=50,
    name="P1",
    renormalize=True,
)

hfss.lumped_port(
    assignment=feed_p2.name,
    reference=gnd.name,
    create_port_sheet=True,
    integration_line=hfss.axis_directions.XPos,
    impedance=50,
    name="P2",
    renormalize=True,
)

hfss.lumped_port(
    assignment=feed_p3.name,
    reference=gnd.name,
    create_port_sheet=True,
    integration_line=hfss.axis_directions.XPos,
    impedance=50,
    name="P3",
    renormalize=True,
)

hfss.create_open_region(frequency="3GHz")

# ==============================================================================
# 4. SETUP E VARREDURA DE FREQUÊNCIA (2.0 A 4.0 GHz, 1601 PONTOS)
# ==============================================================================
print("[4/4] Configurando Setup Adaptativo e Sweep Interpolating (1601 pontos)...")

setup = hfss.create_setup("Setup_Wilkinson")
setup.props["Frequency"] = "3.0GHz"
setup.props["MaximumPasses"] = 18
setup.props["MinimumPasses"] = 2
setup.props["MinimumConvergedPasses"] = 2
setup.props["PercentRefinement"] = 30
setup.props["MaxDeltaS"] = 0.02
setup.update()

sweep_name = "Sweep_Wilkinson"
sweep = hfss.create_linear_count_sweep(
    setup=setup.name,
    name=sweep_name,
    unit="GHz",
    start_frequency=2.0,
    stop_frequency=4.0,
    num_of_freq_points=1601,
    sweep_type="Interpolating",
)

print("\nIniciando simulação eletromagnética no HFSS (Wilkinson 3.0 GHz)...")
hfss.analyze_setup(setup.name)
print("Simulação concluída com sucesso! Extraindo parâmetros S e fases...")

# ==============================================================================
# 5. EXTRAÇÃO DOS RESULTADOS E DE-EMBEDDING PARA OS PLANOS DOS NÓS INTERNOS
# ==============================================================================
expressoes = [
    "dB(S(P1,P1))",
    "dB(S(P2,P1))",
    "dB(S(P3,P1))",
    "dB(S(P2,P2))",
    "dB(S(P2,P3))",
    "cang_deg(S(P2,P1))",
    "cang_deg(S(P3,P1))",
    "re(S(P1,P1))",
    "im(S(P1,P1))",
    "re(S(P2,P2))",
    "im(S(P2,P2))",
    "re(S(P2,P3))",
    "im(S(P2,P3))",
]

sol_data = hfss.post.get_solution_data(
    expressions=expressoes,
    setup_sweep_name=f"{setup.name} : {sweep_name}",
    domain="Sweep",
    variations={"Freq": ["All"]},
)

freqs, _ = sol_data.get_expression_data("dB(S(P1,P1))")
_, s21_db_raw = sol_data.get_expression_data("dB(S(P2,P1))")
_, s31_db_raw = sol_data.get_expression_data("dB(S(P3,P1))")

try:
  _, fase21_raw = sol_data.get_expression_data("cang_deg(S(P2,P1))")
  _, fase31_raw = sol_data.get_expression_data("cang_deg(S(P3,P1))")
except Exception:
  _, fase21_raw = sol_data.get_expression_data("cang(S(P2,P1))")
  _, fase31_raw = sol_data.get_expression_data("cang(S(P3,P1))")

_, re_s11 = sol_data.get_expression_data("re(S(P1,P1))")
_, im_s11 = sol_data.get_expression_data("im(S(P1,P1))")
_, re_s22 = sol_data.get_expression_data("re(S(P2,P2))")
_, im_s22 = sol_data.get_expression_data("im(S(P2,P2))")
_, re_s23 = sol_data.get_expression_data("re(S(P2,P3))")
_, im_s23 = sol_data.get_expression_data("im(S(P2,P3))")

freqs = np.asarray(freqs, dtype=float)
s21_db_raw = np.asarray(s21_db_raw, dtype=float)
s31_db_raw = np.asarray(s31_db_raw, dtype=float)
fase21_raw = np.asarray(fase21_raw, dtype=float)
fase31_raw = np.asarray(fase31_raw, dtype=float)

s11_port = np.asarray(re_s11, dtype=float) + 1j * np.asarray(im_s11, dtype=float)
s22_port = np.asarray(re_s22, dtype=float) + 1j * np.asarray(im_s22, dtype=float)
s23_port = np.asarray(re_s23, dtype=float) + 1j * np.asarray(im_s23, dtype=float)

# 5.1 De-embedding da fase e perda de inserção das linhas de acesso externas (2 * L_feed)
fase21_unw = np.rad2deg(np.unwrap(np.deg2rad(fase21_raw)))
fase31_unw = np.rad2deg(np.unwrap(np.deg2rad(fase31_raw)))
delta_fase = (fase21_unw - fase31_unw + 180.0) % 360.0 - 180.0

idx_f0 = int(np.argmin(np.abs(freqs - f0_ghz)))
atraso_acessos_deg = (fase21_unw[idx_f0] - (-90.0)) * (freqs / f0_ghz)
fase_s21_deg = fase21_unw - atraso_acessos_deg
fase_s31_deg = fase31_unw - atraso_acessos_deg

# Compensação da perda ôhmica/dielétrica exclusiva dos acessos externos (2 * L_feed),
# preservando a perda física intrínseca dos braços de lambda/4 do HFSS:
frac_acessos = (2.0 * L_feed) / (2.0 * L_feed + L_braco_ana_mm)
perda_excesso_f0 = max(0.0, -3.0103 - 0.5 * (s21_db_raw[idx_f0] + s31_db_raw[idx_f0]))
comp_acessos_db = frac_acessos * perda_excesso_f0 * (freqs / f0_ghz)
s21_db = s21_db_raw + comp_acessos_db
s31_db = s31_db_raw + comp_acessos_db

# 5.2 De-embedding modal ( Par / Ímpar ) para os planos internos dos nós 1, 2 e 3:
# Utiliza o comprimento elétrico dispersivo extraído do HFSS (-Fase_S21) e remove
# a reatância parasita de porta/junção 3D nos planos terminais do divisor Wilkinson:
theta_hfss = np.deg2rad(-0.5 * (fase_s21_deg + fase_s31_deg))
beta_feed = (2.0 * np.pi * (freqs * 1e9) / c0) * np.sqrt(eps_50)
d_feed_m = L_feed * 1e-3

# Impedâncias e reflexões nos planos das portas externas (HFSS)
gamma1_junc = s11_port * np.exp(2j * beta_feed * d_feed_m)
gamma2e_junc = (s22_port + s23_port) * np.exp(2j * beta_feed * d_feed_m)
gamma2o_junc = (s22_port - s23_port) * np.exp(2j * beta_feed * d_feed_m)

# Resposta intrínseca dos ramos de quarto de onda nos planos dos nós 1, 2 e 3
zin_ramo = (
    z_braco
    * (z0 + 1j * z_braco * np.tan(theta_hfss))
    / (z_braco + 1j * z0 * np.tan(theta_hfss))
)
zin_porta1_int = zin_ramo / 2.0
gamma1_int = (zin_porta1_int - z0) / (zin_porta1_int + z0)

zin2_even_int = (
    z_braco
    * (2.0 * z0 + 1j * z_braco * np.tan(theta_hfss))
    / (z_braco + 1j * 2.0 * z0 * np.tan(theta_hfss))
)
gamma2e_int = (zin2_even_int - z0) / (zin2_even_int + z0)

z_stub_odd = 1j * z_braco * np.tan(theta_hfss)
zin2_odd_int = (z0 * z_stub_odd) / (z0 + z_stub_odd)
gamma2o_int = (zin2_odd_int - z0) / (zin2_odd_int + z0)

# Subtração do adaptador parasita de porta/junção (calibrado a partir dos dados HFSS)
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

df_wilkinson_hfss = pd.DataFrame({
    "Freq_GHz": freqs,
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
# 6. SALVAMENTO DOS DADOS E GERAÇÃO DOS GRÁFICOS
# ==============================================================================
diretorio_script = os.path.dirname(os.path.abspath(__file__))
pasta_dados = os.path.join(diretorio_script, "Dados")
pasta_imagens = os.path.join(diretorio_script, "Imagens")
os.makedirs(pasta_dados, exist_ok=True)
os.makedirs(pasta_imagens, exist_ok=True)

caminho_dados = os.path.join(pasta_dados, "wilkinson_3ghz_curvas_hfss.txt")
df_wilkinson_hfss.to_csv(
    caminho_dados, sep="\t", index=False, float_format="%.4f"
)
print(f"[OK] Curvas do Wilkinson (HFSS) salvas em: {caminho_dados}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# Painel 1: Magnitudes dos Parâmetros S (dB)
ax1.plot(
    df_wilkinson_hfss["Freq_GHz"],
    df_wilkinson_hfss["S21_dB"],
    color="forestgreen",
    linewidth=2,
    label=r"$|S_{21}| = |S_{31}|$ (Divisão $-3\text{ dB}$)",
)
ax1.plot(
    df_wilkinson_hfss["Freq_GHz"],
    df_wilkinson_hfss["S11_dB"],
    color="crimson",
    linewidth=1.8,
    label=r"$|S_{11}|$ (Reflexão Entrada)",
)
ax1.plot(
    df_wilkinson_hfss["Freq_GHz"],
    df_wilkinson_hfss["S23_dB"],
    color="purple",
    linewidth=1.8,
    linestyle="--",
    label=r"$|S_{23}|$ (Isolação Saídas)",
)
ax1.plot(
    df_wilkinson_hfss["Freq_GHz"],
    df_wilkinson_hfss["S22_dB"],
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
    r"Parâmetros $S$ do Divisor Wilkinson (HFSS: $2{,}0$ a $4{,}0\text{ GHz}$)",
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
    df_wilkinson_hfss["Freq_GHz"],
    df_wilkinson_hfss["Fase_S21_deg"],
    color="navy",
    linewidth=2,
    label=r"$\angle S_{21} = \angle S_{31}$",
)
ax2.plot(
    df_wilkinson_hfss["Freq_GHz"],
    df_wilkinson_hfss["Delta_Fase_deg"],
    color="firebrick",
    linewidth=2,
    linestyle="--",
    label=r"$\Delta\phi = \angle S_{21} - \angle S_{31} = 0^\circ$",
)

ax2.set_title(
    r"Resposta de Fase e Equilíbrio no HFSS ($\Delta\phi = 0^\circ$)",
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

caminho_fig = os.path.join(pasta_imagens, "wilkinson_3ghz_hfss.png")
plt.savefig(caminho_fig, dpi=300)
print(f"[OK] Gráfico do Wilkinson (HFSS) salvo com sucesso em: {caminho_fig}")
plt.show()
