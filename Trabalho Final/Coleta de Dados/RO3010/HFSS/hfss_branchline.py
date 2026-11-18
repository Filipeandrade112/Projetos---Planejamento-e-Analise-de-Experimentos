import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  from ansys.aedt.core import Hfss
  from ansys.aedt.core.generic.constants import Plane
  from ansys.aedt.core.generic.settings import settings
  settings.use_grpc_api = False
except ImportError:
  from pyaedt import Hfss
  from pyaedt.generic.constants import Plane
  from pyaedt.generic.settings import settings
  settings.use_grpc_api = False

# ==============================================================================
# 1. DIMENSIONAMENTO ANALÍTICO E COMPENSAÇÃO 3D DO BRANCH-LINE (3.0 GHz)
# ==============================================================================
f0_ghz = 3.0  # Frequência central de projeto (3.0 GHz)
z0 = 50.0  # Impedância de referência das portas e ramos shunt (50 Ohms)
z0_p = 50.0  # Impedância dos ramos verticais / shunt (50 Ohms)
z0_s = z0 / np.sqrt(2.0)  # Impedância dos ramos horizontais / série (35.355 Ohms)
er = 10.2  # Permissividade relativa do substrato (Rogers RO3010)
tan_d = 0.0022  # Tangente de perdas do Rogers RO3010
h_mm = 1.6  # Espessura do dielétrico (mm)

c0 = 299792458.0
f0 = f0_ghz * 1e9
h = h_mm * 1e-3


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


w_50, eps_50 = calc_largura_microfita(z0_p, er, h)
w_35, eps_35 = calc_largura_microfita(z0_s, er, h)

L_50_ana_mm = (c0 / (4.0 * f0 * np.sqrt(eps_50))) * 1e3  # 13.70 mm
L_35_ana_mm = (c0 / (4.0 * f0 * np.sqrt(eps_35))) * 1e3  # 13.35 mm

# Larguras das fitas em Rogers RO3010 (mm)
W50 = round(w_50 * 1e3, 2)  # 1.49 mm (linhas de 50 Ohms: ramos verticais e acessos)
W35 = round(w_35 * 1e3, 2)  # 2.66 mm (linhas de 35.36 Ohms: ramos horizontais)

# Comprimentos centro-a-centro idênticos ao modelo analítico em RO3010 (lambda_g / 4)
Lx = round(L_35_ana_mm, 2)  # 9.21 mm (braços horizontais série em X)
Ly = round(L_50_ana_mm, 2)  # 9.48 mm (braços verticais shunt em Y)
L_feed = 8.0  # Comprimento dos acessos externos de 50 Ohms em mm

# Substrato dimensionado exatamente para as portas ficarem nas bordas XNeg e XPos
L_sub = round(Lx + 2.0 * L_feed, 2)  # 26.03 mm
W_sub = round(Ly + W35 + 16.0, 2)  # 29.02 mm

print("\n" + "=" * 70)
print("[1/4] Iniciando HFSS para Acoplador Branch-Line 3.0 GHz (90° Hybrid)...")
print(
    f" -> Ramos Série (35.36 Ohms): W35 = {W35:.2f} mm | "
    f"L_ana = {L_35_ana_mm:.2f} mm | Lx_3D = {Lx:.2f} mm"
)
print(
    f" -> Ramos Shunt (50.00 Ohms): W50 = {W50:.2f} mm | "
    f"L_ana = {L_50_ana_mm:.2f} mm | Ly_3D = {Ly:.2f} mm"
)
print("=" * 70)

hfss = Hfss(
    non_graphical=False,
    new_desktop=False,
    project="Trabalho_Final_HFSS",
    design="Branchline_3GHz",
    solution_type="DrivenModal",
    remove_lock=True,
    version="2024.2",
)

hfss.modeler.model_units = "mm"

# Variáveis parametrizadas no HFSS
hfss["W50"] = f"{W50}mm"
hfss["W35"] = f"{W35}mm"
hfss["Lx"] = f"{Lx}mm"
hfss["Ly"] = f"{Ly}mm"
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
# 3. CONSTRUÇÃO DA GEOMETRIA 3D COM QUINAS INTEGRADAS E PORTAS NAS BORDAS
# ==============================================================================
print("[3/4] Desenhando anel Branch-Line compensado e acessos de 50 Ohms...")

# 3.1 Substrato Rogers RO3010 (er = 10.2, tan_d = 0.0022) e Plano de Terra
mat_rogers = "Rogers_RO3010_custom"
if not hfss.materials.exists_material(mat_rogers):
  mat = hfss.materials.add_material(mat_rogers)
else:
  mat = hfss.materials.material_keys[mat_rogers.lower()]
mat.permittivity = str(er)
mat.dielectric_loss_tangent = str(tan_d)

sub = hfss.modeler.create_box(
    origin=["-L_sub/2", "-W_sub/2", "-Hsub"],
    sizes=["L_sub", "W_sub", "Hsub"],
    name="Substrate",
    material=mat_rogers,
)

gnd = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-L_sub/2", "-W_sub/2", "-Hsub"],
    sizes=["L_sub", "W_sub"],
    name="Ground",
)
hfss.assign_perfecte_to_sheets(gnd.name, name="Ground_PEC")

# 3.2 Braços Horizontais Série (35.36 Ohms) cobrindo integralmente as quinas
arm_top = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-Lx/2 - W50/2", "Ly/2 - W35/2", 0],
    sizes=["Lx + W50", "W35"],
    name="Arm_Top",
)
arm_bot = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-Lx/2 - W50/2", "-Ly/2 - W35/2", 0],
    sizes=["Lx + W50", "W35"],
    name="Arm_Bottom",
)

# 3.3 Braços Verticais Shunt (50.0 Ohms)
arm_left = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-Lx/2 - W50/2", "-Ly/2 - W35/2", 0],
    sizes=["W50", "Ly + W35"],
    name="Arm_Left",
)
arm_right = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["Lx/2 - W50/2", "-Ly/2 - W35/2", 0],
    sizes=["W50", "Ly + W35"],
    name="Arm_Right",
)

# 3.4 Quatro Linhas de Acesso de 50 Ohms centradas em y = +/- Ly/2
feed1 = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-L_sub/2", "Ly/2 - W50/2", 0],
    sizes=["L_feed", "W50"],
    name="Feed_P1",
)
feed4 = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-L_sub/2", "-Ly/2 - W50/2", 0],
    sizes=["L_feed", "W50"],
    name="Feed_P4",
)
feed2 = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["Lx/2", "Ly/2 - W50/2", 0],
    sizes=["L_feed", "W50"],
    name="Feed_P2",
)
feed3 = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["Lx/2", "-Ly/2 - W50/2", 0],
    sizes=["L_feed", "W50"],
    name="Feed_P3",
)

todas_tiras = [
    arm_top.name,
    arm_bot.name,
    arm_left.name,
    arm_right.name,
    feed1.name,
    feed2.name,
    feed3.name,
    feed4.name,
]
hfss.assign_perfecte_to_sheets(todas_tiras, name="Branchline_PEC")

# 3.5 Portas Lumped P1, P2, P3 e P4 nas bordas externas XNeg e XPos
hfss.lumped_port(
    assignment=feed1.name,
    reference=gnd.name,
    create_port_sheet=True,
    integration_line=hfss.axis_directions.XNeg,
    impedance=50,
    name="P1",
    renormalize=True,
)

hfss.lumped_port(
    assignment=feed2.name,
    reference=gnd.name,
    create_port_sheet=True,
    integration_line=hfss.axis_directions.XPos,
    impedance=50,
    name="P2",
    renormalize=True,
)

hfss.lumped_port(
    assignment=feed3.name,
    reference=gnd.name,
    create_port_sheet=True,
    integration_line=hfss.axis_directions.XPos,
    impedance=50,
    name="P3",
    renormalize=True,
)

hfss.lumped_port(
    assignment=feed4.name,
    reference=gnd.name,
    create_port_sheet=True,
    integration_line=hfss.axis_directions.XNeg,
    impedance=50,
    name="P4",
    renormalize=True,
)

hfss.create_open_region(frequency="3GHz")

# ==============================================================================
# 4. SETUP E VARREDURA DE FREQUÊNCIA (2.0 A 4.0 GHz, 1601 PONTOS)
# ==============================================================================
print("[4/4] Configurando Setup Adaptativo e Sweep Interpolating (1601 pontos)...")

setup = hfss.create_setup("Setup_Branchline")
setup.props["Frequency"] = "3.0GHz"
setup.props["MaximumPasses"] = 18
setup.props["MinimumPasses"] = 2
setup.props["MinimumConvergedPasses"] = 2
setup.props["PercentRefinement"] = 30
setup.props["MaxDeltaS"] = 0.02
setup.update()

sweep_name = "Sweep_Branchline"
sweep = hfss.create_linear_count_sweep(
    setup=setup.name,
    name=sweep_name,
    unit="GHz",
    start_frequency=2.0,
    stop_frequency=4.0,
    num_of_freq_points=1601,
    sweep_type="Interpolating",
)

print("\nIniciando simulação eletromagnética no HFSS (Branch-Line 3.0 GHz)...")
hfss.analyze_setup(setup.name)
print("Simulação concluída com sucesso! Extraindo parâmetros S e defasagem...")

# ==============================================================================
# 5. EXTRAÇÃO DOS RESULTADOS E DE-EMBEDDING MODAL PARA OS NÓS DO ACOPLADOR
# ==============================================================================
expressoes = [
    "dB(S(P1,P1))",
    "dB(S(P2,P1))",
    "dB(S(P3,P1))",
    "dB(S(P4,P1))",
    "cang_deg(S(P2,P1))",
    "cang_deg(S(P3,P1))",
    "re(S(P1,P1))",
    "im(S(P1,P1))",
    "re(S(P2,P1))",
    "im(S(P2,P1))",
    "re(S(P3,P1))",
    "im(S(P3,P1))",
    "re(S(P4,P1))",
    "im(S(P4,P1))",
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
_, re_s21 = sol_data.get_expression_data("re(S(P2,P1))")
_, im_s21 = sol_data.get_expression_data("im(S(P2,P1))")
_, re_s31 = sol_data.get_expression_data("re(S(P3,P1))")
_, im_s31 = sol_data.get_expression_data("im(S(P3,P1))")
_, re_s41 = sol_data.get_expression_data("re(S(P4,P1))")
_, im_s41 = sol_data.get_expression_data("im(S(P4,P1))")

freqs = np.asarray(freqs, dtype=float)
s21_db_raw = np.asarray(s21_db_raw, dtype=float)
s31_db_raw = np.asarray(s31_db_raw, dtype=float)
fase21_raw = np.asarray(fase21_raw, dtype=float)
fase31_raw = np.asarray(fase31_raw, dtype=float)

s11_port = np.asarray(re_s11, dtype=float) + 1j * np.asarray(im_s11, dtype=float)
s21_port = np.asarray(re_s21, dtype=float) + 1j * np.asarray(im_s21, dtype=float)
s31_port = np.asarray(re_s31, dtype=float) + 1j * np.asarray(im_s31, dtype=float)
s41_port = np.asarray(re_s41, dtype=float) + 1j * np.asarray(im_s41, dtype=float)

# 5.1 De-embedding físico puro das linhas de acesso de 50 Ohms (2 * L_feed)
idx_f0 = int(np.argmin(np.abs(freqs - f0_ghz)))
beta_feed = (2.0 * np.pi * (freqs * 1e9) / c0) * np.sqrt(eps_50)
d_feed_m = L_feed * 1e-3
rot_feed = np.exp(2j * beta_feed * d_feed_m)

s11_deemb = s11_port * rot_feed
s21_deemb = s21_port * rot_feed
s31_deemb = s31_port * rot_feed
s41_deemb = s41_port * rot_feed

s11_db = 20.0 * np.log10(np.clip(np.abs(s11_deemb), 1e-6, 1.0))
s21_db = 20.0 * np.log10(np.clip(np.abs(s21_deemb), 1e-6, 1.0))
s31_db = 20.0 * np.log10(np.clip(np.abs(s31_deemb), 1e-6, 1.0))
s41_db = 20.0 * np.log10(np.clip(np.abs(s41_deemb), 1e-6, 1.0))

fase_s21_deg = np.angle(s21_deemb, deg=True)
fase_s31_deg = np.angle(s31_deemb, deg=True)
delta_fase = (fase_s21_deg - fase_s31_deg) % 360.0

df_bl_hfss = pd.DataFrame({
    "Freq_GHz": freqs,
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
# 6. SALVAMENTO DOS DADOS E GERAÇÃO DOS GRÁFICOS
# ==============================================================================
diretorio_script = os.path.dirname(os.path.abspath(__file__))
pasta_raiz = os.path.dirname(diretorio_script)
pasta_dados = os.path.join(pasta_raiz, "Dados")
pasta_imagens = os.path.join(pasta_raiz, "Imagens")
os.makedirs(pasta_dados, exist_ok=True)
os.makedirs(pasta_imagens, exist_ok=True)

caminho_dados = os.path.join(pasta_dados, "branchline_3ghz_curvas_hfss.txt")
df_bl_hfss.to_csv(caminho_dados, sep="\t", index=False, float_format="%.4f")
print(f"[OK] Curvas do Branchline (HFSS) salvas em: {caminho_dados}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# Painel 1: Magnitudes dos Parâmetros S
ax1.plot(
    df_bl_hfss["Freq_GHz"],
    df_bl_hfss["S21_dB"],
    color="forestgreen",
    linewidth=2,
    label=r"$|S_{21}|$ (Transmissão / Through)",
)
ax1.plot(
    df_bl_hfss["Freq_GHz"],
    df_bl_hfss["S31_dB"],
    color="darkorange",
    linewidth=2,
    label=r"$|S_{31}|$ (Acoplada / Coupled)",
)
ax1.plot(
    df_bl_hfss["Freq_GHz"],
    df_bl_hfss["S11_dB"],
    color="crimson",
    linewidth=1.8,
    linestyle="--",
    label=r"$|S_{11}|$ (Reflexão / Retorno)",
)
ax1.plot(
    df_bl_hfss["Freq_GHz"],
    df_bl_hfss["S41_dB"],
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
    r"Parâmetros $S$ do Acoplador Branchline 3 GHz (HFSS)",
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
    df_bl_hfss["Freq_GHz"],
    df_bl_hfss["Delta_Fase_deg"],
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
    r"Diferença de Fase entre as Saídas ($\Delta\phi$ no HFSS)",
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
caminho_fig = os.path.join(pasta_imagens, "branchline_3ghz_hfss.png")
plt.savefig(caminho_fig, dpi=300)
print(f"[OK] Gráfico do Branchline salvo em: {caminho_fig}")
plt.show()
