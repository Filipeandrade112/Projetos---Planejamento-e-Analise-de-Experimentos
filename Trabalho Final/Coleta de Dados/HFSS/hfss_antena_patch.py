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
# 1. PARÂMETROS CALIBRADOS PARA CONCORDÂNCIA COM O MODELO ANALÍTICO (3.000 GHz)
# ==============================================================================
f0_ghz = 3.0  # Frequência de ressonância alvo (3.0 GHz)
er = 4.4  # Permissividade relativa do substrato (FR-4)
h_mm = 1.6  # Altura do dielétrico (mm)
z0_linha = 50.0  # Impedância de referência (50 Ohms)

c0 = 299792458.0
f0 = f0_ghz * 1e9
h = h_mm * 1e-3

# Cálculo de referência do modelo de linha de transmissão (Balanis):
W_ana = (c0 / (2.0 * f0)) * np.sqrt(2.0 / (er + 1.0))
eps_eff_patch = (er + 1.0) / 2.0 + ((er - 1.0) / 2.0) * (
    1.0 + 12.0 * (h / W_ana)
) ** (-0.5)
delta_L = (
    0.412
    * h
    * ((eps_eff_patch + 0.3) * (W_ana / h + 0.264))
    / ((eps_eff_patch - 0.258) * (W_ana / h + 0.8))
)
delta_L_mm = delta_L * 1e3

# Dimensões físicas calibradas para compensar o efeito de onda completa 3D dos
# recortes (notches) e espraiamento no HFSS, cravando f0 = 3.000 GHz e Zin = 50 Ohms:
W_patch = 30.41  # Largura analítica exata (30.41 mm)
L_patch = 23.36  # Comprimento ajustado p/ ressonância 3D exata em 3.000 GHz (analítico: 23.41 mm)
y0_inset = 7.38  # Profundidade do inset para Rin(y0) = 50 Ohms em 3.000 GHz
W_feed = 3.08  # Largura da microfita de 50 Ohms em FR-4 (h = 1.6 mm)
g_notch = 1.0  # Fresta lateral do inset feed (mm)
L_feed = 15.0  # Extensão externa da linha de alimentação até a borda (mm)

# Substrato com porta posicionada exatamente na face da borda
L_sub = round(L_patch + 2.0 * L_feed, 2)  # 53.36 mm
W_sub = round(W_patch + 30.0, 2)  # 60.41 mm

print("\n" + "=" * 70)
print(
    "[1/4] Iniciando HFSS para Antena Patch Calibrada (3.000 GHz / 50 Ohms)..."
)
print(
    f" -> W = {W_patch:.2f} mm | L = {L_patch:.2f} mm | y0 = {y0_inset:.2f} mm"
)
print("=" * 70)

hfss = Hfss(
    non_graphical=False,
    new_desktop=False,
    project="Trabalho_Final_HFSS",
    design="Antena_Patch_3GHz",
    solution_type="DrivenModal",
    remove_lock=True,
)

hfss.modeler.model_units = "mm"

# Variáveis no HFSS
hfss["Wp"] = f"{W_patch}mm"
hfss["Lp"] = f"{L_patch}mm"
hfss["Wf"] = f"{W_feed}mm"
hfss["y0"] = f"{y0_inset}mm"
hfss["g"] = f"{g_notch}mm"
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
# 3. CONSTRUÇÃO DA GEOMETRIA 3D LIMPA E PARAMÉTRICA
# ==============================================================================
print(
    f"Desenhando patch (W={W_patch:.2f}mm, L={L_patch:.2f}mm, "
    f"y0={y0_inset:.2f}mm)..."
)

# 3.1 Substrato FR-4
sub = hfss.modeler.create_box(
    origin=["-L_sub/2", "-W_sub/2", "-Hsub"],
    sizes=["L_sub", "W_sub", "Hsub"],
    name="Substrate",
    material="FR4_epoxy",
)

# 3.2 Plano de Terra (Face inferior z = -Hsub)
gnd = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-L_sub/2", "-W_sub/2", "-Hsub"],
    sizes=["L_sub", "W_sub"],
    name="Ground",
)
hfss.assign_perfecte_to_sheets(gnd.name, name="Ground_PEC")

# 3.3 Patch Principal (Centrado em x=0, y=0)
patch = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-Lp/2", "-Wp/2", 0],
    sizes=["Lp", "Wp"],
    name="Patch_Main",
)

# 3.4 Notches do Inset Feed
notch_top = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-Lp/2", "Wf/2", 0],
    sizes=["y0", "g"],
    name="Notch_Top",
)
notch_bot = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-Lp/2", "-Wf/2 - g", 0],
    sizes=["y0", "g"],
    name="Notch_Bot",
)
hfss.modeler.subtract(
    patch.name, [notch_top.name, notch_bot.name], keep_originals=False
)

# 3.5 Linha de Alimentação da borda até o fundo do inset (L_feed + y0)
feed = hfss.modeler.create_rectangle(
    orientation=Plane.XY,
    origin=["-L_sub/2", "-Wf/2", 0],
    sizes=["L_feed + y0", "Wf"],
    name="Feed_Line",
)

hfss.modeler.unite([patch.name, feed.name])
hfss.assign_perfecte_to_sheets(patch.name, name="Patch_PEC")

# 3.6 Porta Lumped Única na borda externa e Caixa de Radiação
hfss.lumped_port(
    assignment=patch.name,
    reference=gnd.name,
    create_port_sheet=True,
    integration_line=hfss.axis_directions.XNeg,
    impedance=50,
    name="P1",
    renormalize=True,
)

hfss.create_open_region(frequency="3GHz")

# ==============================================================================
# 4. CONFIGURAÇÃO DE ANÁLISE E VARREDURA (2.7 A 3.3 GHz, 1601 PONTOS)
# ==============================================================================
print("[3/4] Configurando Setup Adaptativo e Sweep Interpolating (1601 pontos)...")

setup = hfss.create_setup("Setup_Patch")
setup.props["Frequency"] = "3.0GHz"
setup.props["MaximumPasses"] = 20
setup.props["MinimumPasses"] = 2
setup.props["MinimumConvergedPasses"] = 2
setup.props["PercentRefinement"] = 30
setup.props["MaxDeltaS"] = 0.02
setup.update()

sweep_name = "Sweep_Patch"
sweep = hfss.create_linear_count_sweep(
    setup=setup.name,
    name=sweep_name,
    unit="GHz",
    start_frequency=2.7,
    stop_frequency=3.3,
    num_of_freq_points=1601,
    sweep_type="Interpolating",
)

print("\n[4/4] Iniciando resolução eletromagnética no HFSS...")
hfss.analyze_setup(setup.name)
print("Simulação concluída com sucesso! Extraindo dados...")

# ==============================================================================
# 5. EXTRAÇÃO DOS RESULTADOS E DE-EMBEDDING PARA O PLANO DO PATCH
# ==============================================================================
sol_data = hfss.post.get_solution_data(
    expressions=["dB(S(P1,P1))", "re(Z(P1,P1))", "im(Z(P1,P1))"],
    setup_sweep_name=f"{setup.name} : {sweep_name}",
    domain="Sweep",
    variations={"Freq": ["All"]},
)

freqs, s11_db = sol_data.get_expression_data("dB(S(P1,P1))")
_, re_z_port = sol_data.get_expression_data("re(Z(P1,P1))")
_, im_z_port = sol_data.get_expression_data("im(Z(P1,P1))")

freqs = np.asarray(freqs, dtype=float)
s11_db = np.asarray(s11_db, dtype=float)
re_z_port = np.asarray(re_z_port, dtype=float)
im_z_port = np.asarray(im_z_port, dtype=float)

# De-embedding da linha de alimentação (L_feed + y0 + delta_L) para obter a
# impedância de entrada Z_in diretamente no plano de alimentação do ressonador Patch
# (mesma referência física do modelo analítico RLC paralelo):
u_feed = W_feed / h_mm
eps_eff_feed = (er + 1.0) / 2.0 + ((er - 1.0) / 2.0) * (
    1.0 + 12.0 / u_feed
) ** (-0.5)
d_deembed_m = (L_feed + y0_inset + delta_L_mm) * 1e-3

z_port = re_z_port + 1j * im_z_port
gamma_port = (z_port - z0_linha) / (z_port + z0_linha)
beta_feed = (2.0 * np.pi * (freqs * 1e9) / c0) * np.sqrt(eps_eff_feed)
gamma_patch = gamma_port * np.exp(2j * beta_feed * d_deembed_m)
z_patch = z0_linha * (1.0 + gamma_patch) / (1.0 - gamma_patch)

re_z = np.real(z_patch)
im_z = np.imag(z_patch)

df_patch_hfss = pd.DataFrame({
    "Freq_GHz": freqs,
    "S11_dB": s11_db,
    "Zin_Real": re_z,
    "Zin_Imag": im_z,
})

idx_min = int(df_patch_hfss["S11_dB"].idxmin())
f_res_hfss = df_patch_hfss.loc[idx_min, "Freq_GHz"]
s11_min_hfss = df_patch_hfss.loc[idx_min, "S11_dB"]
zin_re_res = df_patch_hfss.loc[idx_min, "Zin_Real"]
zin_im_res = df_patch_hfss.loc[idx_min, "Zin_Imag"]

print(
    f" -> Ressonância HFSS: f_res = {f_res_hfss:.4f} GHz | "
    f"|S11|_min = {s11_min_hfss:.2f} dB | "
    f"Zin(patch) = {zin_re_res:.1f} + j({zin_im_res:.1f}) Ohms"
)

diretorio_script = os.path.dirname(os.path.abspath(__file__))
pasta_dados = os.path.join(diretorio_script, "Dados")
pasta_imagens = os.path.join(diretorio_script, "Imagens")
os.makedirs(pasta_dados, exist_ok=True)
os.makedirs(pasta_imagens, exist_ok=True)

caminho_dados = os.path.join(pasta_dados, "patch_3ghz_curvas_hfss.txt")
df_patch_hfss.to_csv(caminho_dados, sep="\t", index=False, float_format="%.4f")
print(f"[OK] Curvas da Antena Patch salvas em: {caminho_dados}")

# ==============================================================================
# 6. GERAÇÃO DOS GRÁFICOS
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# Painel 1: S11 do HFSS
ax1.plot(
    df_patch_hfss["Freq_GHz"],
    df_patch_hfss["S11_dB"],
    color="crimson",
    linewidth=2,
    label=r"$|S_{11}|$ HFSS",
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

mask_bw = df_patch_hfss["S11_dB"] <= -10.0
if np.any(mask_bw):
  f_min = df_patch_hfss.loc[mask_bw, "Freq_GHz"].iloc[0]
  f_max = df_patch_hfss.loc[mask_bw, "Freq_GHz"].iloc[-1]
  bw_mhz = (f_max - f_min) * 1e3
  ax1.fill_between(
      df_patch_hfss["Freq_GHz"],
      df_patch_hfss["S11_dB"],
      -10.0,
      where=mask_bw,
      color="pink",
      alpha=0.5,
      label=f"BW (-10 dB) = {bw_mhz:.1f} MHz",
  )

ax1.set_title(
    r"Parâmetro $S_{11}$ da Antena Patch Retangular (HFSS)",
    fontweight="bold",
    fontsize=11,
)
ax1.set_xlabel("Frequência (GHz)", fontsize=10)
ax1.set_ylabel(r"$|S_{11}|$ (dB)", fontsize=10)
ax1.set_xlim(2.7, 3.3)
ax1.set_ylim(-40, 2)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower right", fontsize=9)

# Painel 2: Impedância Zin no Plano do Patch
ax2.plot(
    df_patch_hfss["Freq_GHz"],
    df_patch_hfss["Zin_Real"],
    color="navy",
    linewidth=2,
    label=r"$\text{Re}(Z_{\text{in}})$ (Resistência)",
)
ax2.plot(
    df_patch_hfss["Freq_GHz"],
    df_patch_hfss["Zin_Imag"],
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
    r"Impedância de Entrada $Z_{\text{in}}$ vs. Frequência (HFSS)",
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
caminho_fig = os.path.join(pasta_imagens, "patch_3ghz_hfss.png")
plt.savefig(caminho_fig, dpi=300)
print(f"[OK] Gráfico final da Antena Patch salvo em: {caminho_fig}")
plt.show()
