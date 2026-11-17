import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  from ansys.aedt.core import Hfss
except ImportError:
  from pyaedt import Hfss

# ==============================================================================
# 1. PARÂMETROS DA CAVIDADE RESSONANTE (mm)
# ==============================================================================
a_mm = 50.0  # Largura (x)
b_mm = 25.0  # Altura (y)
d_mm = 70.0  # Comprimento (z)
n_modos = 20  # Limite máximo de modos do HFSS Eigenmode (máx. 20 no Ansys HFSS)

print("\n" + "=" * 70)
print(f"[1/4] Iniciando HFSS para Cavidade Ressonante ({n_modos} modos)...")
print("=" * 70)

hfss = Hfss(
    non_graphical=False,
    new_desktop=False,
    project="Trabalho_Final_HFSS",
    design="Cavidade_3D_Eigenmode",
    solution_type="Eigenmode",
    remove_lock=True,
)

hfss.modeler.model_units = "mm"
hfss["a_dim"] = f"{a_mm}mm"
hfss["b_dim"] = f"{b_mm}mm"
hfss["d_dim"] = f"{d_mm}mm"

# ==============================================================================
# 2. GEOMETRIA 3D DA CAVIDADE
# ==============================================================================
print("[2/4] Verificando e criando geometria da cavidade...")
if "Cavidade_Ar" in hfss.modeler.object_names:
  hfss.modeler.delete("Cavidade_Ar")

cavidade = hfss.modeler.create_box(
    origin=[0, 0, 0],
    sizes=["a_dim", "b_dim", "d_dim"],
    name="Cavidade_Ar",
    material="vacuum",
)
hfss.assign_perfecte_to_sheets(cavidade.faces, name="Paredes_PEC")

# ==============================================================================
# 3. SETUP EIGENMODE
# ==============================================================================
print(f"[3/4] Configurando Setup Eigenmode com {n_modos} modos...")
if "Setup_Eigen" in hfss.setup_names:
  setup = hfss.get_setup("Setup_Eigen")
else:
  setup = hfss.create_setup("Setup_Eigen")

setup.props["MinFreq"] = "1.0GHz"
setup.props["NumModes"] = n_modos
setup.props["ConvergeOnNumModes"] = n_modos
setup.props["MaximumPasses"] = 15
setup.props["MinimumConvergedPasses"] = 2
setup.props["PercentRefinement"] = 30
setup.update()

# ==============================================================================
# 4. SIMULAÇÃO E EXTRAÇÃO DIRETA DOS AUTOVALORES
# ==============================================================================
print(f"[4/4] Executando simulação modal no HFSS ({n_modos} modos)...")
hfss.analyze_setup(setup.name)
print("Simulação concluída! Extraindo autovalores via API nativa...")

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(diretorio_atual).lower() in [
    "analiticos",
    "hfss",
    "tidy3d",
]:
  pasta_raiz = os.path.dirname(diretorio_atual)
else:
  pasta_raiz = diretorio_atual

pasta_dados = os.path.join(pasta_raiz, "Dados")
pasta_imagens = os.path.join(pasta_raiz, "Imagens")
os.makedirs(pasta_dados, exist_ok=True)
os.makedirs(pasta_imagens, exist_ok=True)
caminho_raw = os.path.join(pasta_dados, "eigenmodes_hfss_raw.txt")

# Chamada nativa da API do HFSS para exportar a tabela de modos próprios
soln_name = f"{setup.name} : LastAdaptive"
sucesso_export = False

try:
  oSolutions = hfss.odesign.GetModule("Solutions")
  oSolutions.ExportEigenmodes(soln_name, "", caminho_raw)
  sucesso_export = True
except Exception as e:
  print("Tentando método alternativo de exportação:", e)
  try:
    hfss.export_convergence(setup.name, file_path=caminho_raw)
    sucesso_export = True
  except Exception:
    pass

# Leitura do arquivo exportado pelo HFSS
lista_modos = []
if sucesso_export and os.path.exists(caminho_raw):
  with open(caminho_raw, "r", encoding="utf-8", errors="ignore") as f:
    linhas = f.readlines()

  for linha in linhas:
    partes = linha.replace('"', "").replace(",", "\t").strip().split()
    if len(partes) >= 2:
      try:
        ordem = int(partes[0])
        freq_val = float(partes[1])
        if freq_val > 1e6:
          freq_val /= 1e9
        q_val = float(partes[2]) if len(partes) > 2 else 0.0
        lista_modos.append({
            "Ordem": ordem,
            "Freq_GHz_HFSS": freq_val,
            "Q_Factor_HFSS": q_val,
        })
      except ValueError:
        continue

# Caso o arquivo raw não tenha lido por restrição de permissão, fallback via SolutionData
if not lista_modos:
  print("Processando via expressões modais individuais...")
  expr_modos = [f"re(Mode({i}))" for i in range(1, n_modos + 1)]
  try:
    sol_data = hfss.post.get_solution_data(
        expressions=expr_modos,
        setup_sweep_name=soln_name,
        report_category="Eigenmode Parameters",
    )
    for i in range(1, n_modos + 1):
      exp = f"re(Mode({i}))"
      val = sol_data.get_expression_data(exp)[1][0]
      f_ghz = val / 1e9 if val > 1e6 else val
      lista_modos.append(
          {"Ordem": i, "Freq_GHz_HFSS": f_ghz, "Q_Factor_HFSS": 0.0}
      )
  except Exception as e:
    print("Aviso ao extrair expressões:", e)

# ==============================================================================
# 5. CRUZAMENTO COM OS DADOS ANALÍTICOS, SALVAMENTO E GRÁFICO DO ESPECTRO
# ==============================================================================
df_final = pd.DataFrame(lista_modos)

if not df_final.empty:
  # Ordenar por ordem dos modos
  df_final = df_final.sort_values(by="Ordem").reset_index(drop=True)

  # Cruzamento com o arquivo analítico cavidade_modos_30.txt gerado anteriormente
  caminho_analitico = os.path.join(pasta_dados, "cavidade_modos_30.txt")
  if os.path.exists(caminho_analitico):
    try:
      df_ana = pd.read_csv(caminho_analitico, sep=r"\s+|,", engine="python")
      col_f_ana = [c for c in df_ana.columns if "freq" in c.lower()][0]
      col_m_ana = [c for c in df_ana.columns if "modo" in c.lower()][0]

      n_comum = min(len(df_final), len(df_ana))
      df_final["Modo_Teorico"] = df_ana[col_m_ana].iloc[:n_comum].values
      df_final["Freq_GHz_Analitica"] = df_ana[col_f_ana].iloc[:n_comum].values

      # Erro percentual relativo: |HFSS - Analítico| / Analítico * 100%
      df_final["Erro_Percentual"] = (
          np.abs(df_final["Freq_GHz_HFSS"] - df_final["Freq_GHz_Analitica"])
          / df_final["Freq_GHz_Analitica"]
      ) * 100.0
    except Exception as e:
      print("Aviso ao cruzar com arquivo analítico:", e)

  caminho_saida = os.path.join(pasta_dados, "cavidade_modos_20_hfss.txt")
  df_final.to_csv(caminho_saida, sep="\t", index=False, float_format="%.4f")

  print("\n" + "=" * 78)
  print(f"[OK] 20 Modos da Cavidade salvos com sucesso em: {caminho_saida}")
  print("=" * 78)
  print(df_final.to_string(index=False))
  print("=" * 78)

  # ============================================================================
  # 6. GERAÇÃO DO GRÁFICO DO ESPECTRO (IDÊNTICO AO ANALÍTICO E TIDY3D)
  # ============================================================================
  fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)

  if "Modo_Teorico" in df_final.columns:
    is_tm = df_final["Modo_Teorico"].astype(str).str.startswith("TM")
    df_te = df_final[~is_tm]
    df_tm = df_final[is_tm]
  else:
    df_te = df_final
    df_tm = df_final.iloc[0:0]

  # Linhas verticais do espectro
  ax.vlines(
      df_te["Freq_GHz_HFSS"],
      ymin=0,
      ymax=1.0,
      colors="royalblue",
      linewidth=1.8,
      label="Modos TE (HFSS)",
      alpha=0.85,
  )
  if not df_tm.empty:
    ax.vlines(
        df_tm["Freq_GHz_HFSS"],
        ymin=0,
        ymax=1.0,
        colors="crimson",
        linewidth=1.8,
        linestyles="dashed",
        label="Modos TM (HFSS)",
        alpha=0.85,
    )

  # Anotação com destaque nos 5 primeiros modos (incluindo o modo fundamental)
  for i in range(min(5, len(df_final))):
    row = df_final.iloc[i]
    modo_lbl = (
        row["Modo_Teorico"]
        if "Modo_Teorico" in df_final.columns
        else f"Modo {int(row['Ordem'])}"
    )
    cor_pt = "crimson" if str(modo_lbl).startswith("TM") else "royalblue"
    ax.plot(row["Freq_GHz_HFSS"], 1.0, marker="o", color=cor_pt, markersize=6)
    ax.annotate(
        f"{modo_lbl}\n{row['Freq_GHz_HFSS']:.3f} GHz",
        xy=(row["Freq_GHz_HFSS"], 1.0),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        fontweight="bold",
        color=cor_pt,
    )

  # Identificação do modo fundamental
  modo_fund = df_final.iloc[0]
  lbl_fund = (
      modo_fund["Modo_Teorico"]
      if "Modo_Teorico" in df_final.columns
      else "TE_101"
  )
  erro_str = (
      f" (Erro: {modo_fund['Erro_Percentual']:.3f}%)"
      if "Erro_Percentual" in df_final.columns
      else ""
  )
  erro_medio_str = (
      f" | Erro Médio (20 modos): {df_final['Erro_Percentual'].mean():.3f}%"
      if "Erro_Percentual" in df_final.columns
      else ""
  )
  ax.text(
      0.02,
      0.86,
      (
          f"Modo Fundamental: {lbl_fund} @ {modo_fund['Freq_GHz_HFSS']:.4f}"
          f" GHz{erro_str}\nDimensões: a = {a_mm} mm, b = {b_mm} mm, d ="
          f" {d_mm} mm{erro_medio_str}"
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

  ax.set_title(
      r"Espectro de Ressonância da Cavidade Retangular ($N = 20$ Modos - HFSS)",
      fontsize=12,
      fontweight="bold",
  )
  ax.set_xlabel("Frequência de Ressonância (GHz)", fontsize=10)
  ax.set_yticks([])  # Remove escala de amplitude arbitrária
  ax.set_ylim(0, 1.35)
  ax.grid(axis="x", linestyle="--", alpha=0.5)
  ax.legend(loc="upper right", framealpha=0.95)

  plt.tight_layout()
  caminho_fig = os.path.join(pasta_imagens, "cavidade_20_modos_hfss.png")
  plt.savefig(caminho_fig, dpi=300)
  print(f"[OK] Gráfico da Cavidade (HFSS) salvo em: {caminho_fig}")
  plt.show()
else:
  print(
      "[ERRO] Não foi possível extrair a lista de modos. Verifique a janela do"
      " HFSS aberta."
  )
