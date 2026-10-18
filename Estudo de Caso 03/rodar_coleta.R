# ============================================================
# Script para rodar a COLETA PRINCIPAL do EC03 e salvar CSV
# RODE DEPOIS do piloto (precisa dos dados do piloto para N e n_final)
# No R:  source("rodar_coleta.R")
# ============================================================

library(ExpDE)
library(smoof)
library(dplyr)
library(tidyr)

# --- Carregar dados do piloto ---
if (!file.exists("dados_piloto_EC03.csv")) {
  stop("Rode primeiro o rodar_piloto.R para gerar dados_piloto_EC03.csv")
}
pilot_data <- read.csv("dados_piloto_EC03.csv", stringsAsFactors = FALSE)
cat("Dados do piloto carregados:", nrow(pilot_data), "linhas\n")

# --- Recuperar pilot_dims e n_pilot ---
set.seed(2026)
pilot_dims <- sort(sample(2:150, 5))
n_pilot    <- 15

# --- Calcular N a partir da tabela de potência ---
calc_power_paired <- function(N, d_star = 0.5, alpha = 0.05) {
  df     <- N - 1
  ncp    <- d_star * sqrt(N)
  t_crit <- qt(1 - alpha/2, df)
  power  <- 1 - pt(t_crit, df, ncp) + pt(-t_crit, df, ncp)
  return(power)
}

N_range <- 30:40
power_table <- data.frame(
  N     = N_range,
  Poder = sapply(N_range, calc_power_paired)
)
N <- min(power_table$N[power_table$Poder >= 0.8])
cat("N (instâncias):", N, "\n")

# --- Calcular n_final a partir do piloto ---
pilot_sds <- pilot_data %>%
  group_by(dim, config) %>%
  summarise(sd_log = sd(log(Fbest)), .groups = "drop") %>%
  pivot_wider(names_from = config, values_from = sd_log, names_prefix = "sd_")

s1_pool <- sqrt(mean(pilot_sds$sd_Config1^2))
s2_pool <- sqrt(mean(pilot_sds$sd_Config2^2))

pilot_block <- pilot_data %>%
  group_by(dim, config) %>%
  summarise(mean_Fbest = mean(Fbest), .groups = "drop") %>%
  pivot_wider(names_from = config, values_from = mean_Fbest)

pilot_block$log_diff <- log(pilot_block$Config1) - log(pilot_block$Config2)
sigma_phi_pilot <- sd(pilot_block$log_diff)

target_se2 <- 0.1 * sigma_phi_pilot^2
n_equal <- ceiling((s1_pool^2 + s2_pool^2) / target_se2)
n_final <- max(n_equal, 30)
cat("n_final (repetições por célula):", n_final, "\n")

# --- Configurações do Grupo B ---
recpars1 <- list(name = "recombination_exp", cr = 0.6)
mutpars1 <- list(name = "mutation_best", f = 2)
recpars2 <- list(name = "recombination_geo", alpha = 0.6)
mutpars2 <- list(name = "mutation_rand", f = 1.2)
selpars  <- list(name = "selection_standard")

# --- Função auxiliar ---
run_expde <- function(dim_val, recpars, mutpars, seed_val) {
  fn <<- function(X) {
    if (!is.matrix(X)) X <- matrix(X, nrow = 1)
    apply(X, 1, smoof::makeRosenbrockFunction(dimensions = dim_val))
  }
  stopcrit <- list(names = "stop_maxeval",
                   maxevals = 5000 * dim_val,
                   maxiter  = 100 * dim_val)
  probpars <- list(name = "fn",
                   xmin = rep(-5, dim_val),
                   xmax = rep(10, dim_val))
  popsize  <- 5 * dim_val

  set.seed(seed_val)
  out <- ExpDE(mutpars  = mutpars,
               recpars  = recpars,
               popsize  = popsize,
               selpars  = selpars,
               stopcrit = stopcrit,
               probpars = probpars,
               showpars = list(show.iters = "none"))
  return(out$Fbest)
}

# --- Sortear instâncias ---
set.seed(933)
dims_disponiveis <- setdiff(2:150, pilot_dims)
dims_experimento <- sort(sample(dims_disponiveis, N))
cat("Dimensões do experimento:", dims_experimento, "\n")

# --- Gerar plano de execução aleatorizado ---
set.seed(7777)
exec_plan <- expand.grid(config = c("Config1", "Config2"),
                         dim    = dims_experimento,
                         rep    = 1:n_final)
exec_plan <- exec_plan[sample(nrow(exec_plan)), ]

total_runs <- nrow(exec_plan)
cat("Total de execuções:", total_runs, "\n")
cat("Início:", format(Sys.time(), "%H:%M:%S"), "\n\n")

# --- Verificar se já existe progresso parcial ---
arquivo_csv <- "dados_EC03_grupoB.csv"
arquivo_parcial <- "dados_EC03_grupoB_parcial.csv"

if (file.exists(arquivo_parcial)) {
  dados_brutos <- read.csv(arquivo_parcial, stringsAsFactors = FALSE)
  ja_feitas <- nrow(dados_brutos)
  cat(sprintf("Retomando de progresso parcial: %d/%d já concluídas\n", ja_feitas, total_runs))

  # Identificar quais já foram feitas
  feitas <- paste(dados_brutos$Algorithm, dados_brutos$Instance, dados_brutos$Rep, sep = "_")
  pendentes <- which(!paste(exec_plan$config, exec_plan$dim, exec_plan$rep, sep = "_") %in% feitas)
  cat(sprintf("Faltam %d execuções\n\n", length(pendentes)))
} else {
  dados_brutos <- data.frame()
  pendentes <- seq_len(total_runs)
  ja_feitas <- 0
}

# --- Executar ---
t_inicio <- Sys.time()

for (idx in seq_along(pendentes)) {
  i   <- pendentes[idx]
  row <- exec_plan[i, ]
  d   <- row$dim
  r   <- row$rep
  cfg <- as.character(row$config)

  if (cfg == "Config1") {
    fb <- run_expde(d, recpars1, mutpars1, seed_val = d * 10000 + r + 100)
  } else {
    fb <- run_expde(d, recpars2, mutpars2, seed_val = d * 10000 + r + 6000)
  }

  dados_brutos <- rbind(dados_brutos,
    data.frame(Algorithm = cfg, Instance = d, Rep = r, Result = fb))

  total_feitas <- ja_feitas + idx

  if (total_feitas %% 10 == 0) {
    elapsed <- as.numeric(difftime(Sys.time(), t_inicio, units = "mins"))
    rate    <- idx / elapsed
    faltam  <- (length(pendentes) - idx) / rate
    cat(sprintf("  [%d/%d] dim=%d %s  (%.1f min, ~%.0f min restantes)\n",
                total_feitas, total_runs, d, cfg, elapsed, faltam))
  }

  # Salvar progresso parcial a cada 50 execuções
  if (total_feitas %% 50 == 0) {
    write.csv(dados_brutos, arquivo_parcial, row.names = FALSE)
  }
}

# --- Salvar resultado final ---
write.csv(dados_brutos, arquivo_csv, row.names = FALSE)
cat(sprintf("\nColeta concluída! Salvo em %s\n", arquivo_csv))
cat("Total de linhas:", nrow(dados_brutos), "\n")
cat("Tempo total:", round(as.numeric(difftime(Sys.time(), t_inicio, units = "hours")), 1), "horas\n")

# Limpar parcial
if (file.exists(arquivo_parcial)) file.remove(arquivo_parcial)
