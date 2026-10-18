# ============================================================
# Script para rodar o PILOTO do EC03 e salvar CSV
# Rode no R com:  source("rodar_piloto.R")
# ============================================================

library(ExpDE)
library(smoof)

# --- Parâmetros do piloto ---
set.seed(2026)
pilot_dims <- sort(sample(2:150, 5))
n_pilot    <- 15

cat("Dimensões do piloto:", pilot_dims, "\n")
cat("Repetições por dimensão:", n_pilot, "\n")
cat("Total de execuções:", length(pilot_dims) * n_pilot * 2, "\n\n")

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

# --- Execução do piloto ---
pilot_data <- data.frame()
contador <- 0
total <- length(pilot_dims) * n_pilot * 2

for (d in pilot_dims) {
  for (r in 1:n_pilot) {
    fb1 <- run_expde(d, recpars1, mutpars1, seed_val = d * 10000 + r)
    contador <- contador + 1
    cat(sprintf("  [%d/%d] dim=%d rep=%d Config1 -> Fbest=%.4e\n", contador, total, d, r, fb1))

    fb2 <- run_expde(d, recpars2, mutpars2, seed_val = d * 10000 + r + 5000)
    contador <- contador + 1
    cat(sprintf("  [%d/%d] dim=%d rep=%d Config2 -> Fbest=%.4e\n", contador, total, d, r, fb2))

    pilot_data <- rbind(pilot_data,
      data.frame(dim = d, rep = r, config = "Config1", Fbest = fb1),
      data.frame(dim = d, rep = r, config = "Config2", Fbest = fb2))
  }
}

write.csv(pilot_data, "dados_piloto_EC03.csv", row.names = FALSE)
cat("\nPiloto concluído! Salvo em dados_piloto_EC03.csv\n")
cat("Total de linhas:", nrow(pilot_data), "\n")
