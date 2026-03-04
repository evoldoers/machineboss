#!/usr/bin/env Rscript
# Plot scaling behavior of Machine Boss DP backends.
# Reads benchmarks/results/*.json, writes benchmarks/figures/*.pdf.

library(jsonlite)
library(ggplot2)
library(dplyr)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

script_dir <- "."
args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("--file=", args, value = TRUE)
if (length(script_arg) > 0) {
  script_dir <- dirname(sub("--file=", "", script_arg[1]))
}
results_dir <- file.path(script_dir, "results")
figures_dir <- file.path(script_dir, "figures")

dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

json_files <- list.files(results_dir, pattern = "\\.json$", full.names = TRUE)
if (length(json_files) == 0) {
  stop("No result files found in ", results_dir)
}

all_data <- do.call(rbind, lapply(json_files, function(f) {
  raw <- fromJSON(f)
  df <- as.data.frame(raw$results, stringsAsFactors = FALSE)
  df$hardware_id <- raw$hardware_id
  df
}))

# Ensure numeric types
all_data$S <- as.integer(all_data$S)
all_data$L <- as.integer(all_data$L)
all_data$Li <- as.integer(all_data$Li)
all_data$Lo <- as.integer(all_data$Lo)
all_data$mean_seconds <- as.numeric(all_data$mean_seconds)
all_data$std_seconds <- as.numeric(all_data$std_seconds)

# Nicer backend labels
backend_levels <- c(
  "cpp", "jax_1d_simple", "jax_1d_optimal",
  "jax_2d_simple", "jax_2d_optimal",
  "jax_gpu_1d", "jax_gpu_2d", "js_cpu"
)
backend_labels <- c(
  "C++ native", "JAX 1D scan", "JAX 1D parallel",
  "JAX 2D scan", "JAX 2D wavefront",
  "JAX GPU 1D", "JAX GPU 2D", "JS CPU"
)
all_data$Backend <- factor(all_data$backend,
  levels = backend_levels,
  labels = backend_labels
)

cat("Loaded", nrow(all_data), "records from", length(json_files), "file(s)\n")

# Common theme
theme_bench <- theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold"),
    plot.title = element_text(size = 13, face = "bold")
  )


# ---------------------------------------------------------------------------
# 1D plots: scaling by sequence length L
# ---------------------------------------------------------------------------

data_1d <- all_data[all_data$problem == "1D", ]

if (nrow(data_1d) > 0) {
  for (algo in unique(data_1d$algorithm)) {
    df <- data_1d[data_1d$algorithm == algo, ]
    if (nrow(df) == 0) next

    p <- ggplot(df, aes(x = L, y = mean_seconds, color = Backend, shape = Backend)) +
      geom_line(linewidth = 0.7) +
      geom_point(size = 2.5) +
      geom_errorbar(aes(ymin = pmax(mean_seconds - std_seconds, 1e-6),
                        ymax = mean_seconds + std_seconds),
                    width = 0.05, linewidth = 0.4) +
      facet_wrap(~ S, labeller = label_both, scales = "free_y") +
      scale_x_log10() +
      scale_y_log10() +
      labs(
        title = paste0("1D ", algo, ": scaling by sequence length L"),
        x = "Sequence length L",
        y = "Time (seconds)"
      ) +
      theme_bench

    fname <- paste0("1d_scaling_L_", algo, ".pdf")
    ggsave(file.path(figures_dir, fname), p, width = 12, height = 5)
    cat("Wrote", fname, "\n")
  }

  # Backend comparison for 1D
  for (algo in unique(data_1d$algorithm)) {
    df <- data_1d[data_1d$algorithm == algo, ]
    if (nrow(df) == 0 || !("cpp" %in% df$backend)) next

    cpp_ref <- df[df$backend == "cpp", c("S", "L", "mean_seconds")]
    names(cpp_ref)[3] <- "cpp_time"
    df_ratio <- merge(df, cpp_ref, by = c("S", "L"))
    df_ratio$ratio <- df_ratio$mean_seconds / df_ratio$cpp_time
    df_ratio$config <- paste0("L=", df_ratio$L)

    if (nrow(df_ratio) == 0) next

    p <- ggplot(df_ratio, aes(x = config, y = ratio, fill = Backend)) +
      geom_col(position = position_dodge(width = 0.8), width = 0.7) +
      geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
      facet_wrap(~ S, labeller = label_both) +
      scale_y_log10() +
      labs(
        title = paste0("1D ", algo, ": time relative to C++ native"),
        x = "Configuration",
        y = "Ratio (vs C++ native)"
      ) +
      theme_bench +
      theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))

    fname <- paste0("1d_comparison_", algo, ".pdf")
    ggsave(file.path(figures_dir, fname), p, width = 12, height = 5)
    cat("Wrote", fname, "\n")
  }
}


# ---------------------------------------------------------------------------
# 2D plots: scaling by input length, output length, and states
# ---------------------------------------------------------------------------

data_2d <- all_data[all_data$problem == "2D", ]

if (nrow(data_2d) > 0) {
  for (algo in unique(data_2d$algorithm)) {
    df <- data_2d[data_2d$algorithm == algo, ]
    if (nrow(df) == 0) next

    # Scaling by Li (faceted by S rows, Lo columns)
    p <- ggplot(df, aes(x = Li, y = mean_seconds, color = Backend, shape = Backend)) +
      geom_line(linewidth = 0.7) +
      geom_point(size = 2.5) +
      geom_errorbar(aes(ymin = pmax(mean_seconds - std_seconds, 1e-6),
                        ymax = mean_seconds + std_seconds),
                    width = 0.05, linewidth = 0.4) +
      facet_grid(S ~ Lo, labeller = label_both, scales = "free_y") +
      scale_x_log10() +
      scale_y_log10() +
      labs(
        title = paste0("2D ", algo, ": scaling by input length"),
        x = expression(L[inp]),
        y = "Time (seconds)"
      ) +
      theme_bench

    fname <- paste0("2d_scaling_Li_", algo, ".pdf")
    ggsave(file.path(figures_dir, fname), p, width = 12, height = 10)
    cat("Wrote", fname, "\n")

    # Scaling by Lo (faceted by S rows, Li columns)
    p <- ggplot(df, aes(x = Lo, y = mean_seconds, color = Backend, shape = Backend)) +
      geom_line(linewidth = 0.7) +
      geom_point(size = 2.5) +
      geom_errorbar(aes(ymin = pmax(mean_seconds - std_seconds, 1e-6),
                        ymax = mean_seconds + std_seconds),
                    width = 0.05, linewidth = 0.4) +
      facet_grid(S ~ Li, labeller = label_both, scales = "free_y") +
      scale_x_log10() +
      scale_y_log10() +
      labs(
        title = paste0("2D ", algo, ": scaling by output length"),
        x = expression(L[out]),
        y = "Time (seconds)"
      ) +
      theme_bench

    fname <- paste0("2d_scaling_Lo_", algo, ".pdf")
    ggsave(file.path(figures_dir, fname), p, width = 12, height = 10)
    cat("Wrote", fname, "\n")

    # Scaling by S (faceted by Li rows, Lo columns)
    p <- ggplot(df, aes(x = S, y = mean_seconds, color = Backend, shape = Backend)) +
      geom_line(linewidth = 0.7) +
      geom_point(size = 2.5) +
      geom_errorbar(aes(ymin = pmax(mean_seconds - std_seconds, 1e-6),
                        ymax = mean_seconds + std_seconds),
                    width = 0.05, linewidth = 0.4) +
      facet_grid(Li ~ Lo, labeller = label_both, scales = "free_y") +
      scale_x_log10() +
      scale_y_log10() +
      labs(
        title = paste0("2D ", algo, ": scaling by number of states"),
        x = "States S",
        y = "Time (seconds)"
      ) +
      theme_bench

    fname <- paste0("2d_scaling_S_", algo, ".pdf")
    ggsave(file.path(figures_dir, fname), p, width = 12, height = 10)
    cat("Wrote", fname, "\n")
  }

  # Backend comparison for 2D
  for (algo in unique(data_2d$algorithm)) {
    df <- data_2d[data_2d$algorithm == algo, ]
    if (nrow(df) == 0 || !("cpp" %in% df$backend)) next

    cpp_ref <- df[df$backend == "cpp", c("S", "Li", "Lo", "mean_seconds")]
    names(cpp_ref)[4] <- "cpp_time"
    df_ratio <- merge(df, cpp_ref, by = c("S", "Li", "Lo"))
    df_ratio$ratio <- df_ratio$mean_seconds / df_ratio$cpp_time
    df_ratio$config <- paste0("Li=", df_ratio$Li, " Lo=", df_ratio$Lo)

    if (nrow(df_ratio) == 0) next

    p <- ggplot(df_ratio, aes(x = config, y = ratio, fill = Backend)) +
      geom_col(position = position_dodge(width = 0.8), width = 0.7) +
      geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
      facet_wrap(~ S, labeller = label_both) +
      scale_y_log10() +
      labs(
        title = paste0("2D ", algo, ": time relative to C++ native"),
        x = "Configuration",
        y = "Ratio (vs C++ native)"
      ) +
      theme_bench +
      theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7))

    fname <- paste0("2d_comparison_", algo, ".pdf")
    ggsave(file.path(figures_dir, fname), p, width = 14, height = 6)
    cat("Wrote", fname, "\n")
  }
}

cat("All plots written to", figures_dir, "\n")
