# ==============================================================================
# 고차원 데이터에서 이질성 유발 변수를 탐색하는 HP 클러스터링 시뮬레이션
# (기존 코드 틀 유지 + 피드백 반영 버전)
# ------------------------------------------------------------------------------
# 반영 내용
# 1) 시뮬레이션 세팅을 보고서에 맞게 수정: n = 300, p = 20, K = 3, q = 5,
#    a in {1.8, 1.5, 1.3}
# 2) Naive Lasso / HP 모두 mu_j = mu0 + delta_j 구조를 E-step과 M-step에 반영
# 3) HP는 sum-to-zero 제약을 Q-basis로 유지하면서 mu0와 delta를 함께 갱신
# 4) HP는 pilot -> adaptive weight -> adaptive HP 순서로 구현
# 5) 공통 diagonal covariance sigma^2를 추정
# 6) HP와 Naive Lasso의 lambda는 BIC로 선택
# 7) n_k = 0 방어 코드 추가, W = diag(n_k)는 변수 루프 밖에서 생성
# 8) Refit / Oracle-feature baseline은 차원에 따라 Mclust 모형명을 분기하여 사용
# 9) Oracle을 두 종류로 분리: (i) 정답 변수만 아는 feature-oracle baseline,
#    (ii) 생성모수까지 아는 true-parameter oracle
# ==============================================================================

if (!require(mclust)) install.packages("mclust")
if (!require(sparcl)) install.packages("sparcl")
library(mclust)
library(sparcl)

options(stringsAsFactors = FALSE)

base_seed <- 2026
n_rep <- 10
signal_grid <- c(1.8, 1.5, 1.3)
print_each_run <- FALSE

all_results <- list()
all_results_numeric <- list()
store_idx <- 1

for (rep_idx in 1:n_rep) {
  set.seed(base_seed + rep_idx - 1)

  cat("\n\n###################################################################################################\n")
  cat(sprintf(" [반복 %d / %d] 시뮬레이션 시작\n", rep_idx, n_rep))
  cat("###################################################################################################\n")

for (scenario_idx in seq_along(signal_grid)) {

  # ==============================================================================
  # Step 1. 시뮬레이션 세팅
  # ==============================================================================
  n <- 200
  p <- 300
  K <- 3
  q <- 10
  a <- signal_grid[scenario_idx]

  pi_true <- rep(1 / K, K)
  Z_true <- sample(1:K, n, replace = TRUE, prob = pi_true)

  true_delta <- matrix(0, nrow = K, ncol = p)
  true_delta[1, 1:q] <- a
  true_delta[2, 1:q] <- 0
  true_delta[3, 1:q] <- -a

  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  for (i in 1:n) {
    X[i, ] <- X[i, ] + true_delta[Z_true[i], ]
  }

  # 노이즈 변수의 분산을 2배로 증가
  X[, (q + 1):p] <- X[, (q + 1):p] * sqrt(2)

  # 데이터 전처리: center만 수행
  X <- scale(X, center = TRUE, scale = FALSE)
  center_vec <- attr(X, "scaled:center")
  X <- as.matrix(X)
  true_vars <- 1:q

  cat("\n\n===================================================================================================\n")
  cat(sprintf(" [시나리오 %d] a = %.1f (n=%d, p=%d, K=%d)\n", scenario_idx, a, n, p, K))
  cat("===================================================================================================\n")

  # ==============================================================================
  # Step 2. 기존 비교 모형 (K-means, PCA, GMM)
  # ==============================================================================
  cat("\n[1] 일반 비교 모형 적합 중...\n")

  km_res <- kmeans(X, centers = K, nstart = 30)
  ari_km <- adjustedRandIndex(Z_true, km_res$cluster)

  pca_res <- prcomp(X, center = FALSE, scale. = FALSE)
  cum_var <- cumsum(pca_res$sdev^2) / sum(pca_res$sdev^2)
  n_comp <- which(cum_var >= 0.80)[1]
  km_pca_res <- kmeans(pca_res$x[, 1:n_comp, drop = FALSE], centers = K, nstart = 30)
  ari_pca <- adjustedRandIndex(Z_true, km_pca_res$cluster)

  gmm_res <- tryCatch(
    Mclust(X, G = K, modelNames = "EEI", verbose = FALSE),
    error = function(e) NULL
  )
  ari_gmm <- if (is.null(gmm_res)) NA else adjustedRandIndex(Z_true, gmm_res$classification)

  # ==============================================================================
  # Step 2-1. EM 계열 방법 초기값 준비
  # ==============================================================================
  tau_init <- matrix(0, n, K)
  for (i in 1:n) tau_init[i, km_res$cluster[i]] <- 1

  n_k_init <- colSums(tau_init)
  x_bar_init <- matrix(0, nrow = K, ncol = p)
  for (k in 1:K) {
    if (n_k_init[k] > 0) {
      x_bar_init[k, ] <- colSums(sweep(X, 1, tau_init[, k], "*")) / n_k_init[k]
    }
  }

  mu0_init <- colMeans(x_bar_init)
  delta_init <- sweep(x_bar_init, 2, mu0_init, "-")

  sigma2_init <- rep(0, p)
  for (k in 1:K) {
    diff_X <- sweep(X, 2, mu0_init + delta_init[k, ], "-")
    sigma2_init <- sigma2_init + colSums(sweep(diff_X^2, 1, tau_init[, k], "*"))
  }
  sigma2_init <- pmax(sigma2_init / n, 1e-6)

  # ==============================================================================
  # Step 3. Sparse K-means & Refit
  # ==============================================================================
  cat("[2] 변수 선택 벤치마크: Sparse K-means 적합 중...\n")

  capture.output(
    km_perm <- KMeansSparseCluster.permute(
      X,
      K = K,
      wbounds = seq(1.5, 6, length.out = 7),
      nperms = 5
    )
  )
  km_sparse <- KMeansSparseCluster(X, K = K, wbounds = km_perm$bestw, silent = TRUE)

  ari_sparse <- adjustedRandIndex(Z_true, km_sparse[[1]]$Cs)
  selected_vars_sparse <- which(km_sparse[[1]]$ws > 1e-4)

  tpr_sparse <- length(intersect(true_vars, selected_vars_sparse)) / length(true_vars)
  fpr_sparse <- length(setdiff(selected_vars_sparse, true_vars)) / (p - length(true_vars))
  shat_sparse <- length(selected_vars_sparse)

  if (length(selected_vars_sparse) > 0) {
    mclust_model_sparse <- if (length(selected_vars_sparse) == 1) "E" else "EEI"
    refit_sparse_res <- tryCatch(
      Mclust(X[, selected_vars_sparse, drop = FALSE], G = K, modelNames = mclust_model_sparse, verbose = FALSE),
      error = function(e) NULL
    )
    ari_sparse_refit <- if (!is.null(refit_sparse_res)) {
      adjustedRandIndex(Z_true, refit_sparse_res$classification)
    } else {
      NA
    }
  } else {
    ari_sparse_refit <- 0
  }

  # ==============================================================================
  # Step 4. Naive Lasso (Element-wise L1 + mu0 반영 + diagonal sigma2) & Refit
  # ==============================================================================
  cat("[3] 과거 버전 보정: Naive Lasso 적합 중...\n")

  lambda_lasso_grid <- c(5, 10, 20, 30, 40, 50, 60)
  best_bic_lasso <- Inf
  best_lambda_lasso <- NA

  best_tau_lasso <- tau_init
  best_mu0_lasso <- mu0_init
  best_delta_lasso <- delta_init
  best_sigma2_lasso <- sigma2_init

  for (lambda_lasso in lambda_lasso_grid) {

    tau_lasso <- tau_init
    mu0_lasso <- mu0_init
    delta_lasso <- delta_init
    sigma2_lasso <- sigma2_init

    for (iter in 1:60) {
      n_k <- colSums(tau_lasso)
      n_k_safe <- pmax(n_k, 1e-8)
      pi_k <- pmax(n_k / n, 1e-8)
      pi_k <- pi_k / sum(pi_k)

      delta_new_lasso <- matrix(0, nrow = K, ncol = p)
      mu0_new_lasso <- numeric(p)

      for (j in 1:p) {
        x_bar_j <- rep(0, K)
        for (k in 1:K) {
          if (n_k[k] > 1e-8) {
            x_bar_j[k] <- sum(tau_lasso[, k] * X[, j]) / n_k[k]
          } else {
            x_bar_j[k] <- mu0_lasso[j] + delta_lasso[k, j]
          }
        }

        delta_j <- delta_lasso[, j]
        mu0_j <- mu0_lasso[j]
        thresh_j <- lambda_lasso * sigma2_lasso[j] / n_k_safe

        for (inner in 1:10) {
          mu0_j <- sum(n_k_safe * (x_bar_j - delta_j)) / sum(n_k_safe)
          temp_delta <- sign(x_bar_j - mu0_j) * pmax(0, abs(x_bar_j - mu0_j) - thresh_j)
          temp_delta <- temp_delta - mean(temp_delta)

          if (max(abs(temp_delta - delta_j)) < 1e-6) {
            delta_j <- temp_delta
            break
          }
          delta_j <- temp_delta
        }

        mu0_j <- sum(n_k_safe * (x_bar_j - delta_j)) / sum(n_k_safe)
        mu0_new_lasso[j] <- mu0_j
        delta_new_lasso[, j] <- delta_j
      }

      sigma2_new_lasso <- rep(0, p)
      for (k in 1:K) {
        diff_X <- sweep(X, 2, mu0_new_lasso + delta_new_lasso[k, ], "-")
        sigma2_new_lasso <- sigma2_new_lasso + colSums(sweep(diff_X^2, 1, tau_lasso[, k], "*"))
      }
      sigma2_new_lasso <- pmax(sigma2_new_lasso / n, 1e-6)

      log_tau <- matrix(0, nrow = n, ncol = K)
      for (k in 1:K) {
        diff_X <- sweep(X, 2, mu0_new_lasso + delta_new_lasso[k, ], "-")
        log_tau[, k] <- log(pi_k[k]) - 0.5 * rowSums(sweep(diff_X^2, 2, sigma2_new_lasso, "/"))
      }

      row_max <- apply(log_tau, 1, max)
      tau_exp <- exp(sweep(log_tau, 1, row_max, "-"))
      tau_den <- rowSums(tau_exp)
      tau_den[tau_den <= 0] <- 1e-8
      tau_new_lasso <- sweep(tau_exp, 1, tau_den, "/")

      diff_lasso <- max(
        abs(delta_lasso - delta_new_lasso),
        abs(mu0_lasso - mu0_new_lasso),
        abs(sigma2_lasso - sigma2_new_lasso)
      )

      delta_lasso <- delta_new_lasso
      mu0_lasso <- mu0_new_lasso
      sigma2_lasso <- sigma2_new_lasso
      tau_lasso <- tau_new_lasso

      if (diff_lasso < 1e-5) break
    }

    pi_k <- pmax(colSums(tau_lasso) / n, 1e-8)
    pi_k <- pi_k / sum(pi_k)

    log_mat <- matrix(0, nrow = n, ncol = K)
    const_term <- -0.5 * (p * log(2 * pi) + sum(log(sigma2_lasso)))
    for (k in 1:K) {
      diff_X <- sweep(X, 2, mu0_lasso + delta_lasso[k, ], "-")
      log_mat[, k] <- log(pi_k[k]) + const_term - 0.5 * rowSums(sweep(diff_X^2, 2, sigma2_lasso, "/"))
    }
    row_max <- apply(log_mat, 1, max)
    loglik_lasso <- sum(row_max + log(rowSums(exp(sweep(log_mat, 1, row_max, "-")))))

    nnz_per_var_lasso <- colSums(abs(delta_lasso) > 1e-4)
    df_lasso <- (K - 1) + p + p + sum(pmax(nnz_per_var_lasso - 1, 0))
    bic_lasso <- -2 * loglik_lasso + log(n) * df_lasso

    if (bic_lasso < best_bic_lasso) {
      best_bic_lasso <- bic_lasso
      best_lambda_lasso <- lambda_lasso
      best_tau_lasso <- tau_lasso
      best_mu0_lasso <- mu0_lasso
      best_delta_lasso <- delta_lasso
      best_sigma2_lasso <- sigma2_lasso
    }
  }

  ari_lasso <- adjustedRandIndex(Z_true, apply(best_tau_lasso, 1, which.max))
  norms_lasso <- apply(best_delta_lasso, 2, function(x) sqrt(sum(x^2)))
  selected_vars_lasso <- which(norms_lasso > 1e-4)

  tpr_lasso <- length(intersect(true_vars, selected_vars_lasso)) / length(true_vars)
  fpr_lasso <- length(setdiff(selected_vars_lasso, true_vars)) / (p - length(true_vars))
  shat_lasso <- length(selected_vars_lasso)

  if (length(selected_vars_lasso) > 0) {
    mclust_model_lasso <- if (length(selected_vars_lasso) == 1) "E" else "EEI"
    refit_lasso_res <- tryCatch(
      Mclust(X[, selected_vars_lasso, drop = FALSE], G = K, modelNames = mclust_model_lasso, verbose = FALSE),
      error = function(e) NULL
    )
    ari_lasso_refit <- if (!is.null(refit_lasso_res)) {
      adjustedRandIndex(Z_true, refit_lasso_res$classification)
    } else {
      NA
    }
  } else {
    ari_lasso_refit <- 0
  }

  # ==============================================================================
  # Step 5. 제안 모형: HP (Pilot Group Lasso -> Adaptive Group Lasso) + Refit
  # ==============================================================================
  cat("[4] 제안 모형 (HP + Refit) 적합 중...\n")

  ones <- matrix(1, K, 1)
  Q <- qr.Q(qr(ones), complete = TRUE)[, 2:K, drop = FALSE]

  # ------------------------------------------------------------------------------
  # Step 5-1. Pilot HP (non-adaptive group lasso) by BIC
  # ------------------------------------------------------------------------------
  lambda_hp_pilot_grid <- c(5, 10, 20, 40, 60, 80)
  best_bic_hp_pilot <- Inf
  best_lambda_hp_pilot <- NA

  best_tau_hp_pilot <- tau_init
  best_mu0_hp_pilot <- mu0_init
  best_delta_hp_pilot <- delta_init
  best_sigma2_hp_pilot <- sigma2_init

  for (lambda_hp_pilot in lambda_hp_pilot_grid) {

    tau_hp <- tau_init
    mu0_hp <- mu0_init
    delta_hp <- delta_init
    sigma2_hp <- sigma2_init

    for (iter in 1:100) {
      n_k <- colSums(tau_hp)
      n_k_safe <- pmax(n_k, 1e-8)
      pi_k_hp <- pmax(n_k / n, 1e-8)
      pi_k_hp <- pi_k_hp / sum(pi_k_hp)
      W <- diag(n_k_safe)

      delta_new_hp <- matrix(0, nrow = K, ncol = p)
      mu0_new_hp <- numeric(p)

      for (j in 1:p) {
        x_bar_j <- rep(0, K)
        for (k in 1:K) {
          if (n_k[k] > 1e-8) {
            x_bar_j[k] <- sum(tau_hp[, k] * X[, j]) / n_k[k]
          } else {
            x_bar_j[k] <- mu0_hp[j] + delta_hp[k, j]
          }
        }

        alpha <- as.vector(crossprod(Q, delta_hp[, j]))
        mu0_j <- mu0_hp[j]

        H_j <- t(Q) %*% W %*% Q / sigma2_hp[j]
        eig_j <- eigen(H_j, symmetric = TRUE, only.values = TRUE)$values
        eta <- 1 / max(max(Re(eig_j)), 1e-8)

        for (inner in 1:20) {
          mu0_j <- sum(n_k_safe * (x_bar_j - as.vector(Q %*% alpha))) / sum(n_k_safe)
          resid_j <- x_bar_j - mu0_j - as.vector(Q %*% alpha)
          grad_j <- -as.vector(t(Q) %*% (W %*% resid_j)) / sigma2_hp[j]

          alpha_step <- alpha - eta * grad_j
          norm_alpha <- sqrt(sum(alpha_step^2))
          thresh_alpha <- eta * lambda_hp_pilot

          if (norm_alpha > thresh_alpha) {
            alpha_new <- alpha_step * (1 - thresh_alpha / norm_alpha)
          } else {
            alpha_new <- rep(0, K - 1)
          }

          if (max(abs(alpha_new - alpha)) < 1e-6) {
            alpha <- alpha_new
            break
          }
          alpha <- alpha_new
        }

        mu0_j <- sum(n_k_safe * (x_bar_j - as.vector(Q %*% alpha))) / sum(n_k_safe)
        mu0_new_hp[j] <- mu0_j
        delta_new_hp[, j] <- as.vector(Q %*% alpha)
      }

      sigma2_new_hp <- rep(0, p)
      for (k in 1:K) {
        diff_X <- sweep(X, 2, mu0_new_hp + delta_new_hp[k, ], "-")
        sigma2_new_hp <- sigma2_new_hp + colSums(sweep(diff_X^2, 1, tau_hp[, k], "*"))
      }
      sigma2_new_hp <- pmax(sigma2_new_hp / n, 1e-6)

      log_tau <- matrix(0, nrow = n, ncol = K)
      for (k in 1:K) {
        diff_X <- sweep(X, 2, mu0_new_hp + delta_new_hp[k, ], "-")
        log_tau[, k] <- log(pi_k_hp[k]) - 0.5 * rowSums(sweep(diff_X^2, 2, sigma2_new_hp, "/"))
      }

      row_max <- apply(log_tau, 1, max)
      tau_exp <- exp(sweep(log_tau, 1, row_max, "-"))
      tau_den <- rowSums(tau_exp)
      tau_den[tau_den <= 0] <- 1e-8
      tau_new_hp <- sweep(tau_exp, 1, tau_den, "/")

      diff_hp <- max(
        abs(delta_hp - delta_new_hp),
        abs(mu0_hp - mu0_new_hp),
        abs(sigma2_hp - sigma2_new_hp)
      )

      delta_hp <- delta_new_hp
      mu0_hp <- mu0_new_hp
      sigma2_hp <- sigma2_new_hp
      tau_hp <- tau_new_hp

      if (diff_hp < 1e-5) break
    }

    pi_k_hp <- pmax(colSums(tau_hp) / n, 1e-8)
    pi_k_hp <- pi_k_hp / sum(pi_k_hp)

    log_mat <- matrix(0, nrow = n, ncol = K)
    const_term <- -0.5 * (p * log(2 * pi) + sum(log(sigma2_hp)))
    for (k in 1:K) {
      diff_X <- sweep(X, 2, mu0_hp + delta_hp[k, ], "-")
      log_mat[, k] <- log(pi_k_hp[k]) + const_term - 0.5 * rowSums(sweep(diff_X^2, 2, sigma2_hp, "/"))
    }
    row_max <- apply(log_mat, 1, max)
    loglik_hp_pilot <- sum(row_max + log(rowSums(exp(sweep(log_mat, 1, row_max, "-")))))

    active_vars_hp_pilot <- apply(delta_hp, 2, function(x) sqrt(sum(x^2))) > 1e-4
    df_hp_pilot <- (K - 1) + p + p + sum(active_vars_hp_pilot) * (K - 1)
    bic_hp_pilot <- -2 * loglik_hp_pilot + log(n) * df_hp_pilot

    if (bic_hp_pilot < best_bic_hp_pilot) {
      best_bic_hp_pilot <- bic_hp_pilot
      best_lambda_hp_pilot <- lambda_hp_pilot
      best_tau_hp_pilot <- tau_hp
      best_mu0_hp_pilot <- mu0_hp
      best_delta_hp_pilot <- delta_hp
      best_sigma2_hp_pilot <- sigma2_hp
    }
  }

  adaptive_w <- 1 / (sqrt(colSums(best_delta_hp_pilot^2)) + 1e-4)
  adaptive_w <- adaptive_w / median(adaptive_w)
  adaptive_w <- pmin(adaptive_w, 1e4)

  # ------------------------------------------------------------------------------
  # Step 5-2. Adaptive HP by BIC
  # ------------------------------------------------------------------------------
  lambda_hp_grid <- c(5, 10, 20, 30, 40, 60, 80)
  best_bic_hp <- Inf
  best_lambda_hp <- NA

  best_tau_hp <- best_tau_hp_pilot
  best_mu0_hp <- best_mu0_hp_pilot
  best_delta_hp <- best_delta_hp_pilot
  best_sigma2_hp <- best_sigma2_hp_pilot

  for (lambda_hp in lambda_hp_grid) {

    tau_hp <- best_tau_hp_pilot
    mu0_hp <- best_mu0_hp_pilot
    delta_hp <- best_delta_hp_pilot
    sigma2_hp <- best_sigma2_hp_pilot

    for (iter in 1:100) {
      n_k <- colSums(tau_hp)
      n_k_safe <- pmax(n_k, 1e-8)
      pi_k_hp <- pmax(n_k / n, 1e-8)
      pi_k_hp <- pi_k_hp / sum(pi_k_hp)
      W <- diag(n_k_safe)

      delta_new_hp <- matrix(0, nrow = K, ncol = p)
      mu0_new_hp <- numeric(p)

      for (j in 1:p) {
        x_bar_j <- rep(0, K)
        for (k in 1:K) {
          if (n_k[k] > 1e-8) {
            x_bar_j[k] <- sum(tau_hp[, k] * X[, j]) / n_k[k]
          } else {
            x_bar_j[k] <- mu0_hp[j] + delta_hp[k, j]
          }
        }

        alpha <- as.vector(crossprod(Q, delta_hp[, j]))
        mu0_j <- mu0_hp[j]

        H_j <- t(Q) %*% W %*% Q / sigma2_hp[j]
        eig_j <- eigen(H_j, symmetric = TRUE, only.values = TRUE)$values
        eta <- 1 / max(max(Re(eig_j)), 1e-8)

        for (inner in 1:20) {
          mu0_j <- sum(n_k_safe * (x_bar_j - as.vector(Q %*% alpha))) / sum(n_k_safe)
          resid_j <- x_bar_j - mu0_j - as.vector(Q %*% alpha)
          grad_j <- -as.vector(t(Q) %*% (W %*% resid_j)) / sigma2_hp[j]

          alpha_step <- alpha - eta * grad_j
          norm_alpha <- sqrt(sum(alpha_step^2))
          thresh_alpha <- eta * lambda_hp * adaptive_w[j]

          if (norm_alpha > thresh_alpha) {
            alpha_new <- alpha_step * (1 - thresh_alpha / norm_alpha)
          } else {
            alpha_new <- rep(0, K - 1)
          }

          if (max(abs(alpha_new - alpha)) < 1e-6) {
            alpha <- alpha_new
            break
          }
          alpha <- alpha_new
        }

        mu0_j <- sum(n_k_safe * (x_bar_j - as.vector(Q %*% alpha))) / sum(n_k_safe)
        mu0_new_hp[j] <- mu0_j
        delta_new_hp[, j] <- as.vector(Q %*% alpha)
      }

      sigma2_new_hp <- rep(0, p)
      for (k in 1:K) {
        diff_X <- sweep(X, 2, mu0_new_hp + delta_new_hp[k, ], "-")
        sigma2_new_hp <- sigma2_new_hp + colSums(sweep(diff_X^2, 1, tau_hp[, k], "*"))
      }
      sigma2_new_hp <- pmax(sigma2_new_hp / n, 1e-6)

      log_tau <- matrix(0, nrow = n, ncol = K)
      for (k in 1:K) {
        diff_X <- sweep(X, 2, mu0_new_hp + delta_new_hp[k, ], "-")
        log_tau[, k] <- log(pi_k_hp[k]) - 0.5 * rowSums(sweep(diff_X^2, 2, sigma2_new_hp, "/"))
      }

      row_max <- apply(log_tau, 1, max)
      tau_exp <- exp(sweep(log_tau, 1, row_max, "-"))
      tau_den <- rowSums(tau_exp)
      tau_den[tau_den <= 0] <- 1e-8
      tau_new_hp <- sweep(tau_exp, 1, tau_den, "/")

      diff_hp <- max(
        abs(delta_hp - delta_new_hp),
        abs(mu0_hp - mu0_new_hp),
        abs(sigma2_hp - sigma2_new_hp)
      )

      delta_hp <- delta_new_hp
      mu0_hp <- mu0_new_hp
      sigma2_hp <- sigma2_new_hp
      tau_hp <- tau_new_hp

      if (diff_hp < 1e-5) break
    }

    pi_k_hp <- pmax(colSums(tau_hp) / n, 1e-8)
    pi_k_hp <- pi_k_hp / sum(pi_k_hp)

    log_mat <- matrix(0, nrow = n, ncol = K)
    const_term <- -0.5 * (p * log(2 * pi) + sum(log(sigma2_hp)))
    for (k in 1:K) {
      diff_X <- sweep(X, 2, mu0_hp + delta_hp[k, ], "-")
      log_mat[, k] <- log(pi_k_hp[k]) + const_term - 0.5 * rowSums(sweep(diff_X^2, 2, sigma2_hp, "/"))
    }
    row_max <- apply(log_mat, 1, max)
    loglik_hp <- sum(row_max + log(rowSums(exp(sweep(log_mat, 1, row_max, "-")))))

    active_vars_hp <- apply(delta_hp, 2, function(x) sqrt(sum(x^2))) > 1e-4
    df_hp <- (K - 1) + p + p + sum(active_vars_hp) * (K - 1)
    bic_hp <- -2 * loglik_hp + log(n) * df_hp

    if (bic_hp < best_bic_hp) {
      best_bic_hp <- bic_hp
      best_lambda_hp <- lambda_hp
      best_tau_hp <- tau_hp
      best_mu0_hp <- mu0_hp
      best_delta_hp <- delta_hp
      best_sigma2_hp <- sigma2_hp
    }
  }

  ari_hp <- adjustedRandIndex(Z_true, apply(best_tau_hp, 1, which.max))
  delta_norms_hp <- apply(best_delta_hp, 2, function(x) sqrt(sum(x^2)))
  selected_vars_hp <- which(delta_norms_hp > 1e-4)

  tpr_hp <- length(intersect(true_vars, selected_vars_hp)) / length(true_vars)
  fpr_hp <- length(setdiff(selected_vars_hp, true_vars)) / (p - length(true_vars))
  shat_hp <- length(selected_vars_hp)

  if (length(selected_vars_hp) > 0) {
    mclust_model_hp <- if (length(selected_vars_hp) == 1) "E" else "EEI"
    refit_hp_res <- tryCatch(
      Mclust(X[, selected_vars_hp, drop = FALSE], G = K, modelNames = mclust_model_hp, verbose = FALSE),
      error = function(e) NULL
    )
    ari_hp_refit <- if (!is.null(refit_hp_res)) {
      adjustedRandIndex(Z_true, refit_hp_res$classification)
    } else {
      NA
    }
  } else {
    ari_hp_refit <- 0
  }

  # ==============================================================================
  # Step 6. Oracle 비교: (1) Feature-oracle baseline, (2) True-parameter oracle
  # ==============================================================================
  cat("[5] Oracle 비교 모형 적합 중...\n")

  # (1) 정답 변수 집합만 알고, 그 위에서 GMM을 다시 추정하는 baseline
  #     -> local optimum / 초기값 영향을 받을 수 있으므로 진짜 상한선은 아님
  oracle_feature_res <- tryCatch(
    Mclust(X[, true_vars, drop = FALSE], G = K, modelNames = "EEI", verbose = FALSE),
    error = function(e) NULL
  )
  ari_oracle_feature <- if (!is.null(oracle_feature_res)) {
    adjustedRandIndex(Z_true, oracle_feature_res$classification)
  } else {
    NA
  }

  # (2) 생성에 사용한 진짜 모수(pi, mean, sigma2)를 알고 있다고 가정한 true-parameter oracle
  mu_true_centered <- sweep(true_delta[, true_vars, drop = FALSE], 2, center_vec[true_vars], "-")
  sigma2_true_vars <- rep(1, q)

  log_post_true <- matrix(0, nrow = n, ncol = K)
  const_true <- -0.5 * (q * log(2 * pi) + sum(log(sigma2_true_vars)))
  for (k in 1:K) {
    diff_k <- sweep(X[, true_vars, drop = FALSE], 2, mu_true_centered[k, ], "-")
    log_post_true[, k] <- log(pi_true[k]) + const_true - 0.5 * rowSums(sweep(diff_k^2, 2, sigma2_true_vars, "/"))
  }
  z_oracle_true <- max.col(log_post_true, ties.method = "first")
  ari_oracle_true <- adjustedRandIndex(Z_true, z_oracle_true)

  # ==============================================================================
  # Step 7. 반복별 결과 저장 및 출력
  # ==============================================================================
  method_names <- c(
    "K-means",
    "PCA + K-means",
    "GMM (Unpenalized)",
    "Sparse K-means (sparcl)",
    "Sparse K-means (sparcl) + Refit",
    "Naive Lasso (L1 + mu0)",
    "Naive Lasso (L1 + mu0) + Refit",
    "Proposed 1: HP (Adaptive Group L2)",
    "Proposed 1: HP (Adaptive Group L2) + Refit",
    "Oracle-feature baseline (True Vars)",
    "True-parameter oracle"
  )

  used_dims_vec <- c(p, n_comp, p, shat_sparse, shat_sparse, shat_lasso, shat_lasso, p, shat_hp, q, q)
  lambda_num_vec <- c(NA, NA, NA, NA, NA, best_lambda_lasso, NA, best_lambda_hp, NA, NA, NA)
  ari_vec <- c(ari_km, ari_pca, ari_gmm, ari_sparse, ari_sparse_refit, ari_lasso, ari_lasso_refit, ari_hp, ari_hp_refit, ari_oracle_feature, ari_oracle_true)
  tpr_vec <- c(NA, NA, NA, tpr_sparse, NA, tpr_lasso, NA, tpr_hp, NA, 1, 1)
  fpr_vec <- c(NA, NA, NA, fpr_sparse, NA, fpr_lasso, NA, fpr_hp, NA, 0, 0)
  s_hat_vec <- c(NA, NA, NA, shat_sparse, NA, shat_lasso, NA, shat_hp, NA, q, q)

  results_numeric <- data.frame(
    Rep = rep(rep_idx, length(method_names)),
    Scenario = rep(paste0("a=", a), length(method_names)),
    Method = method_names,
    Used_Dims = used_dims_vec,
    Lambda = lambda_num_vec,
    ARI = ari_vec,
    TPR = tpr_vec,
    FPR = fpr_vec,
    S_hat = s_hat_vec,
    stringsAsFactors = FALSE
  )

  results_display <- data.frame(
    Rep = rep(rep_idx, length(method_names)),
    Scenario = rep(paste0("a=", a), length(method_names)),
    Method = method_names,
    Used_Dims = used_dims_vec,
    Penalty = c("No", "No", "No", "L1 Weights", "No", "Element L1", "No", "Adaptive Group L2", "No", "No", "No"),
    Selection = c("No", "No", "No", "Yes", "-", "Yes", "-", "Yes", "-", "Oracle Vars", "Oracle Params"),
    Lambda = ifelse(is.na(lambda_num_vec), "-", as.character(lambda_num_vec)),
    ARI = ifelse(is.na(ari_vec), "-", sprintf("%.3f", ari_vec)),
    TPR = ifelse(is.na(tpr_vec), "-", sprintf("%.3f", tpr_vec)),
    FPR = ifelse(is.na(fpr_vec), "-", sprintf("%.3f", fpr_vec)),
    S_hat = ifelse(is.na(s_hat_vec), "-", as.character(s_hat_vec)),
    stringsAsFactors = FALSE
  )

  if (print_each_run) {
    print(results_display, row.names = FALSE)

    cat("\n[튜닝 요약]\n")
    cat(sprintf("- Naive Lasso 선택 lambda: %s\n", best_lambda_lasso))
    cat(sprintf("- HP pilot 선택 lambda: %s\n", best_lambda_hp_pilot))
    cat(sprintf("- HP adaptive 선택 lambda: %s\n", best_lambda_hp))
    cat("\n[결과 해석 포인트]\n")
    cat("- Naive Lasso는 mu0와 공통 diagonal sigma2를 반영하도록 수정됨.\n")
    cat("- HP는 pilot 기반 adaptive weight와 Q-basis sum-to-zero 제약을 사용함.\n")
    cat("- Refit / Oracle-feature baseline은 common diagonal covariance 가정에 맞게 EEI를 사용함.\n")
    cat("- True-parameter oracle은 생성모수를 직접 사용한 분류 성능이며, feature-oracle baseline보다 항상 높거나 같아야 함.\n")
  }

  all_results[[store_idx]] <- results_display
  all_results_numeric[[store_idx]] <- results_numeric
  store_idx <- store_idx + 1
}
}

# ==============================================================================
# Step 8. 전체 반복 결과 종합
# ==============================================================================
cat("\n\n===================================================================================================\n")
cat(" [Step 8] 전체 반복 결과 종합\n")
cat("===================================================================================================\n")
combined_results <- do.call(rbind, all_results)
combined_results_numeric <- do.call(rbind, all_results_numeric)
print(combined_results, row.names = FALSE)
cat("===================================================================================================\n")

# ==============================================================================
# Step 9. 반복 평균 / 표준오차 요약 테이블
# ==============================================================================
cat("\n\n===================================================================================================\n")
cat(sprintf(" [Step 9] 반복 요약 테이블 (반복수 = %d)\n", n_rep))
cat("===================================================================================================\n")

summary_keys <- unique(combined_results_numeric[, c("Scenario", "Method")])
summary_keys <- summary_keys[order(summary_keys$Scenario, summary_keys$Method), ]
summary_list <- vector("list", nrow(summary_keys))

for (i in 1:nrow(summary_keys)) {
  idx <- combined_results_numeric$Scenario == summary_keys$Scenario[i] &
    combined_results_numeric$Method == summary_keys$Method[i]
  tmp <- combined_results_numeric[idx, ]

  n_used_dims <- sum(!is.na(tmp$Used_Dims))
  used_dims_mean <- if (n_used_dims > 0) mean(tmp$Used_Dims, na.rm = TRUE) else NA_real_
  used_dims_se <- if (n_used_dims > 1) sd(tmp$Used_Dims, na.rm = TRUE) / sqrt(n_used_dims) else NA_real_

  n_lambda <- sum(!is.na(tmp$Lambda))
  lambda_mean <- if (n_lambda > 0) mean(tmp$Lambda, na.rm = TRUE) else NA_real_
  lambda_se <- if (n_lambda > 1) sd(tmp$Lambda, na.rm = TRUE) / sqrt(n_lambda) else NA_real_

  n_ari <- sum(!is.na(tmp$ARI))
  ari_mean <- if (n_ari > 0) mean(tmp$ARI, na.rm = TRUE) else NA_real_
  ari_se <- if (n_ari > 1) sd(tmp$ARI, na.rm = TRUE) / sqrt(n_ari) else NA_real_

  n_tpr <- sum(!is.na(tmp$TPR))
  tpr_mean <- if (n_tpr > 0) mean(tmp$TPR, na.rm = TRUE) else NA_real_
  tpr_se <- if (n_tpr > 1) sd(tmp$TPR, na.rm = TRUE) / sqrt(n_tpr) else NA_real_

  n_fpr <- sum(!is.na(tmp$FPR))
  fpr_mean <- if (n_fpr > 0) mean(tmp$FPR, na.rm = TRUE) else NA_real_
  fpr_se <- if (n_fpr > 1) sd(tmp$FPR, na.rm = TRUE) / sqrt(n_fpr) else NA_real_

  n_shat <- sum(!is.na(tmp$S_hat))
  shat_mean <- if (n_shat > 0) mean(tmp$S_hat, na.rm = TRUE) else NA_real_
  shat_se <- if (n_shat > 1) sd(tmp$S_hat, na.rm = TRUE) / sqrt(n_shat) else NA_real_

  summary_list[[i]] <- data.frame(
    Scenario = summary_keys$Scenario[i],
    Method = summary_keys$Method[i],
    Reps = nrow(tmp),
    Used_Dims_Mean = round(used_dims_mean, 3),
    Used_Dims_SE = round(used_dims_se, 3),
    Lambda_Mean = round(lambda_mean, 3),
    Lambda_SE = round(lambda_se, 3),
    ARI_Mean = round(ari_mean, 3),
    ARI_SE = round(ari_se, 3),
    TPR_Mean = round(tpr_mean, 3),
    TPR_SE = round(tpr_se, 3),
    FPR_Mean = round(fpr_mean, 3),
    FPR_SE = round(fpr_se, 3),
    S_hat_Mean = round(shat_mean, 3),
    S_hat_SE = round(shat_se, 3),
    stringsAsFactors = FALSE
  )
}

summary_results_numeric <- do.call(rbind, summary_list)
summary_results_display <- summary_results_numeric
summary_results_display$Used_Dims <- ifelse(
  is.na(summary_results_display$Used_Dims_Mean),
  "-",
  sprintf("%.3f (%.3f)", summary_results_display$Used_Dims_Mean, summary_results_display$Used_Dims_SE)
)
summary_results_display$Lambda <- ifelse(
  is.na(summary_results_display$Lambda_Mean),
  "-",
  sprintf("%.3f (%.3f)", summary_results_display$Lambda_Mean, summary_results_display$Lambda_SE)
)
summary_results_display$ARI <- ifelse(
  is.na(summary_results_display$ARI_Mean),
  "-",
  sprintf("%.3f (%.3f)", summary_results_display$ARI_Mean, summary_results_display$ARI_SE)
)
summary_results_display$TPR <- ifelse(
  is.na(summary_results_display$TPR_Mean),
  "-",
  sprintf("%.3f (%.3f)", summary_results_display$TPR_Mean, summary_results_display$TPR_SE)
)
summary_results_display$FPR <- ifelse(
  is.na(summary_results_display$FPR_Mean),
  "-",
  sprintf("%.3f (%.3f)", summary_results_display$FPR_Mean, summary_results_display$FPR_SE)
)
summary_results_display$S_hat <- ifelse(
  is.na(summary_results_display$S_hat_Mean),
  "-",
  sprintf("%.3f (%.3f)", summary_results_display$S_hat_Mean, summary_results_display$S_hat_SE)
)
summary_results_display <- summary_results_display[, c("Scenario", "Method", "Reps", "Used_Dims", "Lambda", "ARI", "TPR", "FPR", "S_hat")]

print(summary_results_display, row.names = FALSE)
cat("===================================================================================================\n")
