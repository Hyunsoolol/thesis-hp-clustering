## 핵심 선행 연구 지형 분석 및 제안 모형의 차별성

본 연구 관련 기존 문헌을 분석하고 제안 모형과의 관계를 재정리하였습니다. 본 연구의 핵심은 Grouped penalty 자체가 아니라, **$S_\Delta$를 직접 타깃으로 하는 Effects-style unsupervised mean-mixture formulation**에 있습니다.

### 1. 가장 직접적으로 겹치는 방법론

제안 모형과 유사하게 고차원 군집화에서 Grouped/Adaptive regularization을 수행하는 핵심 선행 연구들입니다. 이 논문들과의 명확한 선 긋기가 논문 심사 방어의 핵심입니다.

- **Wang and Zhu (2008) - Variable Selection for Model-Based High-Dimensional Clustering**
    
    - **내용:** 동일한 변수에 대한 군집 평균 파라미터들을 통째로 0으로 만드는 adaptive grouped regularization을 제안했습니다.
        
    - **차별점:** Wang & Zhu는 전반적인 'Informative-variable selection'이 목적이며 Raw mean grouping을 사용합니다. 반면 본 제안 모형은 Baseline-adjusted deviation support $S_\Delta$를 직접 타깃으로 하며, $\delta_{\cdot k}$에 대한 Adaptive group lasso와 $Q$-basis 재파라미터화를 사용합니다.
        
- **Xie, Pan, and Shen (2008) - Penalized Model-Based Clustering with Grouped Variables**
    
    - **내용:** 동일한 변수의 여러 군집 파라미터를 통째로 묶어 $\ell_2$-group으로 축소하는 VMG 패널티를 제안했습니다.
        
    - **차별점:** 제안 모형의 HP-L과 구조적으로 매우 가깝기 때문에 HP-L 자체를 주요 novelty로 내세우는 것은 심사에서 취약점이 될 수 있습니다. Xie et al.은 Raw cluster mean $\mu_{\cdot k}$ 자체를 페널티의 대상으로 삼습니다. 반면 본 연구는 $\mu_j = \mu_0 + \delta_j$ 분해 구조 아래서 편차 벡터인 $\delta_{\cdot k}$를 페널티 대상으로 삼는다는 명확한 차이가 있습니다.
        

### 2. 타깃 및 구조적 접근 방식이 유사한 방법론

- **Guo et al. (2010):** 어떤 변수로 특정 군집 쌍이 분리되는지 찾는 Pairwise variable selection을 수행합니다. 변수 선택에 '구조적인 타깃'을 둔다는 철학은 공유하나, 본 연구는 Pairwise fusion이 아니라 Mean-heterogeneity support $S_\Delta$ 자체를 직접 추정합니다.
    
- **Li et al. (2022):** 지도학습(Supervised) 환경인 유한 혼합 회귀(Finite mixture regression)에서 예측 변수와 이질성 유발 변수(Heterogeneity sources)를 동시에 식별합니다. 본 연구는 이를 반응 변수(outcome)가 없는 순수 비지도(unsupervised) 환경으로 재구성한 대응 모형(unsupervised analogue)입니다.
    
- **Li et al. (2023, ZINBMM):** 카운트 데이터에서 전역 평균(Global mean)과 군집별 평균(Cluster-specific mean)의 차이에 페널티를 부여합니다. 개념적으로 인접하나 제안 모형의 Gaussian mean-effects 분해와 Sum-to-zero coding 최적화는 존재하지 않습니다.
    

---

### 3. 모형 및 방법론 상세 비교표

#### 표 1. 모형 및 방법론 비교표

| **논문 / 방법**                   | **핵심 문제**                                                                               | **기본 모형**                                                                                                                              | **주요 가정**                                                                                | **식별성 / 제약**                                                | **벌점 구조**                                                                                                                                                                                                                                                                                                   | **추정 / 튜닝**                                                                                                                                          |
| ----------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **현재 방법론**                    | 비지도 고차원 클러스터링에서 mean-heterogeneity-driving variables 식별                                 | $P(Z_i=j)=\pi_j, X_i \mid Z_i=j \sim N_p(\mu_j, \Sigma), \mu_j=\mu_0+\delta_j$                                                         | 공통 diagonal covariance 또는 $\Sigma=I_p$, mean shift 중심                                    | $\sum_{j=1}^K \delta_{jk}=0, k=1,\dots,p$                   | HP-L: $L_n^{HP-L} = \frac{1}{n}\ell_n(\Theta) - \lambda_n\sum_{k=1}^p \Vert\delta_{\cdot k}\Vert_2$<br><br>  <br>  <br><br>HP-AL: $L_n^{HP-AL} = \frac{1}{n}\ell_n(\Theta) - \lambda_n\sum_{k=1}^p w_k\Vert\delta_{\cdot k}\Vert_2$, $w_k = (\Vert\tilde{\delta}_{\cdot k}\Vert_2 + \varepsilon)^{-\gamma}$ | EM + $Q$-basis 재파라미터화 $\delta_{\cdot k} = Q\alpha_k$, heuristic BIC, $\hat{S}_\tau = \{k: \Vert\hat{\delta}_{\cdot k}\Vert_2 > \tau\}, \tau=10^{-4}$ |
| **Wang and Zhu (2008)**       | 비지도 고차원 Gaussian model-based clustering에서 informative variable selection                | $f(x_i) = \sum_{k=1}^K \pi_k f_k(x_i; \mu_k, \Sigma)$                                                                                  | 공통 diagonal covariance, centered data                                                    | 명시적 ANOVA형 sum-to-zero coding은 아님                           | L1-GMM: $\ell-\lambda\sum_{k=1}^K\sum_{j=1}^p \vert\mu_{kj}\vert$                                                                                                                                                                                                                                           | -                                                                                                                                                    |
| **Xie, Pan, and Shen (2008)** | penalized model-based clustering에서 grouped parameters를 통한 variable selection            | $f(x_j; \Theta) = \sum_{i=1}^g \pi_i f_i(x_j; \theta_i)$                                                                               | standardization된 data, 공통 diagonal covariance, 확장으로 cluster-specific diagonal covariance | raw cluster means 중심, effects coding 없음                     | VMG: $p_\lambda(\Theta) = \lambda \sum_{k=1}^K \Vert\mu_{\cdot k}\Vert_2$<br><br>  <br>  <br><br>HMG: $p_\lambda(\Theta) = \lambda \sum_{i=1}^g \sum_{m=1}^M k_m \Vert\mu_{im}\Vert_2$                                                                                                                      | EM + grouped mean update, Newton-type solver, BIC                                                                                                    |
| **Guo et al. (2010)**         | 어떤 변수로 어떤 cluster pair가 분리되는지 찾는 pairwise variable selection                            | $f(x_i) = \sum_{k=1}^K w_k \phi(x_i; \mu_k, \Sigma)$                                                                                   | 공통 diagonal covariance, mean differences 중심                                              | zero가 특별한 기준이 아님, pairwise fusion 중심                        | PFP: $\ell-\lambda\sum_{j=1}^p\sum_{1\le k<k'\le K} \vert\mu_{k,j}-\mu_{k',j}\vert$                                                                                                                                                                                                                         | -                                                                                                                                                    |
| **Li et al. (2022)**          | supervised finite mixture regression에서 relevant predictors와 heterogeneity sources 동시 식별 | $f(y \mid x, \theta) = \sum_{j=1}^m \pi_j \frac{\rho_j}{\sqrt{2\pi}} \exp\{-\frac{1}{2}(\rho_j y - x^\top\beta_0 - x^\top\beta_j)^2\}$ | supervised mixture regression, scale differences 중요                                      | $\sum_{j=1}^m \beta_{jk} = 0, k=1,\dots,p$                  | $\ell_{\lambda\gamma}(\theta) = \sum_{i=1}^n \log f(y_i \mid x_i, \theta) - n\lambda\sum_{k=1}^p P_\gamma(\tilde{\beta}_k)$<br><br>  <br>  <br><br>$P_\gamma(\tilde\beta_k)=\sum_{j=0}^m w_{jk} \vert\beta_{jk}\vert$                                                                                       | -                                                                                                                                                    |
| **Li et al. (2023, ZINBMM)**  | scRNA-seq에서 clustering과 gene selection 동시 수행                                            | $\sum_{k=1}^K p_k f_{ZINB}(X_{ij}; \pi_{jk}, \mu_{ijk}, \phi_j)$<br><br>  <br><br>$\log\mu_{ijk} = \beta_{jk} + B_i^\top\gamma$        | count data, dropout, batch effects, zero inflation                                       | global mean $\beta_j^*$는 baseline 역할, sum-to-zero coding 없음 | $\ell_p(\theta)=\ell(\theta)-\eta\sum_j\sum_k \vert\beta_{jk}-\beta_j^*\vert$                                                                                                                                                                                                                               | -                                                                                                                                                    |

> 제안하는 방법론의 핵심은 $\mu_j = \mu_0 + \delta_j$, $\sum_{j=1}^K \delta_{jk}=0$, $S_\Delta = \{k: \Vert\delta_{\cdot k}\Vert_2 \neq 0\}$, 그리고 $\delta_{\cdot k}$에 대한 adaptive group lasso라는 점입니다. 즉, raw cluster means를 바로 shrink하는 것이 아니라 baseline-adjusted deviation vectors를 직접 sparse target으로 둡니다.

---

#### 표 2. 유사성, 차이점

| **비교 논문**                     | **현재 방법과 가장 겹치는 지점**                                                                                                    | **결정적 차이점**                                                                                                                                                                |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Wang and Zhu (2008)**       | 고차원 Gaussian mixture, 공통 diagonal covariance, adaptive grouped regularization, EM                                       | Wang은 informative-variable selection이 목적이고 adaptive $L_\infty$/hierarchical penalty를 사용한다. 너는 $S_\Delta$ target, $\delta_{\cdot k}$-adaptive group lasso, $Q$-basis를 사용한다. |
| **Xie, Pan, and Shen (2008)** | HP-L과 VMG가 매우 가깝다. 둘 다 같은 변수의 여러 군집 mean을 $\ell_2$-group으로 shrink한다.                                                    | Xie는 raw $\mu_{\cdot k}$를 penalize하고, 너는 $\mu_j=\mu_0+\delta_j$ decomposition 아래 $\delta_{\cdot k}$를 penalize한다.                                                           |
| **Guo et al. (2010)**         | generic informative-variable selection을 넘어서 더 구조적인 target을 둔다는 점                                                        | Guo는 pairwise separability가 목표이고, 너는 $S_\Delta$라는 mean-heterogeneity support recovery가 목표다.                                                                                |
| **Li et al. (2022)**          | heterogeneity pursuit, common/cluster-specific decomposition, M3/M4, adaptive vs nonadaptive, BIC, EM framing이 가장 비슷하다. | Li는 supervised finite mixture regression이고 $S_R, S_H$를 다룬다. 너는 outcome-free unsupervised mean-mixture clustering이며 $S_\Delta$를 다룬다.                                        |
| **Li et al. (2023, ZINBMM)**  | cluster-specific mean과 global mean의 차이에 penalty를 준다는 점이 개념적으로 인접하다.                                                     | count/ZINB/scRNA-seq 문제이고, Gaussian mean-effects decomposition, sum-to-zero coding, $Q$-basis, adaptive group lasso on $\delta_{\cdot k}$는 없다.                             |
| **현재 방법론의 self-positioning**  | grouped penalty, adaptive regularization, single-stage fitting                                                          | novelty는 penalty 그 자체가 아니라 target과 formulation의 조합에 있다.                                                                                                                    |


