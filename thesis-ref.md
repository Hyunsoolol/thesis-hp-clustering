## 핵심 선행 연구 지형 분석 및 제안 모형의 차별성

본 연구의 위치를 명확히 하고 박사 학위 논문으로서의 독창성과 안전한 포지셔닝을 확보하기 위해, 기존 문헌의 지형을 분석하고 제안 모형과의 관계를 재정리하였습니다. 본 연구의 핵심은 Grouped penalty 자체가 아니라, **$S_\Delta$를 직접 타깃으로 하는 Effects-style unsupervised mean-mixture formulation**에 있습니다.

### 1. 가장 직접적으로 겹치는 방법론 (심사 리스크가 높은 필수 비교 대상)

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

> **💡 표 1을 읽는 포인트:**
> 
> 현재 방법론의 핵심은 $\mu_j = \mu_0 + \delta_j$, $\sum_{j=1}^K \delta_{jk}=0$, $S_\Delta = \{k: \Vert\delta_{\cdot k}\Vert_2 \neq 0\}$, 그리고 $\delta_{\cdot k}$에 대한 adaptive group lasso라는 점입니다. 즉, raw cluster means를 바로 shrink하는 것이 아니라 baseline-adjusted deviation vectors를 직접 sparse target으로 둡니다.

---

#### 표 2. 유사성, 차이점, 심사 리스크, 논문 작성 시 주의점

|**비교 논문**|**현재 방법과 가장 겹치는 지점**|**결정적 차이점**|**심사 리스크**|**논문 작성 시 주의할 점**|**안전한 포지셔닝 문장**|
|---|---|---|---|---|---|
|**Wang and Zhu (2008)**|고차원 Gaussian mixture, 공통 diagonal covariance, adaptive grouped regularization, EM|Wang은 informative-variable selection이 목적이고 adaptive $L_\infty$/hierarchical penalty를 사용한다. 너는 $S_\Delta$ target, $\delta_{\cdot k}$-adaptive group lasso, $Q$-basis를 사용한다.|높음|“adaptive grouped clustering 최초”나 “global mean + cluster deviation 최초”는 절대 쓰면 안 된다.|“기존 adaptive grouped model-based clustering 위에서, baseline-adjusted deviation support를 직접 타깃으로 두는 formulation”|
|**Xie, Pan, and Shen (2008)**|HP-L과 VMG가 매우 가깝다. 둘 다 같은 변수의 여러 군집 mean을 $\ell_2$-group으로 shrink한다.|Xie는 raw $\mu_{\cdot k}$를 penalize하고, 너는 $\mu_j=\mu_0+\delta_j$ decomposition 아래 $\delta_{\cdot k}$를 penalize한다.|매우 높음|HP-L 자체를 main novelty처럼 쓰면 위험하다. HP-L은 필수 comparator로 두는 것이 안전하다.|“Xie의 raw mean grouping과 달리, 본 연구는 baseline-adjusted deviation vectors를 sparse target으로 둔다.”|
|**Guo et al. (2010)**|generic informative-variable selection을 넘어서 더 구조적인 target을 둔다는 점|Guo는 pairwise separability가 목표이고, 너는 $S_\Delta$라는 mean-heterogeneity support recovery가 목표다.|중간~높음|“구조적 target을 둔 clustering 최초”는 위험하다. Guo를 반드시 related work에 넣어야 한다.|“본 연구는 pairwise fusion이 아니라 effects-style mean-heterogeneity support를 직접 추정한다.”|
|**Li et al. (2022)**|heterogeneity pursuit, common/cluster-specific decomposition, M3/M4, adaptive vs nonadaptive, BIC, EM framing이 가장 비슷하다.|Li는 supervised finite mixture regression이고 $S_R, S_H$를 다룬다. 너는 outcome-free unsupervised mean-mixture clustering이며 $S_\Delta$를 다룬다.|개념적으로 매우 높음|“heterogeneity pursuit 최초” 또는 “effects-model parameterization 최초”는 쓰면 안 된다.|“Li et al.의 heterogeneity pursuit를 비지도 mean-mixture setting으로 재구성한 unsupervised analogue”|
|**Li et al. (2023, ZINBMM)**|cluster-specific mean과 global mean의 차이에 penalty를 준다는 점이 개념적으로 인접하다.|count/ZINB/scRNA-seq 문제이고, Gaussian mean-effects decomposition, sum-to-zero coding, $Q$-basis, adaptive group lasso on $\delta_{\cdot k}$는 없다.|중간|“global mean 대비 cluster-specific deviation shrinkage 최초”는 위험하다.|“최근 count-mixture 문헌에서도 global-vs-cluster mean shrinkage가 사용되나, 본 연구는 Gaussian mean-effects decomposition과 deviation-vector group regularization에 초점을 둔다.”|
|**현재 방법론의 self-positioning**|grouped penalty, adaptive regularization, single-stage fitting|novelty는 penalty 그 자체가 아니라 target과 formulation의 조합에 있다.|—|“first grouped clustering” 또는 “first adaptive grouped model-based clustering”은 피해야 한다.|“effects-style unsupervised heterogeneity pursuit for Gaussian mean mixtures”|

> **💡 표 2를 읽는 포인트:**
> 
> 심사에서 가장 위험한 부분은 HP-L을 독립적인 방법론적 novelty처럼 보이게 쓰는 것과, grouped/adaptive penalized clustering 자체가 새롭다고 쓰는 것입니다. 논문에서 살려야 하는 것은 penalty family 자체가 아니라, $\mu_j=\mu_0+\delta_j$, $\sum_j \delta_{jk}=0$, $S_\Delta=\{k: \Vert\delta_{\cdot k}\Vert_2 \neq 0\}$, $\delta_{\cdot k}=Q\alpha_k$ 라는 formulation과 target의 결합입니다.

---

### 4. 지도교수님 보고용 핵심 요약 (Elevator Pitch)

연구의 차별성과 심사 방어 전략을 교수님께 브리핑하실 때 활용하기 좋은 스크립트입니다.

> **[선행 문헌과의 관계 및 포지셔닝 전략]**
> 
> "문헌을 다시 정리해보니, 제 방법은 Wang and Zhu(2008), Xie et al.(2008)와 broad한 grouped penalized model-based clustering 축에서는 분명히 겹칩니다. Wang and Zhu는 adaptive grouped penalties를 이미 제안했고, Xie et al.은 raw cluster means에 대한 grouped lasso VMG를 이미 제안했습니다. 따라서 **grouped penalty 자체를 제 novelty로 쓰면 위험합니다.**
> 
> 다만 제 방법은 군집 평균을 $\mu_j = \mu_0 + \delta_j$로 분해하고, raw cluster means가 아니라 baseline-adjusted deviation vector $\delta_{\cdot k}$를 sparse target으로 두어 mean-heterogeneity-driving support $S_\Delta$를 직접 추적한다는 점에서 차별화됩니다. 따라서 제 포지셔닝은 'First grouped clustering'이 아니라, Li et al.(2022)의 heterogeneity pursuit 방식을 비지도 mean-mixture setting으로 옮긴 **'Effects-style unsupervised heterogeneity pursuit'**라고 정리하는 것이 가장 안전합니다."

> **[최종 결론 (한 줄 요약)]**
> 
> "제 방법의 Novelty는 Grouped penalty 자체가 아니라, Baseline-adjusted deviation support $S_\Delta$를 직접 타깃으로 하는 **Effects-style unsupervised mean-mixture formulation**과 그에 맞춘 **Adaptive group-lasso single-stage estimation**에 있습니다."
