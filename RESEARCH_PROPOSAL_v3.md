## 고차원 데이터에서 이질성 유발 변수를 탐색하는 희소 혼합평균효과 기반 클러스터링 방법론

### High-Dimensional Clustering via Sparse Mixture Mean-Effects for Heterogeneity Pursuit

### 1. 연구배경 및 문제의식

혼합모형 기반 회귀분석에서는 단순히 유의미한 설명변수를 찾는 것을 넘어, 실제로 군집 간의 차이를 유발하는 변수, 즉 이질성의 원천(source of heterogeneity)을 식별하는 것이 모형의 간명성과 해석력을 극대화하는 핵심 과제이다. 최근의 선행 연구는 예측변수(predictor)의 효과를 공통효과(common effect)와 군집특이효과(cluster-specific effect)로 분해하고, 정규화된 유한 혼합효과 회귀(regularized finite mixture effects regression)를 통해 관련 변수(relevant predictor)와 이질성 유발 변수(heterogeneity-driving predictor)를 동시에 식별하는 방법론을 제안하였다.

그러나 비지도학습, 특히 고차원 클러스터링 분야에서는 이러한 "이질성의 원천 추적"이 상대적으로 덜 정식화되어 있다. 기존의 희소 군집화(sparse clustering)나 모형 기반 군집화(model-based clustering)는 주로 군집의 복원 자체나 단순 변수 선택에 초점을 맞추고 있다. 즉, 군집을 실제로 형성하는 핵심 좌표가 무엇인지 식별하고, 이를 일관되게 추정하기 위한 효과 모형 기반의 파라미터화(effects-style parameterization) 연구는 현저히 부족한 실정이다.

본 연구는 선행 연구의 문제의식을 비지도학습 환경으로 확장한다. 반응변수 $Y_i$ 가 존재하지 않는 군집화 상황에서 군집의 중심을 잠재 평균(latent mean)으로 설정하고, 이를 공통 평균과 군집특이 편차로 분해함으로써 "어떤 변수들이 실제 군집 이질성의 원천인가"를 직접 추적하는 새로운 클러스터링 방법론을 제안하고자 한다. 이는 기존 문헌에서 향후 과제로 명시한 고차원 환경(high-dimensional setting) 및 군집 학습(cluster learning)으로의 직접적이고 논리적인 확장이다.

### 2. 연구목표

본 연구의 주요 목표는 다음과 같다.

**첫째,** 고차원 데이터 환경에서 군집 구조를 추정함과 동시에, 군집 형성에 실질적으로 기여하는 이질적 변수 집합을 식별하는 새로운 비지도 희소 혼합모형을 제안한다.

**둘째,** 기존 회귀 문헌의 효과 모형 파라미터화를 비지도학습에 맞게 재해석하여, 군집 중심을 $\mu_j = \mu_0 + \delta_j$ 형태로 분해하는 간명한(parsimonious) 모형을 구축한다.

**셋째,** 차원이 표본 수보다 훨씬 큰 $p \gg n$ 고차원 환경에서, 제안 모형의 변수 선택 일관성(selection consistency), 군집 오분류율(misclustering rate), 평균 구조 추정 오차 등에 대한 수학적/이론적 보장(theoretical guarantee)을 제시한다.

### 3. 핵심 연구질문

본 연구는 상기 목표를 달성하기 위해 다음의 핵심 질문들에 답하고자 한다.

- **Q1.** 비지도학습 환경에서 "이질성의 원천(source of heterogeneity)"을 어떻게 통계적으로 엄밀하게 정의할 것인가?
    
- **Q2.** 군집 추정과 이질성 유발 변수 선택을 동시에 수행하기 위한 정규화 혼합모형(regularized mixture model) 및 페널티 함수는 어떻게 설계할 것인가?
    
- **Q3.** 고차원 환경에서 제안된 추정량의 변수 선택 일관성과 군집화 일관성(clustering consistency)을 어떻게 이론적으로 증명할 것인가?
    
- **Q4.** 공분산(분산) 구조가 이질적인 상황으로 확장될 때, 이질성(heterogeneity)의 정의와 식별 조건을 어떻게 조정할 것인가?
    

### 4. 제안모형

#### 4.1 기본 모형 (Baseline Model)

관측치 $X_i = (X_{i1}, \dots, X_{ip})^\top \in \mathbb{R}^p$ 와 잠재 군집 라벨 $Z_i \in \{ 1, \dots, K \}$ 에 대하여 다음과 같은 혼합모형을 제안한다.

$$P(Z_i = j) = \pi_j, \quad j = 1, \dots, K$$

$$X_i \mid Z_i = j \sim N_p(\mu_j, \Sigma)$$

$$\mu_j = \mu_0 + \delta_j, \quad \sum_{j=1}^K \delta_{jk} = 0, \quad k = 1, \dots, p$$

여기서 $\mu_0 \in \mathbb{R}^p$ 는 전체 기준 평균(baseline mean)이며, $\delta_j \in \mathbb{R}^p$ 는 군집 $j$ 의 평균 편차(mean shift)이다. 각 군집의 중심은 $\mu_j = E(X_i \mid Z_i = j) = \mu_0 + \delta_j$ 로 표현된다. 이는 선행 연구의 회귀계수 분해( $\beta_{0k}, \beta_{jk}$ ) 아이디어를 비지도 설정의 잠재 평균 모수( $\mu_{0k}, \delta_{jk}$ )로 확장 적용한 것이며, 본 연구는 이 잠재 평균의 구조를 희소하게 분해하여 추정하는 데 초점을 둔다.

#### 4.2 이질적 변수의 정의

변수 $k$ 에 대하여 $\delta_{\cdot k} = (\delta_{1k}, \dots, \delta_{Kk})^\top$ 라 정의할 때, 군집 간 이질성을 유발하는 활성 변수 집합 $S_H$ 를 다음과 같이 정의한다.

$$S_H = \{ k : \|\delta_{\cdot k}\|_2 \neq 0 \}$$

즉, $\delta_{1k} = \dots = \delta_{Kk} = 0$ 이면 변수 $k$ 는 모든 군집에서 평균이 동일하므로 군집 간 차이를 유발하지 않는 노이즈 변수이다. 반대로 $\|\delta_{\cdot k}\|_2 > 0$ 이면 적어도 하나의 군집에서 유의미한 평균 차이를 만들어내므로 이질성 유발 변수(heterogeneity-driving variable)로 식별된다. 이는 기존 회귀 모형에서 군집특이효과를 지닌 예측변수를 판별하던 논리를 비지도 군집화 문제로 확장한 것이다.

#### 4.3 공분산 구조의 설정

본 연구는 모형의 개발 및 이론 전개의 1차적 단계로서 공통 대각 공분산 구조 $\Sigma_j = \Sigma = \mathrm{diag}(\sigma_1^2, \dots, \sigma_p^2)$ 를 가정한다.

이는 변수 간 조건부 독립을 가정한 가장 기본적인 형태(working model)로서, 평균 기반 이질성(mean heterogeneity) 탐색 문제 자체에 수리적으로 집중하기 위한 전략적 선택이다. 선행 연구에서도 성분 분산(component variance)의 차이가 이질성의 해석에 직접적 영향을 미치며, 다변량 확장 시 공분산 행렬의 차이가 이질성 정의를 복잡하게 만든다고 지적한 바 있다. 따라서 본 연구는 공통 대각 공분산을 통해 기본 방법론과 이론을 정립한 후, 상관관계가 존재하는 특성(correlated features)이나 이질적 분산(unequal diagonal variance) 구조로 확장해 나갈 계획이다.

### 5. 추정방법

#### 5.1 정규화된 목적함수

모수 $\Theta = (\pi_1, \dots, \pi_K, \mu_0, \delta_1, \dots, \delta_K, \Sigma)$ 에 대해 다음과 같은 페널티 벌점화 로그 우도함수(penalized log-likelihood)를 목적함수로 설정한다.

$$\ell_n(\Theta) = \sum_{i=1}^n \log \left[ \sum_{j=1}^K \pi_j \phi_p(X_i; \mu_0 + \delta_j, \Sigma) \right] - \lambda \sum_{k=1}^p w_k \|\delta_{\cdot k}\|_2$$

여기서 $w_k$ 는 Adaptive weight로, 사전 추정량(pilot estimator)을 활용하여 $w_k = (\|\tilde{\delta}_{\cdot k}\|_2 + \varepsilon)^{-\gamma}$ 와 같이 구성된다. 이 페널티 항은 변수 단위의 그룹 희소성(group sparsity)을 유도하여, 변수 $k$ 전체가 이질성의 원천인지 여부를 직접적으로 판별하게 해준다.

#### 5.2 식별성 제약 및 최적화 알고리즘

구조식 $\mu_j = \mu_0 + \delta_j$ 의 유일한 분해를 보장하기 위해 $\sum_{j=1}^K \delta_{jk} = 0$ 의 제약을 부여한다. 모수 추정은 EM 알고리즘을 기반으로 수행되며, E-step에서는 사후 책임도(responsibility) $\tau_{ij}$ 를 계산한다.

$$\tau_{ij} = P(Z_i = j \mid X_i, \Theta) = \frac{\pi_j \phi_p(X_i; \mu_0 + \delta_j, \Sigma)}{\sum_{\ell=1}^K \pi_\ell \phi_p(X_i; \mu_0 + \delta_\ell, \Sigma)}$$

M-step에서는 $\Sigma$ 가 대각 행렬임을 이용하여 각 변수 $k$ 에 대한 블록별 최적화 문제로 분리한다.

$$\min_{\mu_{0k}, \delta_{\cdot k}} \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^K \tau_{ij} \sigma_k^{-2} (x_{ik} - \mu_{0k} - \delta_{jk})^2 + \lambda w_k \|\delta_{\cdot k}\|_2 \quad \text{s.t.} \quad \sum_{j=1}^K \delta_{jk} = 0$$

실제 알고리즘 구현에서는 제약식을 내재화하기 위해 $\mathbf{1}_K$ 의 직교여공간 Basis $Q$ 를 도입하여 $\delta_{\cdot k} = Q \alpha_k$ 로 재파라미터화한다. 이를 통해 제약 조건이 없는(unconstrained) Group Lasso 문제로 변환하여 안정적이고 효율적인 최적화를 수행한다. 튜닝 파라미터 $\lambda$ 와 군집 수 $K$ 는 BIC(Bayesian Information Criterion) 등 정보량 기준을 통해 선택한다.

### 6. 이론적 연구목표

본 학위논문(연구)의 핵심 기여는 전통적인 고정 차원(fixed $p, m$)에서 증명되었던 선택 일관성(selection consistency) 이론을 $p \gg n$ 의 고차원 비지도학습 환경으로 확장 증명하는 데 있다. 이를 위한 세부 이론적 목표는 다음과 같다.

**첫째, 식별성(Identifiability):** Label switching 현상을 제외하면 모수 $(\pi, \mu_0, \Delta, \Sigma)$ 가 유일하게 식별됨을 증명한다.

**둘째, 추정오차 경계(Estimation Error Bound):** 실제 활성 변수의 희소도 $s = |S_H|$ 에 대하여 다음 형태의 오차 상한을 유도한다.

$$\|\hat{\Delta} - \Delta^*\|_F = O_p \left( \sqrt{\frac{s K \log p}{n}} \right)$$

**셋째, 변수 선택 일관성(Support Recovery):** 적절한 beta-min 조건 $\min_{k \in S_H} \|\delta_{\cdot k}^*\|_2 \ge c\lambda$ 가 주어졌을 때, 참인 변수 집합을 완벽히 복원할 확률이 1로 수렴함( $P(\hat{S}_H = S_H) \to 1$ )을 증명한다.

**넷째, 군집화 일관성(Clustering Consistency):** MAP(Maximum A Posteriori) 규칙으로 얻은 추정 라벨 $\hat{Z}_i$ 에 대하여 오분류율 상한(misclustering bound)을 도출하거나, 다음을 만족함을 증명한다.

$$\frac{1}{n} \sum_{i=1}^n I(\hat{Z}_i \neq Z_i^*) = o_p(1)$$

이러한 이론 전개를 위해 혼합 비율 및 분산의 유계성(boundedness), 강한 분리 조건(separation condition) $\min_{j \neq \ell} \sum_{k \in S_H} \sigma_k^{-2} (\delta_{jk}^* - \delta_{\ell k}^*)^2 \ge c_0$, 그리고 고차원 희소성 조건 $s \log p = o(n)$ 등을 기본 가정으로 활용할 계획이다.

### 7. 연구의 차별성 및 향후 확장 계획

본 연구의 가장 큰 차별점은 비지도 군집화 과정에서 단순히 "유용한 변수"를 선택하는 데 그치지 않고, 평균 구조의 좌표별 분해를 통해 **"무엇이 군집을 분리하게 만드는가(Why do clusters separate?)"**에 대한 근원적이고 해석 가능한 해답을 제공한다는 점이다. 이는 순수 평균-이동 군집화(pure mean-shift clustering) 문제에서 가장 통계적으로 투명하게 이질성을 추적하는 방법론이 될 것이다.

본 연구는 1차적으로 평균 이질성(mean heterogeneity) 선택 문제에 집중하여 탄탄한 이론적 토대를 마련할 것이다. 이후, 데이터 전반에 흐르는 공통 구조(common structure)와 이질적 구조를 모두 포괄하기 위해 다음과 같은 요인 모형(factor model) 형태의 확장 연구를 계획하고 있다.

$$X_i = \mu_0 + \Lambda f_i + \delta_{Z_i} + \varepsilon_i, \quad \varepsilon_i \sim N_p(0, \Psi)$$

위 모형에서 $\Lambda f_i$ 는 표본 전체에 공통으로 존재하는 저차원 구조(common low-dimensional structure)를, $\delta_{Z_i}$ 는 군집특이적 평균 구조를 나타낸다. 이러한 후속 연구를 통해 공통 기저 변수와 이질성 유발 변수를 동시에 식별하는 통합적 비지도학습 프레임워크를 완성할 수 있을 것으로 기대한다.
