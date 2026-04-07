# [연구 미팅 보고서] 고차원 데이터에서 평균 기반 이질성 유발 변수를 탐색하는 희소 혼합평균효과 기반 클러스터링 방법론

**현재 1차 범위: common-covariance 하의 mean heterogeneity pursuit**

---

## [핵심 요약] 참고 논문을 반영한 현재 버전의 위치

본 보고서는 Li et al.의 heterogeneity pursuit 논문을 직접 복제하는 것이 아니라, 그 핵심 문제의식인 “어떤 변수가 군집 간 차이를 실제로 유발하는가?”를 비지도 평균혼합모형(mean-mixture)으로 재구성한 1차 연구 구상을 담고 있다.

원 논문은 혼합회귀에서 공통효과와 군집특이효과를 분해하고, adaptive $\ell_1$ penalty 및 generalized lasso 표현을 통해 relevant predictor와 source of heterogeneity를 동시에 식별한다. 반면 본 연구의 현재 범위는 outcome이 없는 평균혼합모형에서 군집 간 평균 차이를 유발하는 좌표(mean-heterogeneity-driving coordinates)를 식별하는 데 초점을 둔다. 따라서 본 연구는 원 논문의 직접적 비지도화라기보다, 그 문제의식을 mean-shift clustering으로 옮긴 방법론이라고 이해하는 가장 정확하다.

현재 버전의 핵심 개선 사항은 다음과 같다.

- **변수 단위 선택의 명확성 확보:** 기존 prototype의 element-wise $\ell_1$ penalty는 하나의 변수 안에서도 일부 군집 효과만 0이 되는 파편화를 유발할 수 있었다. 현재는 변수 $k$에 대응하는 군집 편차 벡터 전체 $\delta_{\cdot k}$에 group penalty를 부여하여, “어떤 변수 전체가 mean heterogeneity를 유발하는가”를 직접적으로 추적하도록 정리하였다.
    
- **식별성 제약의 안정화:** 기존의 $\sum_j \pi_j \delta_j = 0$형 제약 대신, 아래와 같은 sum-to-zero constraint를 사용하여 ANOVA식 effects parameterization에 더 가깝게 만들었다. 이는 원 논문에서의 effects-model parameterization과 문제의식상 대응된다.
    
    $$\sum_{j=1}^K \delta_{jk} = 0, \quad k=1,\dots,p$$
    
- **제약을 보존하는 재파라미터화:** 직교여공간 basis $Q$를 사용하여 재파라미터화하면, 제약을 자동으로 만족하면서 희소성을 깨지 않는 안정적인 최적화가 가능하다.
    
    $$\delta_{\cdot k} = Q\alpha_k$$
    
- **adaptive penalty의 필요성 명시:** 원 논문은 nonadaptive estimator보다 adaptive estimator가 selection consistency와 낮은 오선택률 측면에서 더 우수함을 보였다. 따라서 현재 연구미팅용 HP 결과는 nonadaptive pilot으로 이해하고, 최종 논문에서는 adaptive group penalty를 기본형으로 정리하는 것이 더 적절하다.
    
- **현재 시뮬레이션 결과의 위상 정리:** 현재 표는 각 신호 강도에 대한 대표 실행 결과에 해당한다. 정식 논문에서는 원 논문처럼 반복 Monte Carlo 설계, 평균 및 표준오차 보고, correlated predictors와 unequal mixing 보조실험까지 포함하는 방향으로 확장하는 것이 바람직하다.
    

---

## Part I. 이론 및 모형

### 1. 연구배경 및 문제의식

혼합모형 기반 회귀에서는 단순히 중요한 설명변수를 찾는 것만으로 충분하지 않고, 그중에서도 실제로 군집 간 차이를 만들어내는 변수, 즉 source of heterogeneity를 구분하는 것이 더 해석가능하고 더 간명한 모형을 만든다. Li et al.는 predictor effect를 공통효과와 군집특이효과로 분해하고, relevant predictor 집합 $S_R$와 heterogeneity-driving predictor 집합 $S_H$를 동시에 식별하는 regularized finite mixture effects regression을 제안하였다. 특히 이 논문은 component variance가 다를 때 raw effect와 scaled effect를 구분해야 한다는 점을 분명히 하였고, adaptive penalty, generalized lasso, generalized EM, Bregman coordinate descent, BIC tuning, fixed $p, m$ 이론, 그리고 high-dimensional extension·cluster learning·multivariate extension을 후속 과제로 제시했다는 점에서 본 연구의 직접적인 이론적 동기다.

그러나 비지도학습, 특히 고차원 클러스터링에서는 이와 같은 “이질성의 원천 추적”이 상대적으로 덜 정식화되어 있다. 기존 sparse clustering이나 model-based clustering은 주로 군집 복원 또는 변수선택에 초점을 맞추는 경우가 많고, 군집 평균을 공통 부분과 군집특이 부분으로 분해하여 어떤 좌표가 mean heterogeneity를 유발하는지를 직접 추적하는 effects-style parameterization은 상대적으로 부족하다.

본 연구는 이러한 문제의식을 비지도 평균혼합모형으로 확장한다. 즉, 반응변수 $Y_i$가 없는 상황에서 군집 평균을 latent mean structure로 보고, 이를 공통 평균 파라미터와 군집특이 편차로 분해하여 “어떤 변수들이 군집 간 평균 차이를 만들어내는가”를 직접 추적하는 클러스터링 방법론을 개발하고자 한다. 다만 현재 1차 연구 범위는 “모든 형태의 cluster-forming variable”이 아니라, 공통 공분산 구조 하에서 mean shift를 통해 군집 분리를 유발하는 좌표를 식별하는 데 한정된다. 분산 차이 또는 상관구조 차이만으로 군집이 갈리는 경우는 현재 baseline model의 범위 밖에 있다. 이는 원 논문이 covariance structure의 차이가 heterogeneity 정의 자체를 복잡하게 만든다고 지적한 맥락과도 일치한다.

### 2. 원 논문과의 직접 대응 관계

원 논문의 기본 모형은 혼합회귀에서 공통효과 $\beta_0$와 군집특이효과 $\beta_j$를 분해하는 것이다.

$$y = x^\top \beta_0 + x^\top \beta_j + \varepsilon_j$$

이에 대응하여 본 연구는 비지도 setting에서 군집 평균을 다음과 같이 분해한다.

$$\mu_j = \mu_0 + \delta_j$$

즉, 원 논문에서의 공통효과는 현재 모형에서 grand mean parameter $\mu_0$에 대응하고, 원 논문에서의 cluster-specific effects는 현재 모형의 mean deviation $\delta_j$에 대응한다. 다만 원 논문의 $S_R$–$S_H$ 구조를 현재 1차 비지도 모형이 그대로 재현하는 것은 아니다. 현재 모형은 사실상 $S_H$-유사 객체, 즉 mean heterogeneity를 유발하는 좌표 집합만을 직접 식별한다. 이 점을 명확히 하는 것이 연구 범위를 선명하게 만든다.

또한 원 논문은 “heterogeneity pursuit의 이점은 모든 relevant predictor가 heterogeneous한 경우보다, 일부 predictor만 heterogeneity를 유발하는 경우에 더 크게 나타난다”고 보였다. 이 직관은 현재 비지도 setting에도 그대로 중요하다. 즉, mean heterogeneity support가 희소할수록 heterogeneity pursuit 기반의 변량축소와 변수식별 이점이 더 크게 나타날 가능성이 높다.

### 3. 연구목표

본 연구의 1차 목표는 다음과 같다.

1. 고차원 데이터에서 군집 구조를 추정하면서 동시에 군집 간 평균 차이를 유발하는 변수 집합을 식별하는 새로운 비지도 혼합모형을 제안한다.
    
2. 원 논문의 effects-model parameterization을 비지도 setting에 맞게 재해석하여, 군집 평균을 $\mu_j = \mu_0 + \delta_j$ 형태로 분해하는 parsimonious mixture mean-effects model을 구축한다.
    
3. $p \gg n$ 환경에서 support recovery와 mean structure estimation error에 대한 이론적 보장을 우선적으로 제시하고, 추가로 군집 성능에 대해서는 separation-dependent clustering bound 또는 Bayes rule 대비 excess risk consistency를 목표로 한다. 원 논문의 이론은 fixed $p, m$에서의 estimation/selection consistency에 초점을 두고 있으므로, 본 연구의 박사논문 기여는 그 문제의식을 비지도 고차원 setting으로 확장하는 데 있다.
    

### 4. 핵심 연구질문

본 연구는 다음 질문에 답하는 것을 목표로 한다.

- **Q1.** 비지도 혼합모형에서 mean heterogeneity의 source를 어떻게 엄밀히 정의할 것인가?
    
- **Q2.** 군집 추정과 mean-heterogeneity variable selection을 동시에 수행하는 정규화 mixture model은 어떻게 설계할 것인가?
    
- **Q3.** 고차원 환경에서 이 방법의 support recovery와 parameter error bound를 어떻게 보일 것인가?
    
- **Q4.** 분산 구조가 달라질 때 heterogeneity의 정의를 어떻게 조정할 것인가?
    
- **Q5.** adaptive penalty, 군집 수 선택, correlated predictors, unequal mixing 등 실질적 요소를 포함할 때도 heterogeneity pursuit의 이점이 유지되는가?
    

### 5. 제안모형

#### 5.1 기본 모형

관측치 $X_i=(X_{i1},\dots,X_{ip})^\top \in \mathbb{R}^p$, 잠재 군집 $Z_i \in {1,\dots,K}$에 대하여 다음 baseline model을 제안한다.

$$P(Z_i=j) = \pi_j, \quad j=1,\dots,K$$

$$X_i \mid Z_i=j \sim N_p(\mu_j,\Sigma)$$

$$\mu_j = \mu_0 + \delta_j, \quad \sum_{j=1}^K \delta_{jk}=0, \quad k=1,\dots,p$$

여기서 $\mu_0 \in \mathbb{R}^p$는 sum-to-zero coding 하의 grand mean parameter이고, $\delta_j \in \mathbb{R}^p$는 군집 $j$의 mean deviation vector이다. 따라서 각 군집의 중심은 $\mu_j = E(X_i \mid Z_i=j) = \mu_0 + \delta_j$로 표현된다.

중요한 점은, 현재 제약 $\sum_{j=1}^K \delta_{jk}=0$ 하에서 $\mu_0$는 일반적으로 marginal population mean과 동일하지 않다는 것이다. 실제로

$$E(X_i) = \sum_{j=1}^K \pi_j\mu_j = \mu_0 + \sum_{j=1}^K \pi_j\delta_j$$

이므로, $\mu_0$는 $\pi_j$가 모두 같거나 $\sum_j \pi_j\delta_j=0$인 특수한 경우에만 marginal mean과 일치한다. 따라서 본 연구에서 $\mu_0$는 “전체 평균”이라기보다 effects-style parameterization에서의 기준점 역할을 하는 grand mean parameter로 해석하는 것이 정확하다.

#### 5.2 mean heterogeneity의 정의와 parsimonious 구조

변수 $k$에 대하여 $\delta_{\cdot k} = (\delta_{1k},\dots,\delta_{Kk})^\top$라 두면, 평균 기반 이질성을 유발하는 변수 집합을 다음과 같이 정의한다.

$$S_H = \{k: \|\delta_{\cdot k}\|_2 \neq 0\}$$

즉, $\delta_{1k}=\cdots=\delta_{Kk}=0$이면 변수 $k$는 모든 군집에서 평균이 동일하므로 군집 간 mean difference를 유발하지 않는다. 반대로 $\|\delta_{\cdot k}\|_2 > 0$이면 변수 $k$는 적어도 하나의 군집에서 평균 차이를 만들어내므로 mean-heterogeneity-driving variable이다.

이 정의 아래에서 mean structure의 자유도는 full mixture mean model의 $Kp$개에서, heterogeneity-support 크기 $s=|S_H|$가 작을 때 $p + (K-1)s$ 수준으로 줄어든다. 즉, 모든 좌표마다 $K$개의 군집 평균을 따로 추정하는 대신, 대부분의 좌표에서는 공통 평균 $\mu_{0k}$만 남기고 소수의 활성 좌표에서만 군집특이 편차를 허용하게 된다. 이는 원 논문이 혼합회귀에서 $mp$를 $p_0 + (m-1)p_{00}$으로 줄이는 parsimonious 해석을 제시한 것과 대응되는 장점이다.

#### 5.3 scaled mean heterogeneity의 가능성

원 논문은 component variance가 다를 때 raw source와 scaled source of heterogeneity를 구분한다. 현재 1차 baseline에서는 공통 $\Sigma$를 가정하므로 raw mean heterogeneity와 scaled mean heterogeneity의 차이가 사라진다. 그러나 향후 $\Sigma_j = \mathrm{diag}(\sigma_{j1}^2,\dots,\sigma_{jp}^2)$와 같은 unequal diagonal variance를 허용한다면, 대각공분산 setting에서의 자연스러운 확장으로

$$\eta_{jk} = \frac{\mu_{0k}+\delta_{jk}}{\sigma_{jk}}$$

를 정의하고,

$$S_H^{\mathrm{scaled}} = \left\lbrace k: (\eta_{1k},\dots,\eta_{Kk})^\top \neq c\mathbf{1}_K, \forall c\in\mathbb{R} \right\rbrace$$

와 같은 scaled mean heterogeneity 개념을 도입할 수 있다. centered-data 표현에서 $\mu_{0k}=0$이면 이는 $\delta_{jk}/\sigma_{jk}$ 비교와 동일해진다. 이 정의는 원 논문의 scaled source 개념을 평균혼합모형으로 옮긴 직접적인 확장이다.
#### 5.4 공분산 구조: 왜 diagonal covariance부터 시작하는가

본 연구의 1차 모델과 초기 시뮬레이션에서는

$$\Sigma_j = \Sigma = \mathrm{diag}(\sigma_1^2,\dots,\sigma_p^2)$$

또는 가장 단순하게 $\Sigma = I_p$를 가정한다. 이 경우 군집이 주어졌을 때 좌표들이 조건부 독립이므로 mean heterogeneity selection 문제를 가장 선명하게 분리할 수 있다. 이는 “실제 데이터가 반드시 독립이다”라는 뜻이 아니라, 1차 단계에서 mean structure 자체를 먼저 정교하게 분석하기 위한 working model이다.

### 6. 추정방법

#### 6.1 정규화된 목적함수

모수 $\Theta = (\pi_1,\dots,\pi_K,\mu_0,\delta_1,\dots,\delta_K,\Sigma)$에 대해 다음과 같은 normalized penalized log-likelihood를 고려한다.

$$\mathcal{L}_n(\Theta) = \frac{1}{n}\sum_{i=1}^n \log\left[ \sum_{j=1}^K \pi_j \phi_p(X_i;\mu_0+\delta_j,\Sigma) \right] - \lambda_n \sum_{k=1}^p w_k \|\delta_{\cdot k}\|_2$$

여기서 $w_k$는 adaptive weight이며, 예를 들어 pilot estimator $\tilde{\delta}$를 이용해 $w_k = (|\tilde{\delta}_{\cdot k}|_2 + \varepsilon)^{-\gamma}$로 둘 수 있다.

현재 연구미팅용 pilot 구현은 사실상 $w_k \equiv 1$인 nonadaptive group penalty에 가깝다. 그러나 원 논문에서 adaptive version이 selection consistency와 낮은 오선택률 측면에서 더 우수한 것으로 나타났으므로, 최종 논문에서는 adaptive group penalty를 주모형으로 채택하는 것이 더 자연스럽다.

#### 6.2 식별성 제약

$\mu_j = \mu_0 + \delta_j$만으로는 $\mu_0$와 $\delta_j$의 분해가 유일하지 않다. 따라서 아래와 같은 sum-to-zero constraint가 필요하다. 이는 원 논문에서의 effects-model parameterization과 동일한 역할을 하는 식별성 제약이다.

$$\sum_{j=1}^K \delta_{jk}=0, \quad k=1,\dots,p$$

#### 6.3 계산 알고리즘

계산은 generalized EM 형태로 정리할 수 있다. E-step에서는 책임도를 계산한다.

$$\tau_{ij} = P(Z_i=j\mid X_i,\Theta) = \frac{ \pi_j \phi_p(X_i;\mu_0+\delta_j,\Sigma) }{ \sum_{\ell=1}^K \pi_\ell \phi_p(X_i;\mu_0+\delta_\ell,\Sigma) }$$

M-step에서는 $\pi_j,\Sigma,\mu_0,\delta_j$를 갱신한다. 특히 $\Sigma$가 diagonal일 때, 각 변수 $k$에 대한 군집편차 블록은 거의 분리되어 다음과 같은 문제로 귀결된다.

$$\min_{\mu_{0k}, \delta_{\cdot k}} \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^K \tau_{ij}\sigma_k^{-2}(x_{ik}-\mu_{0k}-\delta_{jk})^2 + \lambda_n w_k \|\delta_{\cdot k}\|_2$$

subject to $\sum_{j=1}^K \delta_{jk}=0$

실제 구현에서는 $\mathbf{1}_K$의 직교여공간 basis $Q$를 이용해 $\delta_{\cdot k} = Q\alpha_k$로 재파라미터화하면 제약이 사라져 group-regularized block update 문제로 바뀐다. 이는 원 논문이 linearly constrained penalized least squares를 generalized EM과 Bregman coordinate descent로 푼 것과 구조적으로 대응되며, 현재 모형에서는 mean-mixture setting에 맞는 blockwise group-thresholding 알고리즘으로 구현할 수 있다.

#### 6.4 튜닝 파라미터 선택

연구미팅 단계의 pilot simulation에서는 $K=3$을 고정하고 $\lambda_n$을 수동 또는 제한된 grid에서 조정하였다. 그러나 최종 논문에서는 원 논문처럼 $K$와 $\lambda_n$을 BIC 기반으로 공동 선택하는 체계를 채택하는 것이 타당하다. 또한 group-penalized mean-effects model에 맞는 자유도 근사식을 별도로 정리할 필요가 있다.

### 7. 이론적 연구목표

원 논문은 fixed $p, m$ setting에서 nonadaptive estimator의 $\sqrt{n}$-consistency와 adaptive estimator의 selection consistency를 보였다. 본 연구는 그 문제의식을 비지도 high-dimensional setting으로 확장하되, 현재 단계에서 다음과 같은 목표를 설정한다.

1. **식별성:** label switching을 제외하면 $(\pi,\mu_0,\Delta,\Sigma)$가 유일하게 식별됨을 보인다.
    
2. **추정오차 경계:** 희소도 $s=|S_H|$에 대해 다음과 같은 형태의 오차 경계를 목표로 한다.
    
    $$\|\hat{\Delta}-\Delta^*\|_F = O_p\left( \sqrt{\frac{sK\log p}{n}} \right)$$
    
3. **support recovery:** 적절한 beta-min 조건 $\min_{k\in S_H}\|\delta_{\cdot k}^*\|_2 \ge c\lambda_n$ 하에서 $P(\hat{S}_H = S_H) \to 1$을 목표로 한다.
    
4. **clustering performance:** 현재 baseline에서는 두 종류의 결과 중 하나를 목표로 하는 것이 적절하다.
    
    - $R(\hat{g}) - R(g^*) \to 0$ 형태의 excess classification risk consistency
        
    - 또는 $\Delta_{\min,n}^2 = \min_{j\neq \ell} \sum_{k\in S_H} \frac{(\delta_{jk}^*-\delta_{\ell k}^*)^2}{\sigma_k^2} \to \infty$와 같은 stronger separation regime 하에서의 misclustering bound.
        

반대로 component overlap이 존재하는 고정 separation 상황에서는 Bayes classifier 자체도 양의 오분류율을 가질 수 있으므로, 무조건적인 zero-misclustering consistency를 전면에 내세우는 것은 적절하지 않다.

기본 가정의 예로는 다음 등을 둘 수 있다.

$$\pi_j^* \ge \pi_{\min}>0, \quad 0 < c_\sigma \le \sigma_k^2 \le C_\sigma < \infty, \quad s\log p = o(n)$$

---

## Part II. 연속형 혼합모형 시뮬레이션 결과

본 절의 시뮬레이션은 제안 모형이 “모든 형태의 cluster-forming variable”을 찾는지 검증하는 것이 아니라, 공통 공분산 구조 하에서 군집 간 평균 차이를 유발하는 변수(mean-heterogeneity-driving variables)를 얼마나 정확히 식별하는지, 그리고 그러한 선택이 실제 군집 성능 개선으로 이어지는지를 경험적으로 확인하는 데 목적이 있다.

### 1. 실험 목적

1. 제안 모형이 mean-heterogeneity variable selection을 얼마나 안정적으로 수행하는가.
    
2. 과거 prototype인 Naive Lasso와 대표적 비지도 벤치마크인 Sparse K-means가 어떤 한계를 보이는가.
    
3. HP+refit 파이프라인이 어느 신호 구간까지 near-oracle empirical behavior를 유지하는가.
    

### 2. 실험 세팅

표본 수는 $n=300$, 총 차원은 $p=20$, 군집 수는 $K=3$으로 두었다. 진짜 mean-heterogeneity 좌표는 5개이며, 변수 1–5에서만 군집 간 평균 차이가 존재한다. 각 군집의 평균 편차는 대칭 구조 $(a, 0, -a)$를 따르도록 구성하였다. 변수 6–20은 평균 차이가 없는 노이즈 좌표이며, 거리 기반 방법을 어렵게 만들기 위해 분산을 2배로 증폭하였다. 데이터 전처리는 중심화만 수행하고 변수별 추가 스케일링은 하지 않았다.

비교를 위한 GMM 및 refit 단계는 구현 편의상 mclust의 common spherical covariance 모형(EII)을 사용하였다. 신호 강도는 $a=1.8, 1.5, 1.3$의 세 구간으로 설정하였다.

### 3. 시나리오별 결과표

**[시나리오 1] 명확한 신호 환경 ($a=1.8$)**

|**방법론**|**사용 차원**|**변수 선택**|**ARI**|**TPR**|**FPR**|**S^**|
|---|---|---|---|---|---|---|
|K-means|20|No|0.438|-|-|-|
|GMM (Unpenalized)|20|No|0.409|-|-|-|
|Sparse K-means|20|Yes|0.921|1.000|1.000|20|
|$\rightarrow$ + Refit|20|-|0.409|-|-|-|
|Naive Lasso (prototype)|5|Yes|0.822|1.000|0.000|5|
|$\rightarrow$ + Refit|5|-|0.912|-|-|-|
|Proposed HP (nonadaptive pilot)|20|Yes|0.822|1.000|0.000|5|
|$\rightarrow$ Proposed HP + Refit|5|-|0.912|-|-|-|
|Oracle GMM (True Vars)|5|Ideal|0.912|1.000|0.000|5|

**[시나리오 2] 중간 신호 환경 ($a=1.5$)**

|**방법론**|**사용 차원**|**변수 선택**|**ARI**|**TPR**|**FPR**|**S^**|
|---|---|---|---|---|---|---|
|K-means|20|No|0.366|-|-|-|
|GMM (Unpenalized)|20|No|0.358|-|-|-|
|Sparse K-means|20|Yes|0.737|1.000|1.000|20|
|$\rightarrow$ + Refit|20|-|0.358|-|-|-|
|Naive Lasso (prototype)|7|Yes|0.438|1.000|0.133|7|
|$\rightarrow$ + Refit|7|-|0.406|-|-|-|
|Proposed HP (nonadaptive pilot)|20|Yes|0.469|1.000|0.000|5|
|$\rightarrow$ Proposed HP + Refit|5|-|0.761|-|-|-|
|Oracle GMM (True Vars)|5|Ideal|0.761|1.000|0.000|5|

**[시나리오 3] 약한 신호 환경 ($a=1.3$)**

(이 구간에서는 oracle GMM 자체의 ARI도 0.694까지 하락하므로 강한 overlap이 존재하는 약신호 영역이다.)

|**방법론**|**사용 차원**|**변수 선택**|**ARI**|**TPR**|**FPR**|**S^**|
|---|---|---|---|---|---|---|
|K-means|20|No|0.343|-|-|-|
|GMM (Unpenalized)|20|No|0.298|-|-|-|
|Sparse K-means|20|Yes|0.711|1.000|1.000|20|
|$\rightarrow$ + Refit|20|-|0.298|-|-|-|
|Naive Lasso (prototype)|6|Yes|0.357|1.000|0.067|6|
|$\rightarrow$ + Refit|6|-|0.336|-|-|-|
|Proposed HP (nonadaptive pilot)|20|Yes|0.332|1.000|0.067|6|
|$\rightarrow$ Proposed HP + Refit|6|-|0.336|-|-|-|
|Oracle GMM (True Vars)|5|Ideal|0.694|1.000|0.000|5|

### 4. 시뮬레이션 해석

- **첫째,** 제안 모형은 중간 신호 구간까지 mean-heterogeneity selection 측면에서 매우 안정적이다. $a=1.8$과 $a=1.5$에서 HP는 모두 TPR 1.000, FPR 0.000, $\hat{S}=5$를 보여 true signal coordinates를 정확히 복원하였다.
    
- **둘째,** Sparse K-means는 현재 구현과 thresholding rule 하에서는 가짜 희소성(fake sparsity)을 보였다.
    
- **셋째,** Naive Lasso는 대칭적이고 강한 신호에서는 우연히 잘 작동할 수 있으나, 신호가 약해지면 빠르게 붕괴한다.
    
- **넷째,** 약신호 구간에서는 selection 정확도와 clustering recovery가 분리될 수 있다. $a=1.3$에서 한두 개의 false positive만으로도 refit likelihood landscape가 크게 흔들릴 수 있다.
    
- **다섯째,** 원 논문의 broader message를 감안하면, heterogeneity pursuit의 이점은 “실제로 heterogeneity를 유발하는 좌표 수가 희소할 때” 가장 의미가 크다.
    

### 5. 종합 정리

현재 pilot simulation을 종합하면, 제안 방법은 공통 공분 구조 하에서의 mean heterogeneity selection 문제에 대해 중간 이상의 신호 구간에서 진짜 좌표를 매우 정확히 복원하며, 선택 단계의 shrinkage bias는 refit 단계에서 상당 부분 제거될 수 있음을 확인했다.

---

## Part III. 정식 논문용 추가 시뮬레이션 계획

원 논문의 시뮬레이션 및 supplementary experiments를 참고하면, 최종 논문에는 아래의 보완 실험을 포함하는 것이 좋다.

- **반복 Monte Carlo 실험:** 각 시나리오에 대해 $R=200$ 또는 $R=500$ 반복 수행
    
- **correlated predictors 실험:** $\Sigma_{uv} = 0.5^{|u-v|}$ 형태의 상관구조 도입
    
- **unequal mixing probabilities 실험:** $(\pi_1, \pi_2, \pi_3) = (0.25, 0.25, 0.5)$와 같이 불균형 mixing 도입
    
- **dense heterogeneity regime 실험:** 모든 informative coordinates가 heterogeneous한 경우 포함
    
- **adaptive vs nonadaptive HP 비교:** adaptive version의 selection 안정성 개선 검증
    
- **군집 수 선택 실험:** $K$를 고정하지 않고 BIC 또는 ICL 기반 동시 선택 설계 포함
    

### 8. 기존 연구와의 차별성

본 연구의 차별점은 단순히 “클러스터링에 유용한 변수”를 고르는 것이 아니라, 군집 평균의 좌표별 분해를 통해 **“왜 군집이 갈리는가”**를 직접 묻는다는 점에 있다.

본 연구는 원 논문의 개념을 그대로 비지도화한 것이 아니라, 그 핵심 문제의식인 “heterogeneity의 원천 추적”을 mean-shift clustering 문제로 재구성한 방법론이라고 정리하는 것이 가장 정확하다.

### 9. 후속 확장 방향

현재 1차 모형은 mean heterogeneity selection에 집중하지만, 향후 아래와 같은 구조로 더 풍부한 비지도 모형 확장이 가능하다.

$$X_i = \mu_0 + \Lambda f_i + \delta_{Z_i} + \varepsilon_i, \quad \varepsilon_i \sim N_p(0,\Psi)$$

여기서 $\Lambda f_i$는 전체 표본에 공통적인 저차원 구조를 나타내고, $\delta_{Z_i}$는 군집특이 평균구조를 나타낸다.

현재 연구는 그중 가장 기초적이면서도 핵심적인 출발점인 **“고차원 mean heterogeneity pursuit in clustering”**을 목표로 한다.
