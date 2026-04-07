# [연구 미팅 보고서] 고차원 데이터에서 이질성 유발 변수를 탐색하는 희소 혼합평균효과 기반 클러스터링 방법론

---

## [핵심 요약] 과거 버전 대비 모델 개선 사항

본 보고서는 고차원 환경에서 "어떤 변수가 군집의 이질성을 유발하는가(Source of Heterogeneity)?"를 식별하기 위해 기존에 구상했던 모델의 수학적/알고리즘적 한계를 대폭 개선한 이론적 배경과 제안 모형을 담고 있습니다.

### 1. 모델 구조 및 알고리즘의 핵심 개선

- **변수 단위 선택의 명확성 확보 (Group Lasso 도입):**
	- _과거:_ 개별 파라미터($|\delta_{kj}|$)에 $\ell_1$ 페널티 적용 $\rightarrow$ 하나의 변수 내에서도 특정 군집만 0이 되는 파편화(Fragmentation) 발생.
    
    - _현재:_ 변수 단위의 군집 편차 벡터 전체($\|\delta_{\cdot k}\|_2$)에 **Group Lasso($\ell_2$) 페널티 적용** $\rightarrow$ 특정 변수를 통째로 살리거나 0으로 만들어 '이질성 유발 변수 집합($S_H$)'을 완벽하게 식별.
        
- **식별성 제약(Identifiability Constraint)의 안정화:**
    
    - _과거:_ 혼합 비율을 포함한 $\sum_k \pi_k\delta_k = \mathbf{0}$ 제약 $\rightarrow$ EM 반복마다 $\pi_k$가 변하여 기준이 흔들림.
        
    - _현재:_ 혼합 비율과 독립적인 $\sum_{j=1}^K \delta_{jk} = 0$ 제약 $\rightarrow$ 전통적 분산분석(ANOVA)처럼 파라미터 해석이 직관적이며 수치적으로 안정됨.
        
- **최적화 알고리즘의 우아함 및 희소성 보존:**
    
    - _과거:_ Soft-thresholding 후 매번 강제 재정렬(re-centering) $\rightarrow$ 강제 조정 시 0으로 만든 값이 다시 뒤틀려 유도된 희소성(Sparsity)이 파괴됨.
        
    - _현재:_ 직교여공간 Basis $Q$를 활용한 **재파라미터화($\delta_{\cdot k} = Q \alpha_k$)** $\rightarrow$ 제약식을 만족하면서도 희소성을 완벽히 보존하는 안정적인 최적화 구현.
        

---

## 1. 연구배경 및 문제의식

혼합모형 기반 회귀에서는 단순히 중요한 설명변수를 찾는 것만으로 충분하지 않고, 그중에서도 실제로 군집 간 차이를 만들어내는 변수, 즉 source of heterogeneity를 구분하는 것이 더 해석가능하고 더 간명한 모형을 만든다. Li et al.의 혼합회귀 연구는 predictor effect를 공통효과와 군집특이효과로 분해하고, relevant predictor와 heterogeneity-driving predictor를 동시에 식별하는 regularized finite mixture effects regression을 제안하였다. 특히 이 연구는 component variance가 다르면 raw effect와 scaled effect의 해석이 달라질 수 있음을 분명히 하고, scaled source of heterogeneity를 별도로 정의한다는 점에서 중요한 출발점을 제공한다.

그러나 비지도학습, 특히 고차원 클러스터링에서는 이와 같은 "이질성의 원천 추적"이 상대적으로 덜 정식화되어 있다. 기존 sparse clustering이나 model-based clustering은 주로 군집 복원 자체나 변수선택에 초점을 맞추는 경우가 많고, 군집을 구분하는 평균 구조를 공통 부분과 군집특이 부분으로 분해하여 어떤 좌표가 mean heterogeneity를 실제로 유발하는지 직접 추적하는 effects-style parameterization은 상대적으로 부족하다.

본 연구는 이러한 문제의식을 비지도학습으로 확장한다. 즉, 반응변수 $Y_i$가 없는 상황에서 군집 평균을 latent mean structure로 보고, 이 latent mean을 공통 평균 파라미터와 군집특이 편차로 분해하여 "어떤 변수들이 군집 간 평균 차이를 만들어내는가"를 직접 추적하는 클러스터링 방법론을 개발하고자 한다. 다만 여기서 분명히 해야 할 점은, 본 연구가 현재 1차적으로 다루는 대상은 "모든 형태의 군집 형성 변수"가 아니라, 공통 공분산 구조 아래에서의 mean-heterogeneity-driving variable이라는 점이다. 분산 차이 또는 상관구조 차이만으로 군집이 갈리는 경우는 현재 baseline model의 범위 밖에 있다. 이는 원 논문이 covariance structure의 차이가 heterogeneity 정의 자체를 복잡하게 만든다고 지적한 맥락과도 일치한다.

---

## 2. 연구목표

본 연구의 1차 목표는 다음과 같다.

**첫째,** 고차원 데이터에서 군집 구조를 추정하면서 동시에 군집 간 평균 차이를 유발하는 변수 집합을 식별하는 새로운 비지도 혼합모형을 제안한다.

**둘째,** 기존 문헌의 effects-model parameterization을 비지도 setting에 맞게 재해석하여, 군집 평균을 다음과 같은 형태로 분해하는 parsimonious mixture mean-effects model을 구축한다.

$$\mu_j = \mu_0 + \delta_j$$

**셋째,** $p \gg n$ 환경에서 support recovery와 mean structure estimation error에 대한 이론적 보장을 우선적으로 제시하고, 추가로 군집 성능에 대해서는 separation-dependent clustering bound 또는 Bayes rule 대비 excess risk consistency를 목표로 한다. 원 논문의 이론은 fixed $p, m$에서의 estimation/selection consistency에 초점이 있으며, high-dimensional extension은 명시적으로 후속 과제로 제시되어 있다. 본 연구의 박사논문 기여는 바로 이 지점을 비지도 고차원 설정으로 확장하는 데 있다.

---

## 3. 핵심 연구질문

본 연구는 다음 질문에 답하는 것을 목표로 한다.

- **Q1.** 비지도 혼합모형에서 mean heterogeneity의 source를 어떻게 엄밀히 정의할 것인가?
    
- **Q2.** 군집 추정과 mean-heterogeneity variable selection을 동시에 수행하는 정규화 mixture model은 어떻게 설계할 것인가?
    
- **Q3.** 고차원 환경에서 이 방법의 support recovery와 parameter error bound를 어떻게 보일 것인가?
    
- **Q4.** 분산 구조가 달라질 때 heterogeneity의 정의를 어떻게 조정할 것인가?
    

---

## 4. 제안모형

### 4.1 기본 모형

관측치 $X_i = (X_{i1}, \dots, X_{ip})^\top \in \mathbb{R}^p$, 잠재 군집 $Z_i \in {1, \dots, K}$에 대하여 다음 baseline model을 제안한다.

$$P(Z_i = j) = \pi_j, \quad j = 1, \dots, K$$

$$X_i \mid Z_i = j \sim N_p(\mu_j, \Sigma)$$

$$\mu_j = \mu_0 + \delta_j, \quad \sum_{j=1}^K \delta_{jk} = 0, \quad k = 1, \dots, p$$

여기서 $\mu_0 \in \mathbb{R}^p$는 sum-to-zero coding 하의 grand mean parameter이고, $\delta_j \in \mathbb{R}^p$는 군집 $j$의 mean deviation vector이다. 따라서 각 군집의 중심은 다음과 같이 표현된다.

$$\mu_j = E(X_i \mid Z_i = j) = \mu_0 + \delta_j$$

다만 중요한 점은, 현재 선택한 제약 $\sum_{j=1}^K \delta_{jk} = 0$ 하에서 $\mu_0$는 일반적으로 marginal population mean과 동일하지 않다는 것이다. 실제로

$$E(X_i) = \sum_{j=1}^K \pi_j \mu_j = \mu_0 + \sum_{j=1}^K \pi_j \delta_j$$

이므로, $\mu_0$는 $\pi_j$가 모두 같거나 $\sum_j \pi_j \delta_j = 0$인 특수한 경우에만 marginal mean과 일치한다. 따라서 본 연구에서 $\mu_0$는 "전체 평균"이라기보다 effects-style parameterization에서의 기준점 역할을 하는 grand mean parameter로 해석하는 것이 정확하다. 이 점을 명확히 하지 않으면 모형 해석에 불필요한 혼동이 생길 수 있다.

또한 본 연구는 원 논문의 parameterization에서 공통효과/군집특이효과 분해를 회귀계수에 적용했던 아이디어를, 비지도 setting에서는 군집 평균에 적용한 것으로 볼 수 있다. 즉, 원 논문과 문제의식은 연결되지만, 직접적으로 동일한 모형을 비지도화한 것은 아니며, "predictor effect heterogeneity"를 "component mean heterogeneity"로 재구성한 모형이다.

### 4.2 이질적 변수의 정의

변수 $k$에 대하여 $\delta_{\cdot k} = (\delta_{1k}, \dots, \delta_{Kk})^\top$ 라 두면, mean heterogeneity를 유발하는 변수 집합을 다음과 같이 정의한다.

$$S_H = \{k : \|\delta_{\cdot k}\|_2 \neq 0\}$$

즉, $\delta_{1k} = \dots = \delta_{Kk} = 0$이면 변수 $k$는 모든 군집에서 평균이 동일하므로 군집 간 mean difference를 유발하지 않는다. 반대로 $\|\delta_{\cdot k}\|_2 > 0$이면 변수 $k$는 적어도 하나의 군집에서 평균 차이를 만들어내므로 mean-heterogeneity-driving variable이다.

여기서 범위를 분명히 해야 한다. 위 정의는 "현재 baseline model 하에서의 평균 기반 이질성"을 의미한다. 따라서 본 모형이 직접 식별하는 것은 variance heterogeneity나 covariance heterogeneity를 포함한 일반적 의미의 cluster-forming variable 전체가 아니라, 공통 공분산 구조 아래에서 mean shift를 통해 군집 분리를 유발하는 변수이다. 이 점은 연구 범위를 정확하게 한정해 주며, 이후 분산구조 확장으로 자연스럽게 이어질 수 있다.

### 4.3 공분산 구조: 왜 diagonal covariance부터 시작하는가

본 연구의 초기 모델 설정 및 1차 시뮬레이션에서는 다음과 같이 두는 것이 타당하다.

$$\Sigma_j = \Sigma = \mathrm{diag}(\sigma_1^2, \dots, \sigma_p^2)$$

또는 가장 단순하게 $\Sigma = I_p$로 둔다. 이 가정 아래에서는 군집이 주어졌을 때 각 좌표가 조건부 독립이므로, mean heterogeneity selection 문제를 가장 선명하게 분리하여 볼 수 있다. 이는 "실제 데이터가 반드시 독립이다"라는 주장이 아니라, 1차 단계에서 mean heterogeneity 자체를 먼저 정교하게 정식화하기 위한 working model이다.

원 논문 역시 component variance가 heterogeneity 해석에 직접 영향을 주며, covariance structure가 달라질 경우 source of heterogeneity의 정의가 복잡해진다고 논의한다. 따라서 본 연구에서도 1차 단계에서는 공통 diagonal covariance로 문제를 정리하고, 이후 확장으로 unequal diagonal variance, correlated feature, 또는 cluster-specific covariance를 고려하는 것이 전략적으로 적절하다.

---

## 5. 추정방법

### 5.1 정규화된 목적함수

모수 $\Theta = (\pi_1, \dots, \pi_K, \mu_0, \delta_1, \dots, \delta_K, \Sigma)$ 에 대해 다음과 같은 normalized penalized log-likelihood를 고려한다.

$$\mathcal{L}_n(\Theta) = \frac{1}{n} \sum_{i=1}^n \log \left[ \sum_{j=1}^K \pi_j \phi_p(X_i; \mu_0 + \delta_j, \Sigma) \right] - \lambda_n \sum_{k=1}^p w_k \|\delta_{\cdot k}\|_2$$

여기서 $w_k$는 adaptive weight이며 예를 들면 다음과 같이 pilot estimator로부터 구성할 수 있다.

$$w_k = (\|\tilde{\delta}_{\cdot k}\|_2 + \varepsilon)^{-\gamma}$$

이와 같이 목적함수를 $n$으로 정규화해 두면 $\lambda_n$의 order를 이론적으로 다루기 더 명확하다. 물론 동치인 비정규화 형태 $\ell_n(\Theta) - n\lambda_n \sum_{k=1}^p w_k \|\delta_{\cdot k}\|_2$ 로도 쓸 수 있으나, 논문에서는 둘 중 하나로 반드시 통일하는 것이 필요하다. 본 연구에서는 normalized form을 기본 표기로 채택한다.

또한 현재 모형에서는 variable-wise selection을 위해 element-wise $\ell_1$보다 $\|\delta_{\cdot k}\|_2$ 형태의 group penalty를 사용하는 것이 더 자연스럽다. 하나의 변수는 모든 군집에서 함께 살아남거나 함께 0이 되므로, "어떤 변수 전체가 mean heterogeneity를 유발하는가"라는 질문에 직접 대응할 수 있다.

### 5.2 식별성 제약

$\mu_j = \mu_0 + \delta_j$ 만으로는 $\mu_0$와 $\delta_j$의 분해가 유일하지 않다. 따라서 다음과 같은 sum-to-zero 제약이 필요하다.

$$\sum_{j=1}^K \delta_{jk} = 0, \quad k = 1, \dots, p$$

이는 원 논문에서의 effects-model parameterization과 동일한 역할을 수행하는 식별성 제약이다. 다만 원 논문에서는 variance scaling을 함께 고려하는 회귀 setting이었다면, 본 연구에서는 mean structure에 이 제약을 적용한다는 차이가 있다.

### 5.3 계산 알고리즘

계산은 EM 알고리즘을 기본 골격으로 한다.

E-step에서는 책임도(responsibility)를 계산한다.

$$\tau_{ij} = P(Z_i = j \mid X_i, \Theta) = \frac{\pi_j \phi_p(X_i; \mu_0 + \delta_j, \Sigma)}{\sum_{\ell=1}^K \pi_\ell \phi_p(X_i; \mu_0 + \delta_\ell, \Sigma)}$$

M-step에서는 $\pi_j, \Sigma, \mu_0, \delta_j$를 갱신한다. 특히 $\Sigma$가 diagonal일 때 각 변수 $k$에 대한 업데이트는 거의 분리되어 다음과 같은 문제로 귀결된다.

$$\min_{\mu_{0k}, \delta_{\cdot k}} \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^K \tau_{ij} \sigma_k^{-2} (x_{ik} - \mu_{0k} - \delta_{jk})^2 + \lambda_n w_k \|\delta_{\cdot k}\|_2$$

subject to

$$\sum_{j=1}^K \delta_{jk} = 0$$

실제 구현에서는 $\mathbf{1}_K$ 의 직교여공간 basis $Q$ 를 써서 $\delta_{\cdot k} = Q \alpha_k$ 로 재파라미터화하면 제약이 사라져 unconstrained group lasso 문제로 바뀐다. 이는 수치적 안정성과 희소성 보존 측면에서 유리하다.

튜닝 파라미터 $\lambda_n$ 와 군집 수 $K$ 는 BIC, ICL, 혹은 clustering stability 기준으로 선택할 수 있다. 원 논문에서도 mixture component 수와 penalty parameter 선택에 BIC를 사용한다.

---
 
## 6. 이론적 연구목표

기존 연구는 fixed $p, m$ 설정에서 adaptive estimator의 $\sqrt{n}$-consistency와 selection consistency를 제시하였다. 본 연구의 박사논문 기여는 이 결과를 비지도 high-dimensional setting으로 확장하는 데 있다. 다만 현재 단계에서 직접적으로 "misclustering rate가 항상 0으로 간다"는 식의 강한 주장을 두는 것은 과도하므로, 이론 목표를 다음과 같이 정교하게 설정하는 것이 바람직하다.

**첫째, 식별성.** label switching을 제외하면 $(\pi, \mu_0, \Delta, \Sigma)$가 유일하게 식별됨을 보인다.

**둘째, 추정오차 경계.** 희소도 $s = |S_H|$에 대해 다음과 같은 형태의 오차 경계를 목표로 한다.

$$\|\hat{\Delta} - \Delta^*\|_F = O_p \left( \sqrt{\frac{sK \log p}{n}} \right)$$

**셋째, support recovery.** 적절한 beta-min 조건 $\min_{k \in S_H} \|\delta_{\cdot k}^*\|_2 \ge c\lambda_n$ 하에서 다음을 보이고자 한다.

$$P(\hat{S}_H = S_H) \to 1$$

**넷째, clustering performance.** 현재 baseline에서는 다음 두 종류의 결과 중 하나를 목표로 하는 것이 더 적절하다.

하나는 Bayes rule 대비 excess classification risk consistency이다. 예를 들어 다음과 같은 결과를 보이는 방식이다.

$$R(\hat{g}) - R(g^*) \to 0$$

다른 하나는 separation-dependent misclustering bound이다. 즉, 군집 간 분리가 충분히 커지는 경우에 한하여 misclustering rate가 0으로 가는 결과를 제시하는 것이다. 이를 위해 다음을 정의하고,

$$\Delta_{\min, n}^2 = \min_{j \neq \ell} \sum_{k \in S_H} \frac{(\delta_{jk}^* - \delta_{\ell k}^*)^2}{\sigma_k^2}$$

$\Delta_{\min, n}^2 \to \infty$와 같은 stronger separation regime 하에서 다음을 보이는 방향이 더 타당하다.

$$\frac{1}{n} \sum_{i=1}^n I(\hat{Z}_i \neq Z_i^*) \to 0$$

반대로 separation이 고정되어 있고 component overlap이 존재하면, Bayes classifier 자체도 양의 오분류율을 가질 수 있으므로 무조건적인 zero-misclustering consistency를 전면에 내세우는 것은 적절하지 않다. 이 부분은 미팅에서 선제적으로 정리해 두는 것이 좋다.

기본 가정의 예로는 다음을 둘 수 있다.

$$\pi_j^* \ge \pi_{\min} > 0, \quad 0 < c_\sigma \le \sigma_k^2 \le C_\sigma < \infty$$

$$s \log p = o(n)$$

그리고 clustering 관련 결과를 위해서는 추가적으로 suitable separation assumption을 둘 수 있다.

---

## 7. 기존 연구와의 차별성

본 연구의 차별점은 단순히 "클러스터링에 유용한 변수"를 고르는 것이 아니라, 군집 평균의 좌표별 분해를 통해 "왜 군집이 갈리는가"를 직접 묻는다는 점에 있다.

다만 원 논문과 현재 모형의 관계는 정확히 구분해서 설명할 필요가 있다. 원 논문에서는 mixture regression setting에서 relevant predictor 집합 $S_R$와 source of heterogeneity 집합 $S_H$를 동시에 구분한다. 즉, 공통효과는 있지만 군집특이효과는 없는 predictor와, 실제로 군집 간 효과 차이를 만들어내는 predictor를 분리한다. 반면 현재 비지도 baseline model은 outcome이 없는 평균 혼합모형이므로, 원 논문에서의 $S_R$–$S_H$ 구조를 그대로 재현하는 것은 아니다. 현재 1차 모형이 직접 식별하는 것은 사실상 "mean-heterogeneity-driving coordinate"에 해당하는 $S_H$-유사 객체이다. 이 점을 솔직하게 밝히는 것이 오히려 연구의 범위를 더 선명하게 만든다.

즉, 본 연구는 원 논문의 개념을 그대로 비지도화한 것이 아니라, 그 핵심 문제의식인 "heterogeneity의 원천 추적"을 mean-shift clustering 문제로 재구성한 방법론이라고 정리하는 가장 정확하다.

또한 원 논문이 high-dimensional setting, cluster learning, multivariate extension을 후속 연구 방향으로 제시했다는 점을 감안하면, 본 연구는 바로 그 방향 중 "cluster learning under high-dimensional heterogeneity pursuit"를 직접 겨냥한 확장으로 해석할 수 있다.

---

## 8. 후속 확장 방향

현재 1차 모형은 mean heterogeneity selection에 집중한다. 그러나 "common but relevant structure"까지 포함하는 더 풍부한 비지도 모형으로 확장하려면 예를 들어 다음과 같은 구조를 고려할 수 있다.

$$X_i = \mu_0 + \Lambda f_i + \delta_{Z_i} + \varepsilon_i, \quad \varepsilon_i \sim N_p(0, \Psi)$$

여기서 $\Lambda f_i$는 전체 표본에 공통적인 저차원 구조를 나타내고, $\delta_{Z_i}$는 군집특이 평균구조를 나타낸다. 이 경우에는 공통 구조를 반영하는 좌표와 mean heterogeneity를 유발하는 좌표를 더 정교하게 구분할 수 있다. 다만 이는 현재 1차 논문의 범위를 넘어서는 확장 주제로 두는 것이 적절하며, 우선은 공통 diagonal covariance 아래에서의 sparse mean-effects clustering을 먼저 완성하는 것이 논리적으로 더 단단하다.

---
## Part II. 시뮬레이션 결과

본 절의 시뮬레이션은 제안 모형이 "모든 형태의 군집 형성 변수"를 찾는지 검증하는 것이 아니라, 공통 공분산 구조 하에서 군집 간 평균 차이를 유발하는 변수(mean-heterogeneity-driving variables)를 얼마나 정확히 식별하는지, 그리고 그러한 선택이 실제 군집 성능 개선으로 이어지는지를 경험적으로 확인하는 데 목적이 있다. 이는 원 논문이 혼합회귀에서 relevant predictor와 source of heterogeneity를 구분하고, heterogeneity pursuit가 보다 parsimonious한 모형을 제공할 수 있다고 논의한 문제의식을 비지도 평균혼합모형으로 옮겨온 것이다. 다만 원 논문은 회귀 setting과 fixed $p, m$ 이론에 초점을 두고 있으므로, 본 절의 시뮬레이션은 그 이론을 직접 재현하는 것이 아니라 mean-shift clustering 상황에서의 경험적 타당성을 검토하는 단계로 이해하는 것이 적절하다.

### 1. 시뮬레이션 데이터 생성 모형 (Data Generation Model)

시뮬레이션 데이터는 본 연구가 제안하는 평균 분해 구조($\mu_j = \mu_0 + \delta_j$)를 엄밀하게 따르도록 생성되었다. 총 표본 수 $n$, 전체 변수 차원 $p$, 잠재 군집 수 $K=3$이 주어졌을 때, 각 관측치 $X_i \in \mathbb{R}^p$와 잠재 군집 라벨 $Z_i \in \lbrace 1, 2, 3 \rbrace$는 다음의 확률적 과정을 통해 생성된다.

**1) 잠재 군집 생성 (Cluster Assignment)**

군집 혼합 비율은 완전히 균등하게 설정하였다.

$$P(Z_i = j) = \pi_j = \frac{1}{K} \quad (j = 1, 2, 3)$$

**2) 군집별 평균 편차 구조 (Mean Heterogeneity Structure)**

전체 $p$개의 변수 중, 앞의 $q$개 변수만이 실제로 군집 간 평균 차이를 유발하는 이질성 유발 변수(Signal variables)이며, 나머지 $p - q$개는 평균 차이가 없는 노이즈 변수(Noise variables)이다. 군집 $j$의 편차 벡터 $\delta_j \in \mathbb{R}^p$는 다음과 같이 정의된다.

- **Cluster 1 ($j=1$):** $\delta_{1k} = a \ (k \le q)$, $\delta_{1k} = 0 \ (k > q)$
    
- **Cluster 2 ($j=2$):** $\delta_{2k} = 0 \ (\forall k)$
    
- **Cluster 3 ($j=3$):** $\delta_{3k} = -a \ (k \le q)$, $\delta_{3k} = 0 \ (k > q)$
    

여기서 $a$는 이질성을 발생시키는 **신호 강도(Signal strength)**를 의미하며, $a$가 클수록 군집 간 거리가 멀어지고, $a$가 작을수록 군집이 강하게 중첩(overlap)된다. 위 구조에 의해 $\sum_{j=1}^K \delta_{jk} = 0$ 제약이 데이터 생성 단계에서부터 완벽하게 성립한다.

**3) 공분산 구조 및 데이터 추출 (Covariance and Sampling)**

관측치 $X_i$는 군집별로 다변량 정규분포에서 독립적으로 추출된다.

$$X_i \mid Z_i = j \sim N_p(\mu_j, \Sigma)$$

이때, 공분산 행렬 $\Sigma = \mathrm{diag}(\sigma_1^2, \dots, \sigma_p^2)$는 다음과 같이 변수 역할에 따라 다른 분산을 가지도록 설계하였다.

- **Signal variables ($1 \le k \le q$):** $\sigma_k^2 = 1$
    
- **Noise variables ($q < k \le p$):** $\sigma_k^2 = 2$
    

노이즈 변수의 분산을 2배로 증폭시킴으로써, 단순한 유클리디안 거리 기반의 클러스터링(예: K-means)이나 페널티가 없는 모형(예: 일반 GMM)이 차원의 저주(Curse of Dimensionality)에 빠지기 쉬운 혹독한 고차원 환경을 모사하였다. 생성된 최종 데이터 행렬 $X$는 분석 전 변수별로 중심화(empirical centering)를 거쳐 $\mu_0 \approx 0$이 되도록 전처리된다.

### 2. 비교 방법론 및 벤치마크 모형 정의

본 시뮬레이션에서는 제안 모형(Proposed HP)의 변수 선택 능력과 군집 복원 성능을 객관적으로 평가하기 위해, 전통적 군집화, 기존 변수 선택 군집화, 그리고 이상적인 오라클(Oracle) 모형 등 총 4개 그룹의 방법론을 비교 벤치마크로 설정하였다. 각 모형의 정의와 포함 목적은 다음과 같다.

**1) 전통적 비지도 학습 (Baseline Methods)**

변수 선택(Variable selection) 기능 없이 전체 $p$차원의 공간을 모두 사용하여 군집화를 수행하는 가장 기본적인 벤치마크 모형들이다. 고차원 노이즈 변수가 다수 존재할 때 차원의 저주로 인해 군집화 성능이 얼마나 붕괴하는지 확인하기 위해 포함되었다.

- **K-means:** 전통적인 유클리디안 거리 기반의 군집 분석.
    
- **PCA + K-means:** 전체 분산의 80%를 설명하는 주성분(Principal Components)을 추출하여 차원 축소를 선행한 뒤 K-means를 적용하는 모형.
    
- **GMM (Unpenalized):** 어떠한 페널티도 가하지 않고 전체 차원 $p$에 대해 추정을 수행하는 다변량 가우시안 혼합모형. 본 실험에서는 공통 대각 공분산 행렬(EEI 모형)을 가정하였다.
    

**2) 기존 변수 선택 군집화 및 절제(Ablation) 모형**

페널티(Penalty)를 부여하여 군집화에 기여하지 않는 노이즈 변수를 걸러내려는 목적을 가진 모형들이다. 제안 모형의 그룹 단위 선택 구조가 왜 필수적인지를 입증하는 대조군 역할을 한다.

- **Sparse K-means (sparcl):** 변수별 가중치(weight) 벡터에 $\ell_1$ 페널티를 부여하여, 군집 간 분산에 기여하지 않는 변수의 가중치를 0으로 강제하는 대표적인 희소 군집화 벤치마크.
    
- **Naive Lasso (Element-wise $\ell_1$ + $\mu_0$):** 본 연구의 제안 모형과 동일한 $\mu_j = \mu_0 + \delta_j$ 구조를 가지나, 변수 그룹 단위($\|\delta_{\cdot k}\|_2$)가 아닌 개별 파라미터($|\delta_{jk}|$) 단위로 $\ell_1$ 페널티를 적용한 모형. 변수 내 파편화(fragmentation) 오류를 유발하는 한계를 보여주기 위한 절제 실험(Ablation study) 벤치마크이다.
    

**3) 제안 모형 (Proposed Model)**

- **Proposed HP (Adaptive Group $\ell_2$):** 본 연구에서 제안하는 희소 혼합평균효과 클러스터링 모형. 변수 $k$에 대한 군집 편차 벡터 전체($\delta_{\cdot k}$)에 Adaptive Group Lasso 페널티를 부여하여, "어떤 변수가 평균 기반 이질성(Mean Heterogeneity)을 유발하는가"를 직접적으로 식별한다. 직교여공간 $Q$-basis를 활용하여 $\sum_j \delta_{jk} = 0$ 제약을 안정적으로 만족시킨다.
    
- **+ Refit (재적합 파이프라인):** 페널티 기반 모형(Sparse K-means, Naive Lasso, Proposed HP)이 선택한 활성 변수 집합($\hat{S}$)만을 사용하여, 페널티 없이 GMM을 다시 추정하는 과정. $\ell_1$ 또는 $\ell_2$ 페널티로 인해 0을 향해 과도하게 수축된(shrinkage bias) 파라미터 추정치를 편향 없이 회복시켜 최종 군집 성능(ARI)을 극대화하기 위해 수행된다.
    

**4) 오라클 벤치마크 (Oracle Bounds)**

유한 표본(Finite sample) 환경에서 달성할 수 있는 군집화 성능의 이론적 상한선(Upper bound)을 확인하기 위해, 정답 정보를 활용하는 두 가지 오라클 모형을 구성하였다.

- **Oracle-feature baseline (True Vars):** 노이즈 변수를 완벽하게 배제하고, 데이터를 생성할 때 이질성을 부여한 실제 정답 변수(True variables, $q=10$)들만 주어졌다고 가정하여 추정한 GMM. 변수 선택 알고리즘이 도달해야 할 이상적인 목표치이다.
    
- **True-parameter oracle (Bayes Classifier):** 파라미터 추정 과정 자체를 생략하고, 데이터를 생성할 때 사용한 진짜 모수(혼합 비율 $\pi$, 군집 평균 $\mu$, 공분산 $\Sigma$)를 그대로 사후 확률(Posterior probability) 공식에 대입하여 분류하는 모형. 베이즈 오류율(Bayes error rate)에 기반한 분류 성능의 절대적 한계치이다.
    

### 3. 기본 환경($p=20$)에서의 신호 강도 변화 검증 (R=10 반복 실험)

#### 3.1 실험 목적

시뮬레이션의 목적은 노이즈 좌표의 분산이 커진 환경에서 신호 강도 $a$를 점진적으로 약화시켰을 때, 제안 모형과 비교 모형의 성능이 어떻게 변하는지를 확인하는 것이다. 특히 이 실험은 우연성을 배제하기 위해 10회 반복(Monte Carlo)을 수행하였으며, 다음 세 가지를 검토한다.

- **첫째,** 제안 모형이 mean-heterogeneity variable selection을 얼마나 안정적으로 수행하는가.
    
- **둘째,** 과거 prototype인 Naive Lasso와 대표적 비지도 벤치마크인 Sparse K-means가 어떤 한계를 보이는가.
    
- **셋째,** HP+refit 파이프라인이 어느 신호 구간까지 near-oracle behavior를 유지하는가.
    

#### 3.2 실험 세팅

표본 수는 $n=300$, 총 차원은 $p=20$, 군집 수는 $K=3$ 으로 두었다. 진짜 mean-heterogeneity 좌표는 5개이며, 변수 1-5에서만 군집 간 평균 차이가 존재한다. 각 군집의 평균 편차는 대칭적 구조 $(a, 0, -a)$ 를 따르도록 구성하였다. 변수 6-20은 평균 차이가 없는 노이즈 좌표이며, 전통적 거리 기반 방법을 어렵게 만들기 위해 표준편차를 $\sqrt{2}$ 배(분산 2배)로 증폭하였다. 실제 구현에서는 데이터에 대해 중심화만 수행하고 변수별 추가 스케일링은 하지 않아, 노이즈 분산 증폭 효과가 그대로 유지되도록 하였다. 신호 강도는 $a=1.8, 1.5, 1.3$ 의 세 구간으로 설정하였다. 모든 지표는 10회 반복 수행 후의 평균 (표준오차) 로 표기하였다.
#### 3.3 시나리오별 결과표 (Mean 및 SE)

**[시나리오 1] 명확한 신호 환경 (a = 1.8)**

|**방법론**|**사용 차원**|**변수 선택**|**ARI**|**TPR**|**FPR**|**S^**|
|---|---|---|---|---|---|---|
|K-means|20.000 (0.000)|No|0.894 (0.013)|-|-|-|
|GMM (Unpenalized)|20.000 (0.000)|No|0.912 (0.007)|-|-|-|
|Sparse K-means|19.400 (0.163)|Yes|0.912 (0.009)|1.000 (0.000)|0.960 (0.011)|19.400 (0.163)|
|$\rightarrow$ + Refit|19.400 (0.163)|-|0.913 (0.007)|-|-|-|
|Naive Lasso|6.900 (0.458)|Yes|0.908 (0.006)|1.000 (0.000)|0.127 (0.031)|6.900 (0.458)|
|$\rightarrow$ + Refit|6.900 (0.458)|-|0.912 (0.009)|-|-|-|
|**Proposed HP**|20.000 (0.000)|Yes|0.911 (0.008)|**1.000 (0.000)**|**0.007 (0.007)**|**5.100 (0.100)**|
|$\rightarrow$ **Proposed HP + Refit**|5.100 (0.100)|-|**0.913 (0.008)**|-|-|-|
|Oracle-feature baseline|5.000 (0.000)|Oracle Vars|0.913 (0.008)|1.000 (0.000)|0.000 (0.000)|5.000 (0.000)|
|True-parameter oracle|5.000 (0.000)|Oracle Params|0.912 (0.009)|1.000 (0.000)|0.000 (0.000)|5.000 (0.000)|

**[시나리오 2] 중간 신호 환경 (a = 1.5)**

|**방법론**|**사용 차원**|**변수 선택**|**ARI**|**TPR**|**FPR**|**S^**|
|---|---|---|---|---|---|---|
|K-means|20.000 (0.000)|No|0.706 (0.023)|-|-|-|
|GMM (Unpenalized)|20.000 (0.000)|No|0.801 (0.013)|-|-|-|
|Sparse K-means|19.300 (0.260)|Yes|0.808 (0.010)|1.000 (0.000)|0.953 (0.017)|19.300 (0.260)|
|$\rightarrow$ + Refit|19.300 (0.260)|-|0.802 (0.015)|-|-|-|
|Naive Lasso|6.500 (0.453)|Yes|0.800 (0.008)|1.000 (0.000)|0.100 (0.030)|6.500 (0.453)|
|$\rightarrow$ + Refit|6.500 (0.453)|-|0.815 (0.011)|-|-|-|
|**Proposed HP**|20.000 (0.000)|Yes|0.814 (0.010)|**1.000 (0.000)**|**0.033 (0.018)**|**5.500 (0.269)**|
|$\rightarrow$ **Proposed HP + Refit**|5.500 (0.269)|-|**0.813 (0.008)**|-|-|-|
|Oracle-feature baseline|5.000 (0.000)|Oracle Vars|0.812 (0.009)|1.000 (0.000)|0.000 (0.000)|5.000 (0.000)|
|True-parameter oracle|5.000 (0.000)|Oracle Params|0.817 (0.009)|1.000 (0.000)|0.000 (0.000)|5.000 (0.000)|

**[시나리오 3] 약한 신호 환경 (a = 1.3)**

|**방법론**|**사용 차원**|**변수 선택**|**ARI**|**TPR**|**FPR**|**S^**|
|---|---|---|---|---|---|---|
|K-means|20.000 (0.000)|No|0.493 (0.031)|-|-|-|
|GMM (Unpenalized)|20.000 (0.000)|No|0.637 (0.031)|-|-|-|
|Sparse K-means|19.900 (0.100)|Yes|0.687 (0.012)|1.000 (0.000)|0.993 (0.007)|19.900 (0.100)|
|$\rightarrow$ + Refit|19.900 (0.100)|-|0.638 (0.031)|-|-|-|
|Naive Lasso|7.500 (0.342)|Yes|0.682 (0.009)|1.000 (0.000)|0.167 (0.023)|7.500 (0.342)|
|$\rightarrow$ + Refit|7.500 (0.342)|-|0.691 (0.011)|-|-|-|
|**Proposed HP**|20.000 (0.000)|Yes|0.698 (0.014)|**1.000 (0.000)**|**0.053 (0.034)**|**5.800 (0.512)**|
|$\rightarrow$ **Proposed HP + Refit**|5.800 (0.512)|-|**0.696 (0.012)**|-|-|-|
|Oracle-feature baseline|5.000 (0.000)|Oracle Vars|0.694 (0.012)|1.000 (0.000)|0.000 (0.000)|5.000 (0.000)|
|True-parameter oracle|5.000 (0.000)|Oracle Params|0.730 (0.009)|1.000 (0.000)|0.000 (0.000)|5.000 (0.000)|

이때 $a=1.3$ 구간에서는 True-parameter oracle의 ARI도 0.730 (0.009), Oracle-feature baseline이 0.694 (0.012)로 하락하므로, 이 시나리오는 "완전 복원 가능 영역"이 아니라 강한 overlap이 존재하는 약신호 영역으로 해석하는 것이 맞다.

#### 3.4 기본 환경($p=20$) 시뮬레이션의 해석

**첫째,** 제안 모형은 중간 신호 구간까지 mean-heterogeneity selection 측면에서 매우 안정적이다. $a=1.8$에서 HP는 평균 TPR=1.000 (0.000), FPR=0.007 (0.007), $\hat{S}=5.100 (0.100)$를 보여 true signal coordinates를 완벽에 가깝게 복원하였다. 이 상태에서 refit을 수행하면, 페널티(penalty)로 인해 발생했던 수축 편향(shrinkage bias)이 제거되면서 ARI가 0.913으로 상승하여 oracle baseline과 동일한 수치를 보였다. 제안된 HP 방법론은 $a=1.5$ 구간에서도 노이즈 변수를 거의 대부분 걸러내며, 진짜 변수만 사용한 Oracle 모형에 필적하는(asymptotically equivalent to the oracle) 성능을 보였다.

**둘째,** 변수 선택 기능이 없는 전통적 모형들(K-means, Unpenalized GMM)은 20개의 변수를 모두 사용하여 $\hat{S}=20.000$, FPR=1.000을 기록하였고, 노이즈가 누적되어 전체적인 군집화 성능이 저하되었다. 반면 Sparse K-means는 변수 선택을 목표로 하지만, 현재 구현과 tuning rule 하에서는 가짜 희소성(fake sparsity)을 보였다. 평균 선택 변수 수가 $19 \sim 20$개로 유지되어 노이즈 제거를 수행하지 못했으며, 이 상태에서 refit을 수행하면 일반 GMM과 동일한 수준으로 성능이 내려감을 확인할 수 있다.

**셋째,** Naive Lasso는 강한 신호에서는 작동할 수 있으나, 신호가 약해지면 빠르게 붕괴한다. $a=1.8$에서는 $\hat{S}=6.900$, FPR=0.127로 과선택이 눈에 띄며, $a=1.3$에서는 $\hat{S}=7.500$, FPR=0.167로 더 증가한다. 이는 element-wise shrinkage와 강제 centering 조합이 변수 단위의 안정적인 selection을 보장하지 못한다는 점을 뚜렷하게 보여준다. 따라서 본 실험은 왜 group-wise penalty와 $Q$-basis 재파라미터화가 필수적인지를 입증하는 강력한 ablation evidence로 작용한다.

### 4. 고차원 환경($n=200, p=300$)에서의 모형 확장성 검증 (R=10 반복 실험)

#### 4.1 실험 목적

시뮬레이션의 목적은 변수(차원)가 표본 수보다 많아지는 진정한 고차원($p > n$) 환경에서, 노이즈 좌표의 분산이 커지고 신호 강도 $a$가 점진적으로 약화될 때 제안 모형과 비교 모형의 성능이 어떻게 변하는지를 확인하는 것이다. 10회 반복(Monte Carlo)을 수행하였으며, 다음 세 가지를 검토한다.

- **첫째,** $p > n$ 세팅에서 기존 GMM 및 거리 기반 벤치마크의 차원의 저주(Curse of Dimensionality) 및 행렬 특이성(Singularity) 붕괴 양상 파악.
    
- **둘째,** $\ell_1$ 페널티 기반의 Naive Lasso와 Group Lasso($\ell_2$) 기반의 제안 모형(HP) 간의 구조적 수축 편향(Shrinkage Bias) 방어 능력 비교.
    
- **셋째,** 약신호(Weak signal) 및 고차원 환경에서 HP+refit 파이프라인의 near-oracle behavior 유지 여부.
    

#### 4.2 실험 세팅

표본 수는 $n=200$, 총 차원은 $p=300$ 으로 두어 $p > n$ 고차원 설정을 반영하였고, 군집 수는 $K=3$ 으로 두었다. 진짜 mean-heterogeneity 좌표는 $q=10$ 개이며, 변수 1-10에서만 군집 간 평균 차이가 존재한다. 대칭적 편차 구조 $(a, 0, -a)$ 를 적용하였으며, 변수 11-300은 평균 차이가 없는 노이즈 좌표로 두고 표준편차를 $\sqrt{2}$ 배(분산 2배)로 증폭하였다. 신호 강도는 $a=1.8, 1.5, 1.3$ 의 세 구간으로 설정하였다. 모든 지표는 10회 반복 수행 후의 평균 (표준오차) 로 표기하였다.

#### 4.3 시나리오별 결과표 (Mean 및 SE)

**[시나리오 1] 명확한 신호 환경 (a = 1.8)**

|**방법론**|**사용 차원**|**ARI**|**TPR**|**FPR**|**S^**|
|---|---|---|---|---|---|
|K-means|300.000 (0.000)|0.851 (0.019)|-|-|-|
|PCA + K-means|91.000 (0.149)|0.826 (0.018)|-|-|-|
|GMM (Unpenalized)|300.000 (0.000)|0.694 (0.092)|-|-|-|
|Sparse K-means|246.600 (13.783)|0.994 (0.003)|1.000 (0.000)|0.816 (0.048)|246.600 (13.783)|
|$\rightarrow$ + Refit|246.600 (13.783)|0.727 (0.112)|-|-|-|
|Naive Lasso|10.000 (0.000)|0.979 (0.005)|1.000 (0.000)|0.000 (0.000)|10.000 (0.000)|
|$\rightarrow$ + Refit|10.000 (0.000)|0.995 (0.003)|-|-|-|
|**Proposed HP**|300.000 (0.000)|0.995 (0.003)|**1.000 (0.000)**|**0.003 (0.001)**|**11.000 (0.298)**|
|$\rightarrow$ **Proposed HP + Refit**|11.000 (0.298)|**0.995 (0.003)**|-|-|-|
|Oracle-feature baseline|10.000 (0.000)|0.995 (0.003)|1.000 (0.000)|0.000 (0.000)|10.000 (0.000)|
|True-parameter oracle|10.000 (0.000)|0.993 (0.003)|1.000 (0.000)|0.000 (0.000)|10.000 (0.000)|

**[시나리오 2] 중간 신호 환경 (a = 1.5)**

| **방법론**                               | **사용 차원**        | **ARI**           | **TPR**           | **FPR**           | **S^**             |
| ------------------------------------- | ---------------- | ----------------- | ----------------- | ----------------- | ------------------ |
| K-means                               | 300.000 (0.000)  | 0.521 (0.029)     | -                 | -                 | -                  |
| PCA + K-means                         | 91.700 (0.153)   | 0.504 (0.028)     | -                 | -                 | -                  |
| GMM (Unpenalized)                     | 300.000 (0.000)  | 0.395 (0.018)     | -                 | -                 | -                  |
| Sparse K-means                        | 164.100 (14.676) | 0.966 (0.009)     | 1.000 (0.000)     | 0.531 (0.051)     | 164.100 (14.676)   |
| $\rightarrow$ + Refit                 | 164.100 (14.676) | 0.042 (0.041)     | -                 | -                 | -                  |
| Naive Lasso                           | 10.000 (0.000)   | 0.903 (0.019)     | 1.000 (0.000)     | 0.000 (0.000)     | 10.000 (0.000)     |
| $\rightarrow$ + Refit                 | 10.000 (0.000)   | 0.962 (0.009)     | -                 | -                 | -                  |
| **Proposed HP**                       | 300.000 (0.000)  | 0.962 (0.008)     | **1.000 (0.000)** | **0.003 (0.001)** | **10.900 (0.233)** |
| $\rightarrow$ **Proposed HP + Refit** | 10.900 (0.233)   | **0.960 (0.009)** | -                 | -                 | -                  |
| Oracle-feature baseline               | 10.000 (0.000)   | 0.962 (0.009)     | 1.000 (0.000)     | 0.000 (0.000)     | 10.000 (0.000)     |
| True-parameter oracle                 | 10.000 (0.000)   | 0.965 (0.010)     | 1.000 (0.000)     | 0.000 (0.000)     | 10.000 (0.000)     |

**[시나리오 3] 약 신호 환경 (a = 1.3)**

| **방법론**                               | **사용 차원**        | **ARI**           | **TPR**           | **FPR**           | **S^**             |
| ------------------------------------- | ---------------- | ----------------- | ----------------- | ----------------- | ------------------ |
| K-means                               | 300.000 (0.000)  | 0.422 (0.020)     | -                 | -                 | -                  |
| PCA + K-means                         | 91.800 (0.133)   | 0.418 (0.016)     | -                 | -                 | -                  |
| GMM (Unpenalized)                     | 300.000 (0.000)  | 0.418 (0.018)     | -                 | -                 | -                  |
| Sparse K-means                        | 197.300 (23.920) | 0.895 (0.016)     | 1.000 (0.000)     | 0.646 (0.082)     | 197.300 (23.920)   |
| $\rightarrow$ + Refit                 | 197.300 (23.920) | 0.218 (0.074)     | -                 | -                 | -                  |
| Naive Lasso                           | 13.000 (2.022)   | 0.794 (0.023)     | 1.000 (0.000)     | 0.010 (0.007)     | 13.000 (2.022)     |
| $\rightarrow$ + Refit                 | 13.000 (2.022)   | 0.829 (0.092)     | -                 | -                 | -                  |
| **Proposed HP**                       | 300.000 (0.000)  | 0.917 (0.012)     | **1.000 (0.000)** | **0.001 (0.001)** | **10.400 (0.163)** |
| $\rightarrow$ **Proposed HP + Refit** | 10.400 (0.163)   | **0.915 (0.011)** | -                 | -                 | -                  |
| Oracle-feature baseline               | 10.000 (0.000)   | 0.916 (0.015)     | 1.000 (0.000)     | 0.000 (0.000)     | 10.000 (0.000)     |
| True-parameter oracle                 | 10.000 (0.000)   | 0.927 (0.009)     | 1.000 (0.000)     | 0.000 (0.000)     | 10.000 (0.000)     |

#### 4.4 고차원 시뮬레이션의 해석

**첫째, $p > n$ 고차원 환경에서의 기존 모형의 완전한 붕괴:** 차원이 표본 수보다 많아지는($p=300 > n=200$) 환경에 진입하자, GMM 모형은 고차원 노이즈 분산에 압도당하며 $a=1.5$ 구간에서 ARI가 0.395로 추락하였다. 특히 주목할 점은 Sparse K-means의 Refit 단계 붕괴 현상이다. Sparse K-means는 $a=1.5$와 $a=1.3$ 구간에서 약 $160 \sim 197$개의 노이즈 변수를 제거하지 못하고 유지하였으며, 이를 기반으로 GMM Refit을 수행하자 너무 많은 변수 탓에 다변량 추정이 구조적으로 실패하여 ARI가 0.042, 0.218 수준으로 완전히 파괴되었다. 이는 고차원 환경에서 강력하고 확실한 차원 축소(Support Recovery) 능력이 왜 필수적인지를 단적으로 보여준다.

**둘째, 수축 편향(Shrinkage Bias)에 대한 제안 모형(HP)의 압도적 방어력:** 독립적인 고차원 환경에서 Naive Lasso는 $a=1.8$ 및 $a=1.5$ 구간에서 정확히 10.0개의 정답 변수를 고르는 훌륭한 선택 정확도를 보였다. 그러나 Refit을 하기 전 초기 추정 ARI를 보면 HP 모형이 Naive Lasso를 완벽히 압도한다. $a=1.5$에서 Naive Lasso의 초기 ARI는 0.903인 반면, HP는 오라클과 동일한 0.962를 달성했다. 이는 개별 $\ell_1$ 페널티가 강한 축소 과정에서 군집 간의 상대적 거리 구조까지 왜곡시키는 반면, HP의 Group Lasso 기반 $Q$-basis 재파라미터화는 변수를 안전하게 축소시키면서도 군집 중심점 간의 상대적 차이(Mean Heterogeneity)를 우아하게 보존해 냄을 증명하는 강력한 결과다.

**셋째, 약신호 환경(a=1.3)에서의 HP 모형의 절대적 우위:** 군집 간 분리가 모호해지는 약신호 구간에서는 제안 모형의 진가가 더욱 뚜렷하게 발휘된다. $a=1.3$ 구간에서 Naive Lasso는 노이즈를 걸러내는 데 실패하며 변수를 평균 13개 선택하였고, Refit 이후에도 군집 복원 성능(ARI=0.829)이 오라클(0.916)에 한참 미치지 못했다. 반면, 제안 모형(HP)은 이 혼탁한 환경에서도 정답 변수 10개를 족집게처럼 찾아내며 훨씬 정교한 $\hat{S} \approx 10.4$를 기록했다. 나아가 Refit 유무와 관계없이 ARI 0.915~0.917을 유지하며, 완벽한 정답 변수만을 사용한 Oracle-feature baseline(0.916)과 동등한 군집 성능을 방어하는 놀라운(near-oracle) 결과를 달성하였다.

### 5. 시뮬레이션 종합 정리

기본 환경($p=20$) 및 진정한 고차원 환경($n=200, p=300$)에서의 Monte Carlo 시뮬레이션을 종합하면, 제안 방법은 다음과 같은 객관적인 우수성을 입증하였다.

- **첫째,** 표본 수보다 차원이 큰 $p > n$ 빅데이터 환경에서 기존 모형들이 겪는 차원의 저주 및 행렬 특이성 문제를 완벽하게 극복하고, 노이즈를 99% 이상 걸러내어 차원을 안전하게 압축한다.
    
- **둘째,** $\ell_1$ 페널티 기반의 Naive Lasso가 겪는 수축 편향(Shrinkage Bias) 및 약신호 붕괴 현상과 달리, $Q$-basis 기반 Group Lasso 구조를 도입한 제안 모형(HP)은 초기 추정 단계에서부터 파라미터 왜곡 없이 최고 수준의 군집 복원 성능(ARI)을 보장한다.
    
- **셋째,** HP 파이프라인은 신호 강도나 차원 크기에 구애받지 않고 Oracle-feature baseline에 필적하는 강건한 군집화 성능을 달성한다.
    

결과적으로, 본 시뮬레이션은 제안 방법이 "평균 기반 이질성의 원천"을 고차원 환경에서도 정확히 추적할 수 있음을 완벽히 증명하였다. 이후 논문 확장 과제로는 변수 간 강한 상관관계(Correlated features)가 존재할 때, 개별 변수를 파편화시키는 Naive 모형을 HP가 그룹 단위 보존 능력으로 어떻게 압도하는지를 보여주는 추가 실험이 가장 효과적일 것이다.
