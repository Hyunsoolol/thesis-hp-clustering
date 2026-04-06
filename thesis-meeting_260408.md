# [연구 미팅 보고서] 고차원 데이터에서 이질성 유발 변수를 탐색하는 희소 혼합평균효과 기반 클러스터링 방법론

---

## [핵심 요약] 과거 버전 대비 모델 개선 사항

본 보고서는 고차원 환경에서 "어떤 변수가 군집의 이질성을 유발하는가(Source of Heterogeneity)?"를 식별하기 위해 기존에 구상했던 모델의 수학적/알고리즘적 한계를 대폭 개선한 이론적 배경과 제안 모형을 담고 있습니다.

### 1. 모델 구조 및 알고리즘의 핵심 개선

- **변수 단위 선택의 명확성 확보 (Group Lasso 도입):** * _과거:_ 개별 파라미터($|\delta_{kj}|$)에 $\ell_1$ 페널티 적용 $\rightarrow$ 하나의 변수 내에서도 특정 군집만 0이 되는 파편화(Fragmentation) 발생.
    
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

**둘째,** 기존 문헌의 effects-model parameterization을 비지도 setting에 맞게 재해석하여, 군집 평균을
$$\mu_j = \mu_0 + \delta_j$$
형태로 분해하는 parsimonious mixture mean-effects model을 구축한다.

**셋째,** $p \gg n$ 환경에서 support recovery와 mean structure estimation error에 대한 이론적 보장을 우선적으로 제시하고, 추가로 군집 성능에 대해서는 separation-dependent clustering bound 또는 Bayes rule 대비 excess risk consistency를 목표로 한다. 원 논문의 이론은 fixed **p, m**에서의 estimation/selection consistency에 초점이 있으며, high-dimensional extension은 명시적으로 후속 과제로 제시되어 있다. 본 연구의 박사논문 기여는 바로 이 지점을 비지도 고차원 설정으로 확장하는 데 있다.

---

## 3. 핵심 연구질문

본 연구는 다음 질문에 답하는 것을 목표로 한다.
* **Q1.** 비지도 혼합모형에서 mean heterogeneity의 source를 어떻게 엄밀히 정의할 것인가?
* **Q2.** 군집 추정과 mean-heterogeneity variable selection을 동시에 수행하는 정규화 mixture model은 어떻게 설계할 것인가?
* **Q3.** 고차원 환경에서 이 방법의 support recovery와 parameter error bound를 어떻게 보일 것인가?
* **Q4.** 분산 구조가 달라질 때 heterogeneity의 정의를 어떻게 조정할 것인가?

---

## 4. 제안모형

### 4.1 기본 모형
관측치 $X_i = (X_{i1}, \dots, X_{ip})^\top \in \mathbb{R}^p$, 잠재 군집 $Z_i \in \{1, \dots, K\}$에 대하여 다음 baseline model을 제안한다.

$$P(Z_i = j) = \pi_j, \quad j = 1, \dots, K$$
$$X_i \mid Z_i = j \sim N_p(\mu_j, \Sigma)$$
$$\mu_j = \mu_0 + \delta_j, \quad \sum_{j=1}^K \delta_{jk} = 0, \quad k = 1, \dots, p$$

여기서 $\mu_0 \in \mathbb{R}^p$는 sum-to-zero coding 하의 grand mean parameter이고, $\delta_j \in \mathbb{R}^p$는 군집 **j**의 mean deviation vector이다. 따라서 각 군집의 중심은
$$\mu_j = E(X_i \mid Z_i = j) = \mu_0 + \delta_j$$
로 표현된다.

다만 중요한 점은, 현재 선택한 제약
$$\sum_{j=1}^K \delta_{jk} = 0$$
하에서 $\mu_0$는 일반적으로 marginal population mean과 동일하지 않다는 것이다. 실제로
$$E(X_i) = \sum_{j=1}^K \pi_j \mu_j = \mu_0 + \sum_{j=1}^K \pi_j \delta_j$$
이므로, $\mu_0$는 $\pi_j$가 모두 같거나 $\sum_j \pi_j \delta_j = 0$인 특수한 경우에만 marginal mean과 일치한다. 따라서 본 연구에서 $\mu_0$는 "전체 평균"이라기보다 effects-style parameterization에서의 기준점 역할을 하는 grand mean parameter로 해석하는 것이 정확하다. 이 점을 명확히 하지 않으면 모형 해석에 불필요한 혼동이 생길 수 있다.

또한 본 연구는 원 논문의 parameterization에서 공통효과/군집특이효과 분해를 회귀계수에 적용했던 아이디어를, 비지도 setting에서는 군집 평균에 적용한 것으로 볼 수 있다. 즉, 원 논문과 문제의식은 연결되지만, 직접적으로 동일한 모형을 비지도화한 것은 아니며, "predictor effect heterogeneity"를 "component mean heterogeneity"로 재구성한 모형이다.

### 4.2 이질적 변수의 정의
변수 **k**에 대하여
$$\delta_{\cdot k} = (\delta_{1k}, \dots, \delta_{Kk})^\top$$
라 두면, mean heterogeneity를 유발하는 변수 집합을
$$S_H = \{k : \|\delta_{\cdot k}\|_2 \neq 0\}$$
로 정의한다.

즉, $\delta_{1k} = \dots = \delta_{Kk} = 0$이면 변수 **k**는 모든 군집에서 평균이 동일하므로 군집 간 mean difference를 유발하지 않는다. 반대로 $\|\delta_{\cdot k}\|_2 > 0$이면 변수 **k**는 적어도 하나의 군집에서 평균 차이를 만들어내므로 mean-heterogeneity-driving variable이다.

여기서 범위를 분명히 해야 한다. 위 정의는 "현재 baseline model 하에서의 평균 기반 이질성"을 의미한다. 따라서 본 모형이 직접 식별하는 것은 variance heterogeneity나 covariance heterogeneity를 포함한 일반적 의미의 cluster-forming variable 전체가 아니라, 공통 공분산 구조 아래에서 mean shift를 통해 군집 분리를 유발하는 변수이다. 이 점은 연구 범위를 정확하게 한정해 주며, 이후 분산구조 확장으로 자연스럽게 이어질 수 있다.

### 4.3 공분산 구조: 왜 diagonal covariance부터 시작하는가
본 연구의 초기 모델 설정 및 1차 시뮬레이션에서는
$$\Sigma_j = \Sigma = \text{diag}(\sigma_1^2, \dots, \sigma_p^2)$$
또는 가장 단순하게 $\Sigma = I_p$로 두는 것이 타당하다.

이 가정 아래에서는 군집이 주어졌을 때 각 좌표가 조건부 독립이므로, mean heterogeneity selection 문제를 가장 선명하게 분리하여 볼 수 있다. 이는 "실제 데이터가 반드시 독립이다"라는 주장이 아니라, 1차 단계에서 mean heterogeneity 자체를 먼저 정교하게 정식화하기 위한 working model이다.

원 논문 역시 component variance가 heterogeneity 해석에 직접 영향을 주며, covariance structure가 달라질 경우 source of heterogeneity의 정의가 복잡해진다고 논의한다. 따라서 본 연구에서도 1차 단계에서는 공통 diagonal covariance로 문제를 정리하고, 이후 확장으로 unequal diagonal variance, correlated feature, 또는 cluster-specific covariance를 고려하는 것이 전략적으로 적절하다.

---

## 5. 추정방법

### 5.1 정규화된 목적함수
모수
$$\Theta = (\pi_1, \dots, \pi_K, \mu_0, \delta_1, \dots, \delta_K, \Sigma)$$
에 대해 다음과 같은 normalized penalized log-likelihood를 고려한다.
$$\mathcal{L}_n(\Theta) = \frac{1}{n} \sum_{i=1}^n \log \left[ \sum_{j=1}^K \pi_j \phi_p(X_i; \mu_0 + \delta_j, \Sigma) \right] - \lambda_n \sum_{k=1}^p w_k \|\delta_{\cdot k}\|_2$$
여기서 $w_k$는 adaptive weight이며 예를 들면
$$w_k = (\|\tilde{\delta}_{\cdot k}\|_2 + \varepsilon)^{-\gamma}$$
와 같이 pilot estimator로부터 구성할 수 있다.

이와 같이 목적함수를 **n**으로 정규화해 두면 $\lambda_n$의 order를 이론적으로 다루기 더 명확하다. 물론 동치인 비정규화 형태
$$\ell_n(\Theta) - n\lambda_n \sum_{k=1}^p w_k \|\delta_{\cdot k}\|_2$$
로도 쓸 수 있으나, 논문에서는 둘 중 하나로 반드시 통일하는 것이 필요하다. 본 연구에서는 normalized form을 기본 표기로 채택한다.

또한 현재 모형에서는 variable-wise selection을 위해 element-wise L1보다
$$\|\delta_{\cdot k}\|_2$$
형태의 group penalty를 사용하는 것이 더 자연스럽다. 하나의 변수는 모든 군집에서 함께 살아남거나 함께 0이 되므로, "어떤 변수 전체가 mean heterogeneity를 유발하는가"라는 질문에 직접 대응할 수 있다.

### 5.2 식별성 제약
$$\mu_j = \mu_0 + \delta_j$$
만으로는 $\mu_0$와 $\delta_j$의 분해가 유일하지 않다. 따라서
$$\sum_{j=1}^K \delta_{jk} = 0, \quad k = 1, \dots, p$$
와 같은 sum-to-zero 제약이 필요하다. 이는 원 논문에서의 effects-model parameterization과 동일한 역할을 수행하는 식별성 제약이다. 다만 원 논문에서는 variance scaling을 함께 고려하는 회귀 setting이었다면, 본 연구에서는 mean structure에 이 제약을 적용한다는 차이가 있다.

### 5.3 계산 알고리즘
계산은 EM 알고리즘을 기본 골격으로 한다.

E-step에서는 책임도(responsibility)를 계산한다.
$$\tau_{ij} = P(Z_i = j \mid X_i, \Theta) = \frac{\pi_j \phi_p(X_i; \mu_0 + \delta_j, \Sigma)}{\sum_{\ell=1}^K \pi_\ell \phi_p(X_i; \mu_0 + \delta_\ell, \Sigma)}$$

M-step에서는 $\pi_j, \Sigma, \mu_0, \delta_j$를 갱신한다. 특히 $\Sigma$가 diagonal일 때 각 변수 **k**에 대한 업데이트는 거의 분리되어 다음과 같은 문제로 귀결된다.
$$\min_{\mu_{0k}, \delta_{\cdot k}} \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^K \tau_{ij} \sigma_k^{-2} (x_{ik} - \mu_{0k} - \delta_{jk})^2 + \lambda_n w_k \|\delta_{\cdot k}\|_2$$
subject to
$$\sum_{j=1}^K \delta_{jk} = 0$$

실제 구현에서는 $\mathbf{1}_K$의 직교여공간 basis **Q**를 써서
$$\delta_{\cdot k} = Q \alpha_k$$
로 재파라미터화하면 제약이 사라져 unconstrained group lasso 문제로 바뀐다. 이는 수치적 안정성과 희소성 보존 측면에서 유리하다.
튜닝 파라미터 $\lambda_n$와 군집 수 **K**는 BIC, ICL, 혹은 clustering stability 기준으로 선택할 수 있다. 원 논문에서도 mixture component 수와 penalty parameter 선택에 BIC를 사용한다.

---

## 6. 이론적 연구목표

기존 연구는 fixed **p, m** 설정에서 adaptive estimator의 $\sqrt{n}$-consistency와 selection consistency를 제시하였다. 본 연구의 박사논문 기여는 이 결과를 비지도 high-dimensional setting으로 확장하는 데 있다. 다만 현재 단계에서 직접적으로 "misclustering rate가 항상 0으로 간다"는 식의 강한 주장을 두는 것은 과도하므로, 이론 목표를 다음과 같이 정교하게 설정하는 것이 바람직하다.

**첫째, 식별성.** label switching을 제외하면 $(\pi, \mu_0, \Delta, \Sigma)$가 유일하게 식별됨을 보인다.

**둘째, 추정오차 경계.** 희소도 $s = |S_H|$에 대해 다음과 같은 형태의 오차 경계를 목표로 한다.
$$\|\hat{\Delta} - \Delta^*\|_F = O_p \left( \sqrt{\frac{sK \log p}{n}} \right)$$

**셋째, support recovery.** 적절한 beta-min 조건
$$\min_{k \in S_H} \|\delta_{\cdot k}^*\|_2 \ge c\lambda_n$$
하에서
$$P(\hat{S}_H = S_H) \to 1$$
을 보이고자 한다.

**넷째, clustering performance.** 현재 baseline에서는 다음 두 종류의 결과 중 하나를 목표로 하는 것이 더 적절하다.
하나는 Bayes rule 대비 excess classification risk consistency이다. 예를 들어
$$R(\hat{g}) - R(g^*) \to 0$$
와 같은 결과를 보이는 방식이다.

다른 하나는 separation-dependent misclustering bound이다. 즉, 군집 간 분리가 충분히 커지는 경우에 한하여 misclustering rate가 0으로 가는 결과를 제시하는 것이다. 이를 위해
$$\Delta_{\min, n}^2 = \min_{j \neq \ell} \sum_{k \in S_H} \frac{(\delta_{jk}^* - \delta_{\ell k}^*)^2}{\sigma_k^2}$$
를 정의하고, $\Delta_{\min, n}^2 \to \infty$와 같은 stronger separation regime 하에서
$$\frac{1}{n} \sum_{i=1}^n I(\hat{Z}_i \neq Z_i^*) \to 0$$
를 보이는 방향이 더 타당하다. 반대로 separation이 고정되어 있고 component overlap이 존재하면, Bayes classifier 자체도 양의 오분류율을 가질 수 있으므로 무조건적인 zero-misclustering consistency를 전면에 내세우는 것은 적절하지 않다. 이 부분은 미팅에서 선제적으로 정리해 두는 것이 좋다.

기본 가정의 예로는 다음을 둘 수 있다.
$$\pi_j^* \ge \pi_{\min} > 0, \quad 0 < c_\sigma \le \sigma_k^2 \le C_\sigma < \infty$$
$$s \log p = o(n)$$
그리고 clustering 관련 결과를 위해서는 추가적으로 suitable separation assumption을 둘 수 있다.

---

## 7. 기존 연구와의 차별성

본 연구의 차별점은 단순히 "클러스터링에 유용한 변수"를 고르는 것이 아니라, 군집 평균의 좌표별 분해를 통해 "왜 군집이 갈리는가"를 직접 묻는다는 점에 있다.

다만 원 논문과 현재 모형의 관계는 정확히 구분해서 설명할 필요가 있다. 원 논문에서는 mixture regression setting에서 relevant predictor 집합 $S_R$와 source of heterogeneity 집합 $S_H$를 동시에 구분한다. 즉, 공통효과는 있지만 군집특이효과는 없는 predictor와, 실제로 군집 간 효과 차이를 만들어내는 predictor를 분리한다. 반면 현재 비지도 baseline model은 outcome이 없는 평균 혼합모형이므로, 원 논문에서의 $S_R$–$S_H$ 구조를 그대로 재현하는 것은 아니다. 현재 1차 모형이 직접 식별하는 것은 사실상 "mean-heterogeneity-driving coordinate"에 해당하는 $S_H$-유사 객체이다. 이 점을 솔직하게 밝히는 것이 오히려 연구의 범위를 더 선명하게 만든다.

즉, 본 연구는 원 논문의 개념을 그대로 비지도화한 것이 아니라, 그 핵심 문제의식인 "heterogeneity의 원천 추적"을 mean-shift clustering 문제로 재구성한 방법론이라고 정리하는 것이 가장 정확하다.

또한 원 논문이 high-dimensional setting, cluster learning, multivariate extension을 후속 연구 방향으로 제시했다는 점을 감안하면, 본 연구는 바로 그 방향 중 "cluster learning under high-dimensional heterogeneity pursuit"를 직접 겨냥한 확장으로 해석할 수 있다.

---

## 8. 후속 확장 방향

현재 1차 모형은 mean heterogeneity selection에 집중한다. 그러나 "common but relevant structure"까지 포함하는 더 풍부한 비지도 모형으로 확장하려면 예를 들어 다음과 같은 구조를 고려할 수 있다.
$$X_i = \mu_0 + \Lambda f_i + \delta_{Z_i} + \varepsilon_i, \quad \varepsilon_i \sim N_p(0, \Psi)$$
여기서 $\Lambda f_i$는 전체 표본에 공통적인 저차원 구조를 나타내고, $\delta_{Z_i}$는 군집특이 평균구조를 나타낸다. 이 경우에는 공통 구조를 반영하는 좌표와 mean heterogeneity를 유발하는 좌표를 더 정교하게 구분할 수 있다. 다만 이는 현재 1차 논문의 범위를 넘어서는 확장 주제로 두는 것이 적절하며, 우선은 공통 diagonal covariance 아래에서의 sparse mean-effects clustering을 먼저 완성하는 것이 논리적으로 더 단단하다.

---

## Part II. 연속형 혼합모형 시뮬레이션 결과

본 절의 시뮬레이션은 제안 모형이 "모든 형태의 군집 형성 변수"를 찾는지 검증하는 것이 아니라, 공통 공분산 구조 하에서 군집 간 평균 차이를 유발하는 변수(mean-heterogeneity-driving variables)를 얼마나 정확히 식별하는지, 그리고 그러한 선택이 실제 군집 성능 개선으로 이어지는지를 경험적으로 확인하는 데 목적이 있다. 이는 원 논문이 혼합회귀에서 relevant predictor와 source of heterogeneity를 구분하고, heterogeneity pursuit가 보다 parsimonious한 모형을 제공할 수 있다고 논의한 문제의식을 비지도 평균혼합모형으로 옮겨온 것이다. 다만 원 논문은 회귀 setting과 fixed **p, m** 이론에 초점을 두고 있으므로, 본 절의 시뮬레이션은 그 이론을 직접 재현하는 것이 아니라 mean-shift clustering 상황에서의 경험적 타당성을 검토하는 단계로 이해하는 것이 적절하다.

### 1. 시뮬레이션: 신호 강도 변화에 따른 phase transition 검증

#### 2.1 실험 목적
시뮬레이션의 목적은 노이즈 좌표의 분산이 커진 환경에서 신호 강도 **a**를 점진적으로 약화시켰을 때, 제안 모형과 비교 모형의 성능이 어떻게 변하는지를 확인하는 것이다. 특히 이 실험은 다음 세 가지를 검토한다.
* **첫째,** 제안 모형이 mean-heterogeneity variable selection을 얼마나 안정적으로 수행하는가.
* **둘째,** 과거 prototype인 Naive Lasso와 대표적 비지도 벤치마크인 Sparse K-means가 어떤 한계를 보이는가.
* **셋째,** HP+refit 파이프라인이 어느 신호 구간까지 near-oracle behavior를 유지하는가.

#### 2.2 실험 세팅
표본 수는 **n=300**, 총 차원은 **p=20**, 군집 수는 **K=3**으로 두었다. 진짜 mean-heterogeneity 좌표는 5개이며, 변수 1–5에서만 군집 간 평균 차이가 존재한다. 각 군집의 평균 편차는 대칭적 구조
** (a, 0, -a) **
를 따르도록 구성하였다. 변수 6–20은 평균 차이가 없는 노이즈 좌표이며, 전통적 거리 기반 방법을 어렵게 만들기 위해 분산을 2배로 증폭하였다. 실제 구현에서는 데이터에 대해 중심화만 수행하고 변수별 추가 스케일링은 하지 않아, 노이즈 분산 증폭 효과가 그대로 유지되도록 하였다. 신호 강도는 **a=1.8, 1.5, 1.3**의 세 구간으로 설정하였다. 이는 각각 명확한 신호, 중간 수준의 신호, 그리고 군집 중첩이 심한 약한 신호 구간에 해당한다.

#### 2.3 시나리오별 결과표

**[시나리오 1] 명확한 신호 환경 (a = 1.8)**

| 방법론 | 사용 차원 | 변수 선택 | ARI | TPR | FPR | $\hat{S}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| K-means | 20 | No | 0.438 | - | - | - |
| GMM (Unpenalized) | 20 | No | 0.409 | - | - | - |
| Sparse K-means | 20 | Yes | 0.921 | 1.000 | 1.000 | 20 |
| $\rightarrow$ + Refit | 20 | - | 0.409 | - | - | - |
| Naive Lasso | 5 | Yes | 0.822 | 1.000 | 0.000 | 5 |
| $\rightarrow$ + Refit | 5 | - | 0.912 | - | - | - |
| **Proposed HP** | 20 | Yes | 0.822 | **1.000** | **0.000** | **5** |
| $\rightarrow$ **Proposed HP + Refit** | 5 | - | **0.912** | - | - | - |
| Oracle GMM (True Vars) | 5 | Ideal | 0.912 | 1.000 | 0.000 | 5 |

**[시나리오 2] 중간 신호 환경 (a = 1.5)**

| 방법론 | 사용 차원 | 변수 선택 | ARI | TPR | FPR | $\hat{S}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| K-means | 20 | No | 0.366 | - | - | - |
| GMM (Unpenalized) | 20 | No | 0.358 | - | - | - |
| Sparse K-means | 20 | Yes | 0.737 | 1.000 | 1.000 | 20 |
| $\rightarrow$ + Refit | 20 | - | 0.358 | - | - | - |
| Naive Lasso | 7 | Yes | 0.438 | 1.000 | 0.133 | 7 |
| $\rightarrow$ + Refit | 7 | - | 0.406 | - | - | - |
| **Proposed HP** | 20 | Yes | 0.469 | **1.000** | **0.000** | **5** |
| $\rightarrow$ **Proposed HP + Refit** | 5 | - | **0.761** | - | - | - |
| Oracle GMM (True Vars) | 5 | Ideal | 0.761 | 1.000 | 0.000 | 5 |

**[시나리오 3] 약한 신호 환경 (a = 1.3)**

| 방법론 | 사용 차원 | 변수 선택 | ARI | TPR | FPR | $\hat{S}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| K-means | 20 | No | 0.343 | - | - | - |
| GMM (Unpenalized) | 20 | No | 0.298 | - | - | - |
| Sparse K-means | 20 | Yes | 0.711 | 1.000 | 1.000 | 20 |
| $\rightarrow$ + Refit | 20 | - | 0.298 | - | - | - |
| Naive Lasso | 6 | Yes | 0.357 | 1.000 | 0.067 | 6 |
| $\rightarrow$ + Refit | 6 | - | 0.336 | - | - | - |
| **Proposed HP** | 20 | Yes | 0.332 | **1.000** | **0.067** | **6** |
| $\rightarrow$ **Proposed HP + Refit** | 6 | - | **0.336** | - | - | - |
| Oracle GMM (True Vars) | 5 | Ideal | 0.694 | 1.000 | 0.000 | 5 |

이때 **a=1.3** 구간에서는 oracle GMM 자체의 ARI도 0.694까지 하락하므로, 이 시나리오는 "완전 복원 가능 영역"이 아니라 강한 overlap이 존재하는 약신호 영역으로 해석하는 것이 맞다.

#### 2.4 2차 시뮬레이션의 해석
**첫째,** 제안 모형은 중간 신호 구간까지 mean-heterogeneity selection 측면에서 매우 안정적이다. **a=1.8**과 **a=1.5**에서 HP는 모두 **TPR=1.000**, **FPR=0.000**, **$\hat{S}=5$**를 보여 true signal coordinates를 정확히 복원하였다. 이 상태에서 refit을 수행하면 ARI가 각각 0.912, 0.761로 상승하여 oracle GMM과 동일한 수치를 보였다. 따라서 이 구간에서는 HP+refit이 empirical benchmark 기준으로 near-oracle behavior를 보인다고 정리할 수 있다. 그러나 이것을 이론적 의미의 "oracle convergence"라고 표현하는 것은 적절하지 않다. 본 결과는 현재 simulation regime에서의 경험적 현상으로 해석해야 한다.

**둘째,** Sparse K-means는 현재 구현과 tuning rule 하에서는 가짜 희소성(fake sparsity)을 보였다. 세 시나리오 모두에서 ARI 자체는 비교적 높게 나타났지만, FPR이 항상 1.000이고 선택 변수 수가 20으로 유지되었다. 즉, 노이즈 변수에 작은 가중치를 줄 뿐 완전한 제거는 수행하지 못하였다. 이 상태에서 refit을 하면 모든 차원을 다시 사용하는 일반 GMM과 동일한 수준으로 성능이 내려갔다. 따라서 현재 실험은 Sparse K-means가 "선택 기반의 parsimonious clustering"을 달성했다기보다, 가중치 부여를 통한 간접적 완화에 머물렀음을 시사한다. 다만 이 평가는 어디까지나 현재 구현과 tuning 하에서의 결과로 해석하는 것이 적절하다.

**셋째,** Naive Lasso는 대칭적이고 강한 신호에서는 우연히 잘 작동할 수 있으나, 신호가 약해지면 빠르게 붕괴한다. **a=1.8**에서는 **$\hat{S}=5$**, **FPR=0.000**으로 이상적으로 보이지만, **a=1.5**에서는 **$\hat{S}=7$**, **FPR=0.133**으로 즉시 과선택이 발생한다. 이는 element-wise shrinkage와 강제 centering 조합이 변수 단위의 안정적인 selection을 보장하지 못한다는 점을 보여준다. 따라서 본 실험은 왜 group-wise penalty와 **Q**-basis 재파라미터화가 필요한지를 뒷받침하는 ablation evidence로 해석할 수 있다.

**넷째,** 약신호 구간에서는 selection 정확도와 clustering recovery가 분리될 수 있다. **a=1.3**에서 HP는 **TPR=1.000**, **FPR=0.067**, **$\hat{S}=6$**으로 비교적 양호한 선택 성능을 보였으나, HP+refit의 ARI는 0.336에 머물렀고 oracle GMM 0.694와 큰 차이를 보였다. 이는 약한 신호 영역에서 한두 개의 false positive만으로도 refit likelihood landscape가 크게 흔들릴 수 있고, 초기값 민감도나 local optimum 문제가 실제 군집 성능에 직접 영향을 미칠 수 있음을 시사한다. 따라서 이 구간은 "제안 방법이 실패했다"기보다, 약신호·중첩 군집 환경에서 현재 refit 단계의 수치적 불안정성이 남아 있다고 해석하는 것이 타당하다. 향후에는 start 수 확대, thresholded refit, stability-based selection, adaptive weighting 개선 등을 통해 이 구간을 보완할 필요가 있다.

#### 2.5 종합 정리
시뮬레이션을 종합하면, 제안 방법은 공통 공분산 구조 하에서의 mean heterogeneity selection 문제에 대해 다음과 같은 경험적 패턴을 보였다.
* **첫째,** 중간 이상의 신호 구간에서는 진짜 mean-heterogeneity coordinates를 매우 정확히 복원한다.
* **둘째,** 선택 단계의 shrinkage bias는 refit 단계에서 상당 부분 제거될 수 있다.
* **셋째,** 현재 설정에서는 HP+refit이 **a=1.8**과 **a=1.5**에서 oracle benchmark와 동일한 ARI를 보였으나, 이는 empirical near-oracle behavior로 이해해야 하며 asymptotic guarantee를 의미하지는 않는다.
* **넷째,** 약신호 구간에서는 selection 자체보다 refit 안정성이 병목이 될 수 있다.

결국 본 절의 시뮬레이션은 제안 방법이 "평균 기반 이질성의 원천"을 추적하는 비지도 모형으로서 충분한 가능성을 가진다는 점, 그리고 실제 논문에서는 HP(선택) + refit(최종 추정)을 기본 파이프라인으로 채택하는 것이 합리적이라는 점을 뒷받침한다. 또한 원 논문이 high-dimensional setting과 more general mixture learning을 후속 과제로 남겨 둔 것과 마찬가지로, 본 연구에서도 correlated features, unequal variances, cluster-number learning, mixed-type extension이 다음 단계의 핵심 과제로 이어질 것이다.
