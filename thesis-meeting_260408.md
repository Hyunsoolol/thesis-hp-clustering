# [연구 미팅 보고서] 고차원 데이터에서 평균 기반 이질성 유발 변수를 탐색하는 희소 혼합평균효과 기반 클러스터링 방법론

---

## 핵심 요약

본 보고서는 고차원 환경에서 평균 구조의 관점에서 “어떤 변수가 군집 간 차이를 실제로 유발하는가”를 식별하기 위한 비지도 혼합모형을 정리한 것이다. 직접적인 목표는 모든 형태의 군집 형성 변수를 찾는 것이 아니라, 공통 공분산 구조 하에서 **군집 간 평균 차이를 유발하는 변수(mean-heterogeneity-driving variables)**를 식별하는 데 있다.

현재 버전에서의 핵심 개선은 다음과 같다.

1. **변수 단위 선택의 명확화** 기존 prototype의 element-wise $\ell_1$ penalty는 하나의 변수 안에서도 일부 군집 효과만 0이 되는 파편화를 유발할 수 있었다. 현재는 변수 $k$에 대응하는 군집 편차 벡터 전체 $\delta_{\cdot k}$에 **Adaptive Group Lasso** penalty를 부여하여 변수 단위의 선택을 유도한다.
    
2. **식별성 제약의 안정화** 혼합비율에 의존하는
    
    $$\sum_{j=1}^K \pi_j \delta_j = 0$$
    
    대신
    
    $$\sum_{j=1}^K \delta_{jk} = 0,\qquad k=1,\dots,p$$
    
    제약을 사용하여 ANOVA형 effects parameterization에 더 가깝게 만들었다.
    
3. **제약을 보존하는 재파라미터화** 직교여공간 basis $Q$를 이용해
    
    $$\delta_{\cdot k}=Q\alpha_k$$
    
    로 재파라미터화하면, 제약을 자동으로 만족시키면서 희소성을 안정적으로 유지할 수 있다.
    
1. **adaptive penalty의 반영** 참고 논문은 non-adaptive보다 adaptive regularization이 변수선택 성능에서 더 안정적임을 보인다. 현재 제안 모형은 이 점을 반영하여 adaptive group penalty를 중심으로 정리한다.



## 1. 연구배경 및 문제의식

혼합모형 기반 회귀에서는 단순히 중요한 설명변수를 찾는 것만으로 충분하지 않고, 그중에서도 실제로 군집 간 차이를 만들어내는 변수, 즉 source of heterogeneity를 구분하는 것이 더 해석가능하고 더 간명한 모형을 만든다. Li et al.의 혼합회귀 연구는 predictor effect를 공통효과와 군집특이효과로 분해하고, relevant predictor와 heterogeneity-driving predictor를 동시에 식별하는 regularized finite mixture effects regression을 제안하였다. 특히 이 연구는 component variance가 다를 때 raw effect와 scaled effect를 구분해야 함을 강조하고, adaptive penalty, generalized lasso, generalized EM, BIC tuning, fixed $p,m$ 이론, 그리고 correlated predictors·unequal mixing·all-heterogeneous setting까지 폭넓게 다루었다.

그러나 비지도학습, 특히 고차원 클러스터링에서는 이와 같은 이질성의 원천 추적이 상대적으로 덜 정식화되어 있다. 기존 sparse clustering이나 model-based clustering은 주로 군집 복원 자체나 변수선택에 초점을 맞추는 경우가 많고, 군집 평균을 공통 부분과 군집특이 부분으로 분해하여 어떤 좌표가 mean heterogeneity를 실제로 유발하는지 직접 추적하는 effects-style parameterization은 상대적으로 부족하다.

본 연구는 이러한 문제의식을 비지도학습으로 확장한다. 즉, 반응변수 $Y_i$가 없는 상황에서 군집 평균을 latent mean structure로 보고, 이를 공통 평균 파라미터와 군집특이 편차로 분해하여 “어떤 변수들이 군집 간 평균 차이를 만들어내는가”를 직접 추적하는 클러스터링 방법론을 개발하고자 한다. 다만 현재 1차 범위는 “모든 형태의 군집 형성 변수”가 아니라, **공통 공분산 구조 하에서 mean shift를 통해 군집 분리를 유발하는 변수**를 식별하는 데 한정된다. 분산 차이나 상관구조 차이만으로 군집이 갈리는 경우는 현재 baseline model의 범위 밖에 있다.

---

## 2. 연구목표

본 연구의 1차 목표는 다음과 같다.

**첫째,** 고차원 데이터에서 군집 구조를 추정하면서 동시에 군집 간 평균 차이를 유발하는 변수 집합을 식별하는 새로운 비지도 혼합모형을 제안한다.

**둘째,** 기존 문헌의 effects-model parameterization을 비지도 setting에 맞게 재해석하여, 군집 평균을

$$\mu_j=\mu_0+\delta_j$$

형태로 분해하는 parsimonious mixture mean-effects model을 구축한다.

**셋째,** $p \gg n$ 환경에서 support recovery와 mean structure estimation error에 대한 이론적 보장을 우선적으로 제시하고, 군집 성능에 대해서는 separation-dependent clustering bound 또는 Bayes rule 대비 excess risk consistency를 목표로 한다.

---

## 3. 핵심 연구질문

- **Q1.** 비지도 혼합모형에서 mean heterogeneity의 source를 어떻게 엄밀히 정의할 것인가?
    
- **Q2.** 군집 추정과 mean-heterogeneity variable selection을 동시에 수행하는 정규화 mixture model은 어떻게 설계할 것인가?
    
- **Q3.** 고차원 환경에서 이 방법의 support recovery와 parameter error bound를 어떻게 보일 것인가?
    
- **Q4.** 분산 구조가 달라질 때 heterogeneity의 정의를 어떻게 조정할 것인가?


---

## 4. 제안모형

### 4.1 기본 모형

관측치 $X_i=(X_{i1},\dots,X_{ip})^\top\in\mathbb{R}^p$, 잠재 군집 $Z_i\in{1,\dots,K}$에 대하여 다음 baseline model을 제안한다.

$$P(Z_i=j)=\pi_j,\qquad j=1,\dots,K$$

$$X_i\mid Z_i=j \sim N_p(\mu_j,\Sigma)$$

$$\mu_j=\mu_0+\delta_j,\qquad \sum_{j=1}^K \delta_{jk}=0,\qquad k=1,\dots,p$$

여기서 $\mu_0\in\mathbb{R}^p$는 sum-to-zero coding 하의 grand mean parameter이고, $\delta_j\in\mathbb{R}^p$는 군집 $j$의 mean deviation vector이다. 따라서 각 군집의 중심은

$$\mu_j=E(X_i\mid Z_i=j)=\mu_0+\delta_j$$

로 표현된다.

다만 중요한 점은, 현재 선택한 제약

$$\sum_{j=1}^K \delta_{jk}=0$$

하에서 $\mu_0$는 일반적으로 marginal population mean과 동일하지 않다는 것이다. 실제로

$$E(X_i)=\sum_{j=1}^K \pi_j\mu_j=\mu_0+\sum_{j=1}^K \pi_j\delta_j$$

이므로, $\mu_0$는 $\pi_j$가 모두 같거나 $\sum_j \pi_j\delta_j=0$인 특수한 경우에만 marginal mean과 일치한다. 따라서 본 연구에서 $\mu_0$는 “전체 평균”이라기보다 effects-style parameterization에서의 기준점 역할을 하는 grand mean parameter로 해석하는 것이 정확하다.

또한 본 연구는 원 논문의 parameterization에서 공통효과/군집특이효과 분해를 회귀계수에 적용했던 아이디어를, 비지도 setting에서는 군집 평균에 적용한 것으로 볼 수 있다. 즉, 원 논문과 문제의식은 연결되지만, 동일한 모형을 그대로 비지도화한 것은 아니며, “predictor effect heterogeneity”를 “component mean heterogeneity”로 재구성한 모형이다.

### 4.2 이질적 변수의 정의

변수 $k$에 대하여

$$\delta_{\cdot k}=(\delta_{1k},\dots,\delta_{Kk})^\top$$

라 두면, mean heterogeneity를 유발하는 변수 집합을 다음과 같이 정의한다.

$$S_H=\{k:\|\delta_{\cdot k}\|_2\neq 0\}$$

즉, $\delta_{1k}=\cdots=\delta_{Kk}=0$이면 변수 $k$는 모든 군집에서 평균이 동일하므로 군집 간 mean difference를 유발하지 않는다. 반대로 $\|\delta_{\cdot k}\|_2>0$이면 변수 $k$는 적어도 하나의 군집에서 평균 차이를 만들어내므로 mean-heterogeneity-driving variable이다.

여기서 범위를 분명히 해야 한다. 위 정의는 “현재 baseline model 하에서의 평균 기반 이질성”을 의미한다. 따라서 본 모형이 직접 식별하는 것은 variance heterogeneity나 covariance heterogeneity를 포함한 일반적 의미의 cluster-forming variable 전체가 아니라, 공통 공분산 구조 아래에서 mean shift를 통해 군집 분리를 유발하는 변수이다.

### 4.3 공분산 구조: 왜 diagonal covariance부터 시작하는가

본 연구의 초기 모델 설정 및 1차 시뮬레이션에서는 다음과 같이 두는 것이 타당하다.

$$\Sigma_j = \Sigma = \mathrm{diag}(\sigma_1^2, \dots, \sigma_p^2)$$

또는 가장 단순하게 $\Sigma=I_p$로 둔다. 이 가정 아래에서는 군집이 주어졌을 때 각 좌표가 조건부 독립이므로, mean heterogeneity selection 문제를 가장 선명하게 분리하여 볼 수 있다. 이는 “실제 데이터가 반드시 독립이다”라는 뜻이 아니라, 1차 단계에서 mean heterogeneity 자체를 먼저 정교하게 정식화하기 위한 working model이다.

---

## 5. 추정방법

### 5.1 정규화된 목적함수

모수

$$\Theta=(\pi_1,\dots,\pi_K,\mu_0,\delta_1,\dots,\delta_K,\Sigma)$$

에 대해 다음과 같은 normalized penalized log-likelihood를 고려한다.

$$\mathcal{L}_n(\Theta) = \frac{1}{n}\sum_{i=1}^n \log\left[ \sum_{j=1}^K \pi_j\phi_p(X_i;\mu_0+\delta_j,\Sigma) \right] - \lambda_n \sum_{k=1}^p w_k \|\delta_{\cdot k}\|_2$$

여기서 $w_k$는 adaptive weight이며, 예를 들면

$$w_k=(\|\tilde{\delta}_{\cdot k}\|_2+\varepsilon)^{-\gamma}$$

와 같이 pilot estimator로부터 구성할 수 있다.

실제 구현에서는 이론에서의 $\lambda_n$과 완전히 동일한 스케일의 튜닝 파라미터를 쓰기보다, **raw penalty parameter**를 grid search와 **heuristic BIC**를 통해 선택한다. 따라서 이론 표기와 구현상 penalty scale은 구분해서 해석하는 것이 필요하다.

또한 현재 모형에서는 variable-wise selection을 위해 element-wise $\ell_1$보다 $\|\delta_{\cdot k}\|_2$ 형태의 group penalty를 사용하는 것이 더 자연스럽다. 하나의 변수는 모든 군집에서 함께 살아남거나 함께 0이 되므로, “어떤 변수 전체가 mean heterogeneity를 유발하는가”라는 질문에 직접 대응할 수 있다.

### 5.2 식별성 제약

$$\mu_j=\mu_0+\delta_j$$

만으로는 $\mu_0$와 $\delta_j$의 분해가 유일하지 않다. 따라서 다음과 같은 sum-to-zero 제약이 필요하다.

$$\sum_{j=1}^K \delta_{jk}=0,\qquad k=1,\dots,p$$

이는 원 논문에서의 effects-model parameterization과 동일한 역할을 수행하는 식별성 제약이다.

### 5.3 계산 알고리즘

계산은 EM 알고리즘을 기본 골격으로 한다.

E-step에서는 책임도(responsibility)를 계산한다.

$$\tau_{ij} = P(Z_i=j\mid X_i,\Theta) = \frac{\pi_j\phi_p(X_i;\mu_0+\delta_j,\Sigma)}{\sum_{\ell=1}^K \pi_\ell \phi_p(X_i;\mu_0+\delta_\ell,\Sigma)}$$

M-step에서는 $\pi_j,\Sigma,\mu_0,\delta_j$를 갱신한다. 특히 $\Sigma$가 diagonal일 때 각 변수 $k$에 대한 업데이트는 거의 분리되어 다음과 같은 문제로 귀결된다.

$$\min_{\mu_{0k},\delta_{\cdot k}} \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^K \tau_{ij}\sigma_k^{-2}(x_{ik}-\mu_{0k}-\delta_{jk})^2 + \lambda_n w_k \|\delta_{\cdot k}\|_2$$

subject to

$$\sum_{j=1}^K \delta_{jk}=0$$

실제 구현에서는 $\mathbf{1}_K$의 직교여공간 basis $Q$를 사용하여

$$\delta_{\cdot k}=Q\alpha_k$$

로 재파라미터화하면 제약이 사라져 unconstrained group-regularized update 문제로 바뀐다. 이는 수치적 안정성과 희소성 보존 측면에서 유리하다.

### 5.4 구현상 튜닝과 해석 주의점

- 구현에서는 exact BIC라기보다 **heuristic BIC**를 사용한다.
    
- `Adaptive Group L2`가 아니라 **Adaptive Group Lasso** 또는 **Adaptive Group $\ell_{2,1}$ penalty**가 더 정확한 명칭이다.
    
- Sparse K-means의 “사용 차원”은 실제 clustering 단계에서 사용한 변수 수가 아니라, 가중치 threshold를 기준으로 후처리한 **유효 선택 변수 수**로 해석하는 것이 맞다.
    
- HP+Refit은 진단 목적의 보조 실험으로 계산할 수 있으나, 본 보고서의 주표에서는 **single-stage HP** 성능을 중심으로 제시한다.

---
## 6. 이론적 연구목표

기존 연구는 fixed $p,m$ 설정에서 adaptive estimator의 $\sqrt{n}$-consistency와 selection consistency를 제시하였다. 본 연구의 박사논문 기여는 이 결과를 비지도 high-dimensional setting으로 확장하는 데 있다. 다만 현재 단계에서 직접적으로 “misclustering rate가 항상 0으로 간다”는 식의 강한 주장을 두는 것은 과도하므로, 이론 목표를 다음과 같이 정교하게 설정하는 것이 바람직하다.

**첫째, 식별성.** label switching을 제외하면 $(\pi,\mu_0,\Delta,\Sigma)$가 유일하게 식별됨을 보인다.

**둘째, 추정오차 경계.** 희소도 $s=|S_H|$에 대해 다음과 같은 형태의 오차 경계를 목표로 한다.

$$\|\hat{\Delta}-\Delta^*\|_F = O_p\left(\sqrt{\frac{sK\log p}{n}}\right)$$

**셋째, support recovery.** 적절한 beta-min 조건

$$\min_{k\in S_H}\|\delta_{\cdot k}^*\|_2 \ge c\lambda_n$$

하에서

$$P(\hat S_H=S_H)\to 1$$

을 보이고자 한다.

**넷째, clustering performance.** 현재 baseline에서는 다음 두 종류의 결과 중 하나를 목표로 하는 것이 더 적절하다.

하나는 Bayes rule 대비 excess classification risk consistency이다. 예를 들어

$$R(\hat g)-R(g^*)\to 0$$

와 같은 결과이다.

다른 하나는 separation-dependent misclustering bound이다. 이를 위해

$$\Delta_{\min,n}^2 = \min_{j \neq \ell} \sum_{k \in S_H} \frac{(\delta_{jk}^{\ast} - \delta_{\ell k}^{\ast})^2}{\sigma_k^2}$$

를 정의하고, $\Delta_{\min,n}^2\to\infty$와 같은 stronger separation regime 하에서

$$\frac{1}{n}\sum_{i=1}^n I(\hat Z_i\neq Z_i^*)\to 0$$

를 목표로 한다.

반대로 separation이 고정되어 있고 component overlap이 존재하면, Bayes classifier 자체도 양의 오분류율을 가질 수 있으므로 무조건적인 zero-misclustering consistency를 전면에 내세우는 것은 적절하지 않다.

기본 가정의 예로는 다음을 둘 수 있다.

$$\pi_j^{\ast} \ge \pi_{\min} > 0, \qquad 0 < c_\sigma \le \sigma_k^2 \le C_\sigma < \infty$$

$$s \log p = o(n)$$

---
## 7. 기존 연구와의 차별성

본 연구의 차별점은 단순히 “클러스터링에 유용한 변수”를 고르는 것이 아니라, 군집 평균의 좌표별 분해를 통해 “왜 군집이 갈리는가”를 직접 묻는다는 점에 있다.

다만 원 논문과 현재 모형의 관계는 정확히 구분할 필요가 있다. 원 논문에서는 mixture regression setting에서 relevant predictor 집합 $S_R$와 source of heterogeneity 집합 $S_H$를 동시에 구분한다. 반면 현재 비지도 baseline model은 outcome이 없는 평균 혼합모형이므로, 원 논문에서의 $S_R$–$S_H$ 구조를 그대로 재현하는 것은 아니다. 현재 1차 모형이 직접 식별하는 것은 사실상 mean-heterogeneity-driving coordinate에 해당하는 $S_H$-유사 객체이다.

즉, 본 연구는 원 논문의 개념을 그대로 비지도화한 것이 아니라, 그 핵심 문제의식인 “heterogeneity의 원천 추적”을 mean-shift clustering 문제로 재구성한 방법론이라고 정리하는 것이 가장 정확하다.

---

## Part II. 시뮬레이션 결과

본 절의 시뮬레이션은 제안 모형이 “모든 형태의 군집 형성 변수”를 찾는지 검증하는 것이 아니라, 공통 공분산 구조 하에서 군집 간 평균 차이를 유발하는 변수(mean-heterogeneity-driving variables)를 얼마나 정확히 식별하는지, 그리고 그러한 선택이 실제 군집 성능 개선으로 이어지는지를 경험적으로 확인하는 데 목적이 있다.

### 1. 시뮬레이션 개요

본 발표의 표는 **R=10회의 pilot Monte Carlo 평균**을 기반으로 한다. 아래 표들은 동일 알고리즘을 저차원, 고차원, 초고차원 환경으로 확장한 반복 실험 결과를 정리한 것이다.

또한 본문 표에서는 **single-stage HP (No Refit)**를 주 결과로 제시한다. 다만 HP+Refit은 진단 목적의 보조 실험으로 별도 계산 가능하며, Naive Lasso 및 Sparse K-means와의 구조적 비교에는 여전히 유용하다.

### 2. 데이터 생성 모형

시뮬레이션 데이터는 본 연구가 제안하는 평균 분해 구조

$$\mu_j=\mu_0+\delta_j$$

를 따르도록 생성하였다. 각 관측치 $X_i\in\mathbb{R}^p$와 잠재 군집 라벨 $Z_i\in{1,2,3}$는 다음 과정을 따른다.

**1) 잠재 군집 생성**

$$P(Z_i=j)=\pi_j=\frac{1}{K},\qquad j=1,2,3$$

**2) 군집별 평균 편차 구조**

앞의 $q$개 변수만 signal variables이며, 나머지는 noise variables이다. 군집별 편차 벡터는 대칭 구조

$$(a,0,-a)$$

를 따르도록 설계한다.

**3) 공분산 구조**

$$X_i \mid Z_i=j \sim N_p(\mu_j,\Sigma)$$

이며, 본문 시뮬레이션에서는 signal variables의 분산을 1, noise variables의 분산을 2로 설정한다. 생성된 최종 데이터는 변수별 중심화만 수행한다.

### 3. 평가 지표

선택 성능은

$$\mathrm{TPR} = \frac{|S_H\cap \hat S_H|}{|S_H|} ,\qquad \mathrm{FPR} = \frac{|\hat S_H\setminus S_H|}{p-|S_H|}$$

로 정의한다. 여기서 $\hat S_H$는 선택된 mean-heterogeneity 변수 집합이다. 표에서는 $\hat S$로 표기한다.

### 4. 비교 방법론 및 벤치마크

**1) 전통적 비지도 학습**

- K-means
    
- PCA + K-means
    
- GMM (Unpenalized)
    

**2) 기존 변수 선택 군집화 및 절제(Ablation) 모형**

- Sparse K-means (sparcl)
    
- Naive Lasso (element-wise $\ell_1$ + $\mu_0$)
    
- - Refit (보조 진단용)
        

**3) 제안 모형**

- Proposed HP (Adaptive Group Lasso)
    

**4) 오라클 벤치마크**

- Oracle-feature baseline (True Vars): 정답 변수 집합만 알고 GMM을 다시 추정한 baseline
    
- True-parameter oracle: 생성에 사용한 진짜 모수 $(\pi,\mu,\Sigma)$를 알고 있다고 가정한 Bayes-like benchmark
    

주의할 점은, **Oracle-feature baseline은 변수 집합만 알고 있을 뿐 실제로는 다시 적합을 수행하므로 local optimum과 초기값 영향으로 인해 true upper bound가 아니다.** true upper bound에 더 가까운 기준은 true-parameter oracle이다.

또한 Sparse K-means의 “사용 차원”은 clustering 단계에서 실제 사용된 변수 수가 아니라, 가중치 threshold를 기준으로 후처리한 **유효 선택 변수 수**이다.

---

## 5. 기본 환경 ($p=20, q=3$)

### 5.1 실험 세팅

표본 수는 $n=300$, 총 차원 $p=20$, 정답 변수는 $q=3$개이다. 신호 강도는 $a\in{1.6,1.4,1.2}$로 설정하였다.

### 5.2 시나리오별 결과표 (평균)

**[시나리오 1] 신호 환경 ($a=1.6$)**

| **방법론**                    | **사용 차원** | **ARI**   | **TPR**   | **FPR**   | $\hat{S}$ |
| -------------------------- | --------- | --------- | --------- | --------- | --------- |
| K-means                    | 20.000    | 0.413     | -         | -         | -         |
| PCA + K-means              | 14.000    | 0.400     | -         | -         | -         |
| GMM (Unpenalized)          | 20.000    | 0.522     | -         | -         | -         |
| Sparse K-means             | 19.600    | 0.679     | 1.000     | 0.976     | 19.600    |
| $\rightarrow$ + Refit      | 19.600    | 0.581     | -         | -         | -         |
| Naive Lasso                | 5.000     | 0.660     | 1.000     | 0.118     | 5.000     |
| $\rightarrow$ + Refit      | 5.000     | 0.639     | -         | -         | -         |
| **Proposed HP (No Refit)** | 20.000    | **0.673** | **1.000** | **0.059** | **4.000** |
| Oracle-feature baseline    | 3.000     | 0.682     | 1.000     | 0.000     | 3.000     |
| True-parameter oracle      | 3.000     | 0.705     | 1.000     | 0.000     | 3.000     |

**[시나리오 2] 중간 신호 환경 ($a=1.4$)**

| **방법론**                    | **사용 차원** | **ARI**   | **TPR**   | **FPR**   | $\hat{S}$ |
| -------------------------- | --------- | --------- | --------- | --------- | --------- |
| K-means                    | 20.000    | 0.350     | -         | -         | -         |
| PCA + K-means              | 14.000    | 0.346     | -         | -         | -         |
| GMM (Unpenalized)          | 20.000    | 0.456     | -         | -         | -         |
| Sparse K-means             | 19.800    | 0.601     | 1.000     | 0.988     | 19.800    |
| $\rightarrow$ + Refit      | 19.800    | 0.478     | -         | -         | -         |
| Naive Lasso                | 3.300     | 0.472     | 1.000     | 0.018     | 3.300     |
| $\rightarrow$ + Refit      | 3.300     | 0.567     | -         | -         | -         |
| **Proposed HP (No Refit)** | 20.000    | **0.591** | **1.000** | **0.065** | **4.100** |
| Oracle-feature baseline    | 3.000     | 0.596     | 1.000     | 0.000     | 3.000     |
| True-parameter oracle      | 3.000     | 0.626     | 1.000     | 0.000     | 3.000     |

**[시나리오 3] 약한 신호 환경 ($a=1.2$)**

| **방법론**                    | **사용 차원** | **ARI**   | **TPR**   | **FPR**   | $\hat{S}$ |
| -------------------------- | --------- | --------- | --------- | --------- | --------- |
| K-means                    | 20.000    | 0.297     | -         | -         | -         |
| PCA + K-means              | 14.000    | 0.290     | -         | -         | -         |
| GMM (Unpenalized)          | 20.000    | 0.388     | -         | -         | -         |
| Sparse K-means             | 19.800    | 0.393     | 1.000     | 0.988     | 19.800    |
| $\rightarrow$ + Refit      | 19.800    | 0.390     | -         | -         | -         |
| Naive Lasso                | 3.100     | 0.412     | 1.000     | 0.006     | 3.100     |
| $\rightarrow$ + Refit      | 3.100     | 0.466     | -         | -         | -         |
| **Proposed HP (No Refit)** | 20.000    | **0.448** | **1.000** | **0.029** | **3.500** |
| Oracle-feature baseline    | 3.000     | 0.463     | 1.000     | 0.000     | 3.000     |
| True-parameter oracle      | 3.000     | 0.498     | 1.000     | 0.000     | 3.000     |

### 5.3 해석

첫째, 기본 환경에서는 single-stage HP가 feature-oracle baseline에 근접하는 양상을 보인다. 예를 들어 $a=1.4$ 구간에서 HP의 ARI는 0.591이고 feature-oracle baseline은 0.596이다. 이는 HP가 별도 refit 없이도 상당히 안정적인 군집 복원 성능을 낼 수 있음을 시사한다.

둘째, Naive Lasso는 변수 선택 자체는 양호하게 보일 수 있으나, 초기 ARI가 낮고 refit 이후에야 성능이 회복되는 경향을 보인다. 이는 element-wise shrinkage가 수축 편향에 더 민감하다는 점을 보여준다.

셋째, 다만 기본 환경에서도 true-parameter oracle은 HP보다 여전히 높다. 따라서 현재 결과는 오라클 결과와 유사한 경향으로 볼 수 있다.

---

## 6. 고차원 환경 ($p=100, q=5$)

### 6.1 실험 세팅

표본 수 $n=300$, 차원 $p=100$, 정답 변수 $q=5$, 신호 강도 $a\in{1.6,1.4,1.2}$이다.

### 6.2 시나리오별 결과표 (평균)

**[시나리오 1] 신호 환경 ($a=1.6$)**

| **방법론**                    | **사용 차원** | **ARI**   | **TPR**   | **FPR**   | $\hat{S}$ |
| -------------------------- | --------- | --------- | --------- | --------- | --------- |
| K-means                    | 100.000   | 0.479     | -         | -         | -         |
| PCA + K-means              | 56.200    | 0.494     | -         | -         | -         |
| GMM (Unpenalized)          | 100.000   | 0.000     | -         | -         | -         |
| Sparse K-means             | 97.400    | 0.846     | 1.000     | 0.973     | 97.400    |
| $\rightarrow$ + Refit      | 97.400    | 0.000     | -         | -         | -         |
| Naive Lasso                | 5.000     | 0.798     | 1.000     | 0.000     | 5.000     |
| $\rightarrow$ + Refit      | 5.000     | 0.845     | -         | -         | -         |
| **Proposed HP (No Refit)** | 100.000   | **0.843** | **1.000** | **0.012** | **6.100** |
| Oracle-feature baseline    | 5.000     | 0.845     | 1.000     | 0.000     | 5.000     |
| True-parameter oracle      | 5.000     | 0.851     | 1.000     | 0.000     | 5.000     |

**[시나리오 2] 중간 신호 환경 ($a=1.4$)**

| **방법론**                    | **사용 차원** | **ARI**   | **TPR**   | **FPR**   | $\hat{S}$ |
| -------------------------- | --------- | --------- | --------- | --------- | --------- |
| K-means                    | 100.000   | 0.395     | -         | -         | -         |
| PCA + K-means              | 56.700    | 0.394     | -         | -         | -         |
| GMM (Unpenalized)          | 100.000   | 0.000     | -         | -         | -         |
| Sparse K-means             | 88.800    | 0.777     | 1.000     | 0.882     | 88.800    |
| $\rightarrow$ + Refit      | 88.800    | 0.075     | -         | -         | -         |
| Naive Lasso                | 6.800     | 0.652     | 1.000     | 0.019     | 6.800     |
| $\rightarrow$ + Refit      | 6.800     | 0.784     | -         | -         | -         |
| **Proposed HP (No Refit)** | 100.000   | **0.790** | **1.000** | **0.002** | **5.200** |
| Oracle-feature baseline    | 5.000     | 0.793     | 1.000     | 0.000     | 5.000     |
| True-parameter oracle      | 5.000     | 0.800     | 1.000     | 0.000     | 5.000     |

**[시나리오 3] 약한 신호 환경 ($a=1.2$)**

| **방법론**                    | **사용 차원** | **ARI**   | **TPR**   | **FPR**   | $\hat{S}$ |
| -------------------------- | --------- | --------- | --------- | --------- | --------- |
| K-means                    | 100.000   | 0.350     | -         | -         | -         |
| PCA + K-means              | 56.900    | 0.342     | -         | -         | -         |
| GMM (Unpenalized)          | 100.000   | 0.000     | -         | -         | -         |
| Sparse K-means             | 98.900    | 0.691     | 1.000     | 0.988     | 98.900    |
| $\rightarrow$ + Refit      | 98.900    | 0.000     | -         | -         | -         |
| Naive Lasso                | 5.100     | 0.439     | 1.000     | 0.001     | 5.100     |
| $\rightarrow$ + Refit      | 5.100     | 0.700     | -         | -         | -         |
| **Proposed HP (No Refit)** | 100.000   | **0.700** | **1.000** | **0.008** | **5.800** |
| Oracle-feature baseline    | 5.000     | 0.698     | 1.000     | 0.000     | 5.000     |
| True-parameter oracle      | 5.000     | 0.712     | 1.000     | 0.000     | 5.000     |

### 6.3 해석

차원이 $p=100$으로 늘어나자 unpenalized GMM은 현재 working implementation에서 사실상 0에 가까운 ARI를 보였고, Sparse K-means는 많은 노이즈 변수를 남긴 채 refit을 수행하면서 크게 불안정해졌다.

약신호 $a=1.2$ 구간에서 HP의 ARI는 0.700, feature-oracle baseline은 0.698, true-parameter oracle은 0.712이다. 따라서 HP가 **feature-oracle baseline을 소폭 상회하는 finite-sample regularization benefit**을 보였다고는 말할 수 있지만, true-parameter oracle보다 여전히 낮으므로 “오라클을 능가했다”고 해석하는 것은 적절하지 않다.

즉, 이 구간의 올바른 해석은 다음과 같다.

- HP는 strong regularization과 구조적 제약 덕분에 **정답 변수만 알고 무벌점으로 다시 적합한 baseline**보다 더 안정적일 수 있다.
    
- 그러나 true-parameter oracle이 더 높기 때문에, 현재 결과는 **feature-oracle baseline 대비 상대적 이점**이지 절대적 상한선 초과가 아니다.
    

---

## 7. 초고차원 환경 ($p=300, q=5$)

### 7.1 실험 세팅

표본 수 $n=300$, 차원 $p=300$, 정답 변수 $q=5$인 ultra-high-dimensional setting이다.

### 7.2 시나리오별 결과표 (평균)

**[시나리오 1] 신호 환경 ($a=1.6$)**

| **방법론**                    | **사용 차원** | **ARI**   | **TPR**   | **FPR**   | $\hat{S}$ |
| -------------------------- | --------- | --------- | --------- | --------- | --------- |
| K-means                    | 300.000   | 0.389     | -         | -         | -         |
| PCA + K-means              | 115.600   | 0.392     | -         | -         | -         |
| GMM (Unpenalized)          | 300.000   | 0.385     | -         | -         | -         |
| Sparse K-means             | 195.700   | 0.845     | 1.000     | 0.646     | 195.700   |
| $\rightarrow$ + Refit      | 195.700   | 0.169     | -         | -         | -         |
| Naive Lasso                | 5.000     | 0.798     | 1.000     | 0.000     | 5.000     |
| $\rightarrow$ + Refit      | 5.000     | 0.845     | -         | -         | -         |
| **Proposed HP (No Refit)** | 300.000   | **0.832** | **1.000** | **0.009** | **7.600** |
| Oracle-feature baseline    | 5.000     | 0.845     | 1.000     | 0.000     | 5.000     |
| True-parameter oracle      | 5.000     | 0.851     | 1.000     | 0.000     | 5.000     |

**[시나리오 2] 중간 신호 환경 ($a=1.4$)**

| **방법론**                    | **사용 차원** | **ARI**   | **TPR**   | **FPR**   | $\hat{S}$ |
| -------------------------- | --------- | --------- | --------- | --------- | --------- |
| K-means                    | 300.000   | 0.349     | -         | -         | -         |
| PCA + K-means              | 115.700   | 0.345     | -         | -         | -         |
| GMM (Unpenalized)          | 300.000   | 0.377     | -         | -         | -         |
| Sparse K-means             | 179.300   | 0.782     | 1.000     | 0.591     | 179.300   |
| $\rightarrow$ + Refit      | 179.300   | 0.071     | -         | -         | -         |
| Naive Lasso                | 5.100     | 0.620     | 1.000     | 0.000     | 5.100     |
| $\rightarrow$ + Refit      | 5.100     | 0.776     | -         | -         | -         |
| **Proposed HP (No Refit)** | 300.000   | **0.771** | **1.000** | **0.009** | **7.600** |
| Oracle-feature baseline    | 5.000     | 0.773     | 1.000     | 0.000     | 5.000     |
| True-parameter oracle      | 5.000     | 0.800     | 1.000     | 0.000     | 5.000     |

**[시나리오 3] 약한 신호 환경 ($a=1.2$)**

|**방법론**|**사용 차원**|**ARI**|**TPR**|**FPR**| $\hat{S}$ |
|---|---|---|---|---|---|
|K-means|300.000|0.233|-|-| -         |
|PCA + K-means|115.900|0.240|-|-| -         |
|GMM (Unpenalized)|300.000|0.221|-|-| -         |
|Sparse K-means|230.200|0.641|1.000|0.763| 230.200   |
|$\rightarrow$ + Refit|230.200|0.054|-|-| -         |
|Naive Lasso|5.400|0.452|1.000|0.001| 5.400     |
|$\rightarrow$ + Refit|5.400|0.615|-|-| -         |
|**Proposed HP (No Refit)**|300.000|**0.652**|**1.000**|**0.004**| **6.300** |
|Oracle-feature baseline|5.000|0.625|1.000|0.000| 5.000     |
|True-parameter oracle|5.000|0.680|1.000|0.000| 5.000     |

### 7.3 해석

초고차원 환경에서는 HP가 선택 성능 측면에서 여전히 매우 낮은 FPR을 유지하며 안정적인 패턴을 보인다. 특히 $a=1.2$ 구간에서 HP의 ARI는 0.652, feature-oracle baseline은 0.625, true-parameter oracle은 0.680이다.

이 결과는 다음처럼 해석하는 것이 적절하다.

- HP가 feature-oracle baseline보다 높은 값을 보인 것은, 고차원 유한표본 환경에서 regularization이 추정 분산을 낮추는 **finite-sample stabilization effect** 때문일 수 있다.
    
- 그러나 true-parameter oracle은 여전히 0.680으로 더 높다.
    
- 따라서 여기서도 “오라클을 능가했다”고 말하기보다, **feature-oracle baseline을 상회하는 empirical regularization benefit이 관찰되었다**고 정리하는 편이 더 정확하다.
    

---

## 8. 시뮬레이션 종합 정리

기본 환경 $(p=20)$부터 초고차원 환경 $(p=300)$까지의 pilot Monte Carlo를 종합하면, 제안 방법론은 다음과 같은 경향을 보였다.

1. **낮은 FPR과 높은 TPR 유지** HP는 저차원·고차원·초고차원 환경 모두에서 mean-heterogeneity variables를 비교적 안정적으로 복원하는 경향을 보였다.
    
2. **Naive Lasso 대비 구조적 안정성** Naive Lasso는 선택 자체는 잘 될 수 있으나, 수축 편향으로 인해 refit에 더 강하게 의존한다. 반면 HP는 single-stage에서도 강한 성능을 보인다.
    
3. **feature-oracle baseline 대비 경쟁력** 여러 구간에서 HP는 feature-oracle baseline에 매우 근접하거나 이를 소폭 상회하였다. 이는 structured regularization의 finite-sample benefit 가능성을 시사한다.
    
4. **true-parameter oracle과의 차이 유지** true-parameter oracle은 여전히 가장 높은 기준으로 남아 있다. 따라서 현재 결과는 near-oracle empirical behavior로 해석해야 하며, asymptotic guarantee로 일반화해서는 안 된다.
    
5. **pilot 결과라는 점을 명시할 필요** 현재는 $R=10$ 반복의 pilot Monte Carlo 결과이므로, 더 큰 반복 수와 보조 시나리오에서 재검증이 필요하다.
    

---

## 9. 향후 진행 계획 및 후속 시뮬레이션

참고 논문의 simulation design과 supplementary experiments를 반영하면, 최종 논문에는 다음 보완이 필요하다.

1. **반복 수 확대** 현재 $R=10$ pilot Monte Carlo를 $R=200$ 또는 $R=500$ 수준으로 확대하고, 평균과 표준오차를 함께 보고한다.
    
2. **correlated predictors 실험**
    
    $$\Sigma_{uv}=0.5^{|u-v|}$$
    
    형태의 상관구조를 도입하여 selection stability를 점검한다.
    
3. **unequal mixing 실험**
    
    $$(\pi_1,\pi_2,\pi_3)=(0.25,0.25,0.5)$$
    
    와 같은 불균형 mixing에서 강건성을 확인한다.
    
4. **all-heterogeneous setting** 모든 informative variables가 heterogeneous한 경우에는 heterogeneity pursuit의 이점이 약해질 수 있으므로, 이 구간도 점검한다.
    
5. **adaptive vs non-adaptive 비교** 참고 논문과 동일한 메시지, 즉 adaptive penalty가 non-adaptive보다 더 안정적이라는 점을 mean-mixture setting에서도 검토한다.
    
6. **군집 수 선택** 현재는 $K$를 고정했지만, 최종 논문에서는 $K$와 penalty parameter를 함께 선택하는 방향으로 확장한다.
    

---

## Appendix A. $\Sigma = I_p$ 환경의 보조 시뮬레이션

### A.1 실험 목적

본 부록에서는 본문과 달리 모든 변수의 분산이 1인 항등행렬 공분산 환경에서 시뮬레이션을 수행하였다. 목적은 “분산이 동일할 때도 본문과 같은 경향이 유지되는가”를 확인하는 것이다. 현재 결과는 흥미로운 보조 관찰로 해석하며, 이를 곧바로 일반적인 “분산-적응성의 실증적 입증”으로 확대 해석하지는 않는다.

### A.2 저차원 환경 ($p=20,q=3$) 결과표

**[시나리오 1] 신호 환경 ($a=1.6$)**

| **방법론**                    | **사용 차원** | **ARI**   | **TPR**   | **FPR**   | $\hat{S}$ |
| -------------------------- | --------- | --------- | --------- | --------- | --------- |
| K-means                    | 20.000    | 0.659     | -         | -         | -         |
| PCA + K-means              | 13.900    | 0.644     | -         | -         | -         |
| GMM (Unpenalized)          | 20.000    | 0.523     | -         | -         | -         |
| Sparse K-means             | 19.500    | 0.683     | 1.000     | 0.971     | 19.500    |
| $\rightarrow$ + Refit      | 19.500    | 0.581     | -         | -         | -         |
| Naive Lasso                | 5.600     | 0.581     | 1.000     | 0.153     | 5.600     |
| $\rightarrow$ + Refit      | 5.600     | 0.633     | -         | -         | -         |
| **Proposed HP (No Refit)** | 20.000    | **0.674** | **1.000** | **0.053** | **3.900** |
| Oracle-feature baseline    | 3.000     | 0.682     | 1.000     | 0.000     | 3.000     |
| True-parameter oracle      | 3.000     | 0.705     | 1.000     | 0.000     | 3.000     |

**[시나리오 2] 중간 신호 환경 ($a=1.4$)**

| **방법론**                    | **사용 차원** | **ARI**   | **TPR**   | **FPR**   | $\hat{S}$ |
| -------------------------- | --------- | --------- | --------- | --------- | --------- |
| K-means                    | 20.000    | 0.575     | -         | -         | -         |
| PCA + K-means              | 14.000    | 0.591     | -         | -         | -         |
| GMM (Unpenalized)          | 20.000    | 0.457     | -         | -         | -         |
| Sparse K-means             | 19.600    | 0.601     | 1.000     | 0.976     | 19.600    |
| $\rightarrow$ + Refit      | 19.600    | 0.479     | -         | -         | -         |
| Naive Lasso                | 3.400     | 0.438     | 1.000     | 0.024     | 3.400     |
| $\rightarrow$ + Refit      | 3.400     | 0.569     | -         | -         | -         |
| **Proposed HP (No Refit)** | 20.000    | **0.590** | **1.000** | **0.118** | **5.000** |
| Oracle-feature baseline    | 3.000     | 0.596     | 1.000     | 0.000     | 3.000     |
| True-parameter oracle      | 3.000     | 0.626     | 1.000     | 0.000     | 3.000     |

**[시나리오 3] 약 신호 환경 ($a=1.2$)**

| **방법론**                    | **사용 차원** | **ARI**   | **TPR**   | **FPR**   | $\hat{S}$ |
| -------------------------- | --------- | --------- | --------- | --------- | --------- |
| K-means                    | 20.000    | 0.404     | -         | -         | -         |
| PCA + K-means              | 14.000    | 0.408     | -         | -         | -         |
| GMM (Unpenalized)          | 20.000    | 0.388     | -         | -         | -         |
| Sparse K-means             | 19.600    | 0.457     | 1.000     | 0.976     | 19.600    |
| $\rightarrow$ + Refit      | 19.600    | 0.393     | -         | -         | -         |
| Naive Lasso                | 3.300     | 0.411     | 1.000     | 0.018     | 3.300     |
| $\rightarrow$ + Refit      | 3.300     | 0.465     | -         | -         | -         |
| **Proposed HP (No Refit)** | 20.000    | **0.445** | **1.000** | **0.076** | **4.300** |
| Oracle-feature baseline    | 3.000     | 0.463     | 1.000     | 0.000     | 3.000     |
| True-parameter oracle      | 3.000     | 0.498     | 1.000     | 0.000     | 3.000     |

### A.3 고차원 환경 ($p=100,q=5$) 결과표

**[시나리오 1] 신호 환경 ($a=1.6$)**

| **방법론**                    | **사용 차원** | **ARI**   | **TPR**   | **FPR**   | $\hat{S}$  |
| -------------------------- | --------- | --------- | --------- | --------- | ---------- |
| K-means                    | 100.000   | 0.802     | -         | -         | -          |
| PCA + K-means              | 55.800    | 0.794     | -         | -         | -          |
| GMM (Unpenalized)          | 100.000   | 0.000     | -         | -         | -          |
| Sparse K-means             | 93.900    | 0.843     | 1.000     | 0.936     | 93.900     |
| $\rightarrow$ + Refit      | 93.900    | 0.000     | -         | -         | -          |
| Naive Lasso                | 7.100     | 0.797     | 1.000     | 0.022     | 7.100      |
| $\rightarrow$ + Refit      | 7.100     | 0.845     | -         | -         | -          |
| **Proposed HP (No Refit)** | 100.000   | **0.828** | **1.000** | **0.085** | **13.100** |
| Oracle-feature baseline    | 5.000     | 0.845     | 1.000     | 0.000     | 5.000      |
| True-parameter oracle      | 5.000     | 0.851     | 1.000     | 0.000     | 5.000      |

**[시나리오 2] 중간 신호 환경 ($a=1.4$)**

| **방법론**                    | **사용 차원** | **ARI**   | **TPR**   | **FPR**   | $\hat{S}$  |
| -------------------------- | --------- | --------- | --------- | --------- | ---------- |
| K-means                    | 100.000   | 0.663     | -         | -         | -          |
| PCA + K-means              | 56.200    | 0.681     | -         | -         | -          |
| GMM (Unpenalized)          | 100.000   | 0.000     | -         | -         | -          |
| Sparse K-means             | 95.300    | 0.782     | 1.000     | 0.951     | 95.300     |
| $\rightarrow$ + Refit      | 95.300    | 0.000     | -         | -         | -          |
| Naive Lasso                | 5.700     | 0.588     | 1.000     | 0.007     | 5.700      |
| $\rightarrow$ + Refit      | 5.700     | 0.794     | -         | -         | -          |
| **Proposed HP (No Refit)** | 100.000   | **0.767** | **1.000** | **0.067** | **11.400** |
| Oracle-feature baseline    | 5.000     | 0.793     | 1.000     | 0.000     | 5.000      |
| True-parameter oracle      | 5.000     | 0.800     | 1.000     | 0.000     | 5.000      |

**[시나리오 3] 약 신호 환경 ($a=1.2$)**

| **방법론**                    | **사용 차원** | **ARI**   | **TPR**   | **FPR**   | $\hat{S}$ |
| -------------------------- | --------- | --------- | --------- | --------- | --------- |
| K-means                    | 100.000   | 0.525     | -         | -         | -         |
| PCA + K-means              | 56.700    | 0.520     | -         | -         | -         |
| GMM (Unpenalized)          | 100.000   | 0.000     | -         | -         | -         |
| Sparse K-means             | 97.900    | 0.693     | 1.000     | 0.978     | 97.900    |
| $\rightarrow$ + Refit      | 97.900    | 0.000     | -         | -         | -         |
| Naive Lasso                | 5.700     | 0.440     | 1.000     | 0.007     | 5.700     |
| $\rightarrow$ + Refit      | 5.700     | 0.696     | -         | -         | -         |
| **Proposed HP (No Refit)** | 100.000   | **0.687** | **1.000** | **0.044** | **9.200** |
| Oracle-feature baseline    | 5.000     | 0.698     | 1.000     | 0.000     | 5.000     |
| True-parameter oracle      | 5.000     | 0.712     | 1.000     | 0.000     | 5.000     |

### A.4 보조 해석

$\Sigma=I_p$ 환경에서는 noise variables의 분산이 작아져 K-means류가 상대적으로 유리해지는 경향이 관찰된다. 또한 HP에서 $\hat S$가 다소 커지는 구간이 있었는데, 이는 분산 추정, penalty scale, tuning, finite-sample variability가 복합적으로 작용한 결과일 수 있다.

따라서 이 부록 결과는 “HP가 분산 정보에 의해 실제 선택 강도가 달라질 가능성”을 시사하는 흥미로운 보조 관찰로 해석하는 것이 적절하다. 다만 이를 일반적인 “분산-적응형 메커니즘의 실증적 입증”으로 단정하려면, 더 큰 반복 수와 $\lambda$ 재튜닝, 그리고 체계적인 민감도 분석이 추가로 필요하다.
