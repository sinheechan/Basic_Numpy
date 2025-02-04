## 로지스틱 회귀분석 (Logistic Regression)

<br/>

## 1. 배경 

<br/>

- 로지스틱 회귀분석은 이진분류란, 문제에 대한 정답을 두 가지 답 중 하나로 분류하는 문제에 활용된다.

- 선형 회귀는 Outlier(이상치)에 약하기 때문에 분류 문제에 잘 동작하지 않는다.

- 따라서, 이진 분류 문제를 해결하기 위한 회귀 방법 중 하나가, 로지스틱 회귀(Logistic Regression)이다.

<br/>

### ① 기초 개념

<br/>

- 선형회귀에서는 가설로 직선을 사용한다. ⇒ 하지만, 직선은 이상치를 분류하지 못한다.

- 우리가 원하는 값은 0과 1 사이의 값이지만, 직선으로 예측값을 추출하면 보통 0보다 작거나 1보다 큰 값이 나온다.

- 따라서 로지스틱 회귀에서는 가설로 시그모이드 함수를 사용한다
  - 선형 회귀분석 (Linear Regression) : 직선 / MSE ( Mean Square Error ) : 평균제곱오차
  - 로지스틱 회귀분석 (Logistic Regression) : Sigmoid / ( Binary Cross Entropy ) : 이진 교차 엔트로피

<br/>

### ② 시그모이드 함수

<br/>

- 시그모이드 함수를 사용한다면, x의 값이 작을 때의 예측값은 1/(1+∂) 으로 0 에 수렴할 것이며, x의 값이 클 때의 예측값은 1/(1+0) 으로 1 에 수렴할 것이다.
    
- 따라서, 예측 값을 0 과 1 사이의 값으로 추출할 수 있게 된다.

<br/>

![Logistic Regression](image/Logistic%20Regression_0.png)

<br/>

### ③ 비용함수

<br/>

- 선형회귀에서는 직선을 가설로 사용하고, mse(mean square error)를 비용 함수로 사용한다.

- 하지만, 로지스틱 회귀(Logistic Regression)에서는 가설로 시그모이드(Sigmoid) 함수를 사용하기 때문에, mse 로는 비용 함수를 구하기 어렵다.

- 만약, 가설로 시그모이드(Sigmoid) 함수를 사용하고, 비용 함수로 mse 를 사용한다면 아래와 같은 결과가 나타날 것이다.

<br/>

- 아래 그림처럼, 평야가 많아지기 때문에, 아무리 학습을 많이해도,
  mse 를 사용해 구한 cost 함수에서는 gradient descent(경사 하강) 알고리즘이 제대로 동작하지 않게 된다.
    
- 이를 해결하기 위해, Logistic Regression 에서는 비용함수로 Binary Cross Entropy(이진 교차 엔트로피) 를 사용한다.

<br/>

![Logistic Regression](image/Logistic%20Regression_1.png)

<br/>

### ④ 이진 교차 엔트로피 : Binary Cross Entropy

<br/>

- H(x) 는 시그모이드 함수로부터 나온 0 과 1 사이의 노드 예측값이다.

- Binary Cross Entropy 는 cost 를 계산할 때, 노드의 실제 값에 따라 다른 방식으로 오차를 구한다.

  - n : 분류 노드 개수
    
  - 노드가 1개 일 땐, 단일 이진 분류
    
  - 노드가 여러 개 일 땐, 멀티 이진 분류이다.

- 아래 그림은 실제 값과 예측값에 따른 비용함수이다.

  - 실제 값(y)이 1 일 때, 손실은 -log(H(x)) 로 계산된다.
    
  - log(H(x))가 작아지기 위해선, log(H(x))값이 커져야하기 때문에, H(x) 값이 커져야한다.

  - H(x)가 1일 때 log(H(x))는 0이며, 0일 때 log(H(x))는 -무한대이기 때문이다.

  - 따라서, 실제 값(y)이 1일 땐, H(x)가 1이 되도록 진행된다.

  - 실제 값(y)이 0 일 때, 손실은 -log(1-H(x)) 로 계산된다.

  - log(1-H(x))가 작아지기 위해선, log(1-H(x))값이 커져야하기 때문에, H(x) 값이 작아져야한다.

  - H(x)가 1일 때 log(1-H(x))는 -무한대이며, 0일 때 log(1-H(x))는 0이기 때문이다.

  - 따라서, 실제 값(y)이 0일 땐, H(x)가 0이 되도록 진행된다.

<br/>

![Logistic Regression](image/Logistic%20Regression_2.png)
