# 사례 연구 - 단계 4: 평가 결과에 대한 추상화

먼저 수정된 스크립트 [run_regressor_evaluation.py](run_regressor_evaluation.py)를 확인하세요.

이전 단계에서 결과를 저장할 때 저수준 데이터 구조(판다스 DataFrame)를 사용해서 정보를 검색하기가 불편했습니다. 특히 데이터 프레임에서 최고의 모델 이름을 검색하는 것은 어려웠는데 이 단계에서는 평가 결과에 대한 추상화를 도입해 검색을 더 편리하게 만듭니다. 이로써 우리의 사례 연구를 마무리합니다. 주목할 점은 다음과 같습니다.

- 여전히 데이터 프레임에 액세스하여 보고를 위해 인쇄 할 수 있습니다.
- 또한 이제 최고의 모델 이름 (메소드 `get_best_model_name`) 및 해당 메트릭 값 (메소드 `get_best_metric_value`)을 검색할 수 있습니다.
- 우리의 평가 코드는 이제 유연하게 매개 변수화 되었습니다. 쉽게 데이터를 분활하는 매개변수를 지정할 수 있고 평가지표를 변경할 수 있습니다.

```python
metrics = [MetricR2(), MetricMeanAbsError(), MetricRelFreqErrorWithin(10)]
ev = ModelEvaluation(X_scaled, y, metrics, test_size=0.2, random_seed=23)
```

- 평가 추상화는 재사용 가능한 구성 요소로 앞으로 완전히 다른 맥락에서 사용할 수도 있습니다(예 : 하이퍼 파라미터 최적화).
- 클래스의 매소드는 의미있는 개념을 나타냅니다. 각 메소드는 명확한 목적을 가지고 간결해야 유지 보수가 쉽습니다.


[다음: 일반적인 디자인 및 개발 원칙](../03-general-principles/README.md)