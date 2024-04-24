# 사례 연구 - 단계 3: 평가지표 계산을 위한 추상화

먼저 수정된 스크립트 [run_regressor_evaluation.py](run_regressor_evaluation.py)를 살펴봅니다.

지금까지 사용된 (단일)평가지표는 하드 코딩되어 있었습니다 (MAE). 
다른 평가지표을 계산하는 것은 평가 클래스를 직접 수정하여야만 가능했으며, 동적으로 평가지표을 변경할 방법이 없었습니다.

이 단계에서는 평가지표을 사용자 구성 가능하도록 일반화합니다.

사용자는 추상 기본 클래스 `Metric`에 의해 정의된 잘 정의된 인터페이스를 기반으로 하나 이상의 평가지표을 제공할 수 있어야 합니다. 각 평가지표은 다음을 정의합니다.

- 평가지표 계산 (메서드 `compute_value`)
- 더 큰 값 또는 더 작은 값이 더 나은 것으로 간주되는지 여부 (메서드 `is_larger_better`)
- 평가지표의 값을 보고하는 데 사용되는 평가지표의 이름 (메서드 `get_name`)
  
```python
class Metric(ABC):
    @abstractmethod
    def compute_value(self, y_ground_truth: np.ndarray, y_predicted: np.ndarray) -> float:
        """
        :param y_ground_truth: 실제 값
        :param y_predicted: 모델의 예측
        :return: 평가지표 값
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        :return: 평가지표의 이름
        """
        pass

    @abstractmethod
    def is_larger_better(self) -> bool:
        """
        :return: True이면 평가지표이 더 큰 값이 더 좋은 품질 평가지표이고,
            False이면 더 낮은 값이 더 좋은 오류 평가지표인 경우
        """
        pass
```
  
- MSE(`MetricMeanAbsError`) 및 결정 계수 $R^2$ (`MetricR2`)의 구현을 sklearn의 함수를 활용하여 제공합니다. 또한 사용자가 지정한 임계 값 이내의 절대 오차가 발생하는 상대적 빈도를 계산하는 사용자 정의 평가지표 (`MetricRelFreqErrorWithin`)을 추가합니다. 이는 특히 우리의 응용 프로그램에 유용할 수 있습니다.

- 평가는 이제 생성 시 주어진 모든 평가지표을 계산하고 결과 데이터 프레임을 첫 번째 평가지표에 따라 정렬합니다 (최상의 값이 첫 번째 행에 있도록), 평가지표의 이름을 열 이름으로 사용합니다.

새로운 출력은 다음과 같습니다:

```shell
INFO  2024-02-07 10:50:36,016 __main__:main - Results:
                                                    model        R²        MAE  RelFreqErrWithin[10]
2                              RandomForestRegressor()  0.233388  11.177063              0.520333
3  DecisionTreeRegressor(max_depth=2, random_state=42)  0.143303  11.911864              0.458000
1                   KNeighborsRegressor(n_neighbors=1) -0.538974  15.165667              0.445000
0                    LogisticRegression(max_iter=1000) -1.130103  17.485667              0.401000
```

이제 우리의 평가는 평가에 사용할 평가지표을 자유롭게 정의할 수 있게 되어 크게 유연해졌습니다.

[다음 단계](../02e-case-study-4-results-abstraction/README.md)
