# 사례 연구 - 단계 2: 평가를 위한 추상화

먼저 수정된 스크립트 [run_regressor_evaluation.py](run_regressor_evaluation.py)를 살펴봅니다.

이 단계에서는 다음 인터페이스를 갖춘 `ModelEvaluation`이라는 추상화를 생성합니다. 이는 다음과 같은 `split` 기반 평가 작업을 나타냅니다.

```python
class ModelEvaluation:
    """
    회귀 모델을 평가하고 결과를 수집하는 기능을 지원합니다.
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series,
            test_size: float = 0.3, shuffle: bool = True, random_state: int = 42):
        """
        :param X: 입력 데이터
        :param y: 예측 대상
        :param test_size: 테스트 데이터의 비율
        :param shuffle: 데이터를 분할하기 전에 섞을지 여부
        :param random_state: 데이터를 섞을 때 사용할 랜덤 시드
        """

    def evaluate_model(self, model) -> float:
        """
        주어진 모델을 학습하고 평가하여 결과를 수집합니다.

        :param model: 평가할 모델
        :return: 평균 절대 오차 (MAE)
        """

    def get_results(self) -> pd.DataFrame:
        """
        :return: 모든 평가 결과를 포함하는 데이터 프레임
        """
```

- 이 추상화는 평가뿐만 아니라 데이터 분할도 처리합니다.
- 분할은 *매개변수화*될 수 있으며, 호출자가 테스트 세트의 상대적 크기, 랜덤 시드 및 섞기 여부를 변경할 수 있습니다.
- 데이터는 이제 평가 객체의 속성 내에 저장되므로 실제 평가 메서드는 더 적은 매개변수만 필요로 합니다. 여기에서는 모델만 전달하면 됩니다.
- 이 추상화는 잠재적으로 여러 평가를 수행한 결과를 수집하고 내부 상태에 저장하여 수집된 결과를 요청 시 검색할 수 있습니다.

최종 출력은 다음과 같습니다.

```
INFO  2024-02-07 10:48:38,676 __main__:main - Results:
                                                 model        MAE
0                    LogisticRegression(max_iter=1000)  17.485667
1                   KNeighborsRegressor(n_neighbors=1)  15.165667
2                              RandomForestRegressor()  11.169653
3  DecisionTreeRegressor(max_depth=2, random_state=42)  11.911864
```

**우리는 모듈성과 매개변수화를 크게 향상**시켰으며, 이 컴포넌트를 재사용할 가능성을 크게 높였습니다.

[다음 단계](../02d-case-study-3-metric-abstraction/README.md)