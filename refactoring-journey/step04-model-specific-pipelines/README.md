# 단계 4: 모델별 파이프라인

이번 단계에서는 더 유연한 코드를 만드는 작업을 해봅니다. 데이터 처리 파이프라인을 모델과 분리해서 다양한 실험을 할 수 있도록 하겠습니다. 여기서 중요한 점은 서로 다른 모델이 서로 다른 데이터 셋을 사용할 수 있다는 것입니다.

지금까지는 모든 모델이 동일한 데이터를 사용하고 있고 `StandardScaler`로 데이터 전처리를 했습니다. 이것은 분명히 문제가 있습니다. 왜냐면 일부 모델은 범주형 특성에서 더 좋은 성능을 보여줄 수 있고  또한 `StandardScaler`가 모든 특성에 대해 최적이 아닐 수 있습니다. 그러니 데이터 처리 파이프라인을 새로 만들어 언제든지 새로운 모델을 시도할 수 있는 유연성을 확보해야 합니다.

```python
class ModelFactory:
    COLS_USED_BY_ORIGINAL_MODELS = [COL_YEAR, *COLS_MUSICAL_DEGREES, COL_KEY, COL_MODE, COL_TEMPO, COL_TIME_SIGNATURE, COL_LOUDNESS,
        COL_DURATION_MS]

    @classmethod
    def create_logistic_regression_orig(cls):
        return Pipeline([
            ("project_scale", ColumnTransformer([("scaler", StandardScaler(), cls.COLS_USED_BY_ORIGINAL_MODELS)])),
            ("model", linear_model.LogisticRegression(solver='lbfgs', max_iter=1000))])
```

중요한 점은 파이프라인 구성 요소를 모델로 이동함으로써 원래 코드의 미묘한 문제도 해결되었다는 것입니다. (`StandardScaler`가 전체 데이터 세트에서 학습되는 문제입니다. 엄밀히 말하면 이는 데이터 누수입니다.)

```python
# 데이터를 훈련 및 테스트로 분할하기 전에 StandardScaler를 학습하는 것은 데이터 누수의 가능성이 있음
scaler = StandardScaler()
model_X = scaler.fit(X)
X_scaled = model_X.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42, test_size=0.3, shuffle=True)
```

훈련 데이터가 충분하게 전체 데이터를 대표한다면 이런 문제가 크게 문제가 되지 않을 수 있습니다. 그러나 일반적으로 결과를 의미 있게 만들기 위해 훈련 세트를 훈련 과정에서 완전히 배제하는 것이 좋습니다.

기술적으로는 우리는 현재 고려 중인 네 가지 모델의 인스턴스를 생성할 수 있는 [모델 팩토리](songpop/model_factory.py)를 소개합니다. 우리는 이러한 팩토리 함수들을 원래 모델에 해당한다는 점을 나타내는 `_orig` 접미사를 사용해 명명했습니다.

## Scikit-learn 파이프라인이 최종 답안인가요?

`Scikit-learn` 파이프라인 객체를 사용하면 모델별 데이터 처리를 정의할 수 있지만 이 개념은 일시적인 해결책입니다. `Scikit-learn` 파이프라인 개념은 새로운 추상화 즉 `Scikit-learn` `fit/transform` 프로토콜를 사용하지만 코드가 여전히 절차적으로 작성되어 있기 때문입니다.

우리는 객체에 캡슐화된 연산을 연결하고, 이전에 언급한 프로토콜을 구현하는 객체를 사용했습니다. 즉, 특성 집합을 변경하거나 새로운 모델별 특성 값 변환을 추가하거나 정규화를 변경하는 것과 같은 모든 수정에는 새로운 파이프라인을 정의해야 합니다. 특성/전처리 조합 당 파이프라인의 조합이 아주 많기 때문에 이것들을 모두 수동으로 정의하는 것은 쉽지 않습니다.

다시 말해, 파이프라인 객체만 사용하는 것은 너무 "저수준"입니다. 우리는 우리가 하고자 하는 작업을 **선언**만 하면 되고 해당 파이프라인이 자동으로 구성하려 합니다. 즉, 각 모델에 대해 사용하려는 특성과 적용할 변환을 선언하고 해당 특성에 적절하게 적용할 파이프라인을 선언하고자 합니다.

다음 두 단계의 목표는 이미 이 기능을 제공하는 [sensAI](https://github.com/aai-institute/sensAI) 프레임워크를 소개하는 것입니다. sensAI는 사용자가 정의한 매개변수에 따라 데이터 처리 파이프라인을 자동으로 만들수 있습니다.

## 이 단계에서 다룬 원칙

- 적절한 추상화 찾기

[다음 단계](../step05-sensai/README.md)
