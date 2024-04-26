# 단계 8: 고수준 평가

이전 두 단계의 결과로 우리는 모델 파이프라인의 정의에 대한 완전한 선언적 의미론을 달성했습니다. 이와 비교하여 모델 평가는 여전히 상당히 원시적이며 완전히 절차적인 방식으로 작성되어 재사용 가능한 구성 요소로 구성되지 않았습니다. 우리는 모델 파이프라인에 적용한 것과 같은 원칙을 평가에도 적용해야 합니다. 즉, 우리는 절차적인 세부사항 대신 평가를 **선언적으로** 선언하고 싶습니다.


## 모델 평가


우리의 모델을 유효성 검사/평가하기 위해 고수준의 유틸리티 클래스를 사용합니다. 후속 단계에서는 이 클래스의 기능을 더 많이 사용할 것입니다. 지금은 그냥 몇 가지 호출을 자동화하여 우리가 원하는 것을 선언적으로 정의할 수 있게 해줍니다. 절차적인 세부사항에 시간을 낭비하지 않습니다.

이전:

```python
# 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, shuffle=True)

# 절차적인 방식으로 모델을 평가합니다.
for model in models:
    print(f"Evaluating model:\n{model}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
```

이후:

```python
# 평가에 사용할 매개변수를 선언하십시오, 즉 데이터를 어떻게 분할할지 지정하십시오 (분할 비율 및 랜덤 시드)
evaluator_params = VectorClassificationModelEvaluatorParams(fractional_split_test_fraction=0.3,
    fractional_split_random_seed=42,
    binary_positive_label=dataset.class_positive)

# 이러한 매개변수를 기반으로 모델을 평가하기 위해 고수준의 유틸리티 클래스를 사용합니다.
ev = ClassificationEvaluationUtil(io_data, evaluator_params=evaluator_params)
ev.compare_models(models, fit_models=True)
```

## 로깅

우리는 어차피 무엇이 일어나고 있는지를 추적하기에 충분하지 않은 `print` 문을 없앴습니다. `print` 문은 프로덕션 코드 (실제로는 대부분의 다른 코드도)에는 적합하지 않습니다. 왜냐하면 그것들은 상당히 유연하지 않기 때문입니다. 반면에 로깅 프레임워크를 사용하면 로깅의 정도를 완전히 제어할 수 있습니다 (즉, 어떤 패키지/모듈이 어느 수준에서 로그를 남길 수 있는지 정의할 수 있음) 그리고 로그가 어디에 끝나는지 유연하게 정의할 수 있습니다 (심지어 여러 장소에).

sensAI의 고수준 평가 클래스는 기본적으로 모든 중요한 단계를 로그로 남깁니다. 따라서 우리는 실제로 많은 로그 문을 작성할 필요가 없을 것입니다. 우리는 `Dataset` 클래스에 단일 로그 문 하나만 추가하기로 결정했습니다: 데이터를 결정하는 모든 관련 매개변수를 로그로 남기고, 이를 용이하게 하기 위해 sensAI의 `ToStringMixin`을 사용했습니다.

로깅을 활성화하기 위해 우리는 간단히 Python의 `logging` 패키지를 통해 로그 핸들러를 등록할 수 있지만, 우리는 확장된 대체품으로 `sensai.util.logging`을 사용하고 이를 단순화하기 위해 그것의 `run_main` 함수를 적용했습니다: 이것은 합리적인 기본값으로 `stdout`에 로깅을 설정하고, `main` 함수 실행 중에 발생할 수 있는 예외를 로그로 남길 수 있도록 합니다.

```python
if __name__ == '__main__':
    logging.run_main(main)
```

[로그 출력](output.txt)을 살펴보고 처음에 [가지고 있던 출력](../step02-dataset-representation/output.txt)과 비교해보세요.

## 이번 단계에서 다루는 원칙

- 로그를 광범위하게 기록하기.
- 선언적 의미론을 선호하기.

[다음 단계](../step09-tracking-experiments/README.md)