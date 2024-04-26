# 단계 12: 하이퍼파라미터 최적화

모델 매개변수는 데이터와 학습 알고리즘에 따라 자동으로 조정되지만, 하이퍼파라미터는 미리 설정되어 학습 중에 고정됩니다. 신경망의 레이어/뉴런 수나 트리 모델의 최대 깊이와 같은 하이퍼파라미터는 모델의 일반화 성능에 심각한 영향을 줄 수 있습니다. 하이퍼파라미터와 모델 성능 사이의 복잡한 관계는 항상 명확하지 않습니다. 따라서 체계적인 최적화는 수동 실험에서 누락될 수 있는 효과적인 조합을 발견할 수 있습니다.

이번 단계에서는 XGBoost 회귀 모델의 하이퍼파라미터 검색을 조사합니다. 우리의 요구에 맞는 한, 기존 라이브러리를 사용하는 것이 항상 좋은 생각입니다. 우리는 모델의 하이퍼파라미터를 조정하기 위해 잘 알려진 프레임워크인 [hyperopt](https://github.com/hyperopt/hyperopt/tree/master)를 사용하기로 결정했습니다. 이를 적용하기 위해서는 기본적으로 다음 두 가지를 정의해야 합니다:
- 하이퍼파라미터의 검색 공간
- 최소화될 목적 함수

`hyperopt`에서 검색 공간은 각 매개변수에 대한 원하는 범위를 설명하는 객체를 포함한 사전으로 정의됩니다. 우리는 과적합을 조절하는 데 사용되는 매개변수에 집중합니다. 이러한 매개변수의 자세한 설명은 [XGBoost 문서](https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster)를 참조하십시오.

```python
search_space = {
        'max_depth': hp.uniformint("max_depth", 3, 10),
        'gamma': hp.uniform('gamma', 0, 9),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.25, 1),
        'min_child_weight': hp.uniformint('min_child_weight', 1, 100),
    }
```

이번 탐색 과정에서는 휴리스틱하게 검색 공간을 탐색할 것입니다. 이전에 사용한 매개변수 구성이 평가되도록 보장하기 위해 `initial_space`에 매개변수 구성 목록을 추가로 지정합니다. 이는 검색 시작 시 고려될 것입니다.

특정 매개변수 구성에 대한 모델을 평가하기 위해 해당하는 매개변수 구성으로 XGBoost 모델을 생성하는 팩토리가 필요합니다. 이전 단계에서 이미 모델 팩토리를 정의했으므로, 이는 검색 공간 요소의 항목을 사전으로 제공하는 것으로 축소됩니다.

그러므로 우리는 이전 단계에서 이미 정의한 모델 팩토리를 사용하여 XGBoost 모델을 생성할 수 있습니다. 매개변수 검색 공간의 각 항목을 모델 팩토리에 전달하여 모델을 생성하면 됩니다. 

검색 알고리즘이 하이퍼파라미터 조합을 찾아내고, 초기에 `initial_space`에서 지정된 매개변수 구성을 고려하여 탐색을 시작할 것입니다. 이를 통해 이전에 사용한 매개변수 구성을 평가할 수 있게 됩니다.


```python
def create_model(search_space_element: Dict[str, Any]):
            return RegressionModelFactory.create_xgb(
                verbosity=1,
                max_depth=int(search_space_element['max_depth']),
                gamma=search_space_element['gamma'],
                reg_lambda=search_space_element['reg_lambda'],
                min_child_weight=int(search_space_element['min_child_weight']),
                colsample_bytree=search_space_element['colsample_bytree'])
```

이 팩토리를 기반으로 주어진 매개변수 구성에 대해 모델을 평가하는 실제 목적 함수를 정의할 수 있습니다.

```python
io_data = dataset.load_io_data()
metric = RegressionMetricRRSE()
evaluator_params = VectorRegressionModelEvaluatorParams(fractional_split_test_fraction=0.3, fractional_split_random_seed=21)
ev = RegressionEvaluationUtil(io_data, evaluator_params=evaluator_params)

def objective(search_space_element: Dict[str, Any]):
    log.info(f"Evaluating {search_space_element}")
    model = create_model(search_space_element)
    loss = ev.perform_simple_evaluation(model).get_eval_stats().compute_metric_value(metric)
    log.info(f"Loss[{metric.name}]={loss}")
    return {'loss': loss, 'status': hyperopt.STATUS_OK}
```

저희는 *루트 상대 제곱 오차* (RRSE)를 최소화할 메트릭으로 선택했고, sensAI의 익숙한 평가 유틸리티 클래스를 사용하여 이 값을 계산했습니다. 다른 난수 시드를 사용하여 단일 분할에 기반한 간단한 평가를 적용했는데, 이는 편향을 피하는 데 도움이 되며 나중에 평가된 결과가 모델 품질을 정확하게 반영할 것을 보장합니다 (나중에 사용되는 분할과 일치하지 않기 때문입니다).

10시간 동안 검색을 실행한 후, 다음과 같은 최적의 매개변수를 얻었습니다:

```python
{
 'colsample_bytree': 0.9869550725977663,
 'gamma': 8.022497033174522,
 'max_depth': 10,
 'min_child_weight': 48.0,
 'reg_lambda': 0.3984639652186364
}
```

따라서 해당 모델을 공장에 추가하고, 다음 단계에서 그 품질을 철저히 평가할 것입니다.

```python
@classmethod
def create_xgb_meanpop_opt(cls):
    params = {'colsample_bytree': 0.9869550725977663,
              'gamma': 8.022497033174522,
              'max_depth': 10,
              'min_child_weight': 48.0,
              'reg_lambda': 0.3984639652186364} 
    return cls.create_xgb("-meanPop-opt", add_features=[FeatureName.MEAN_ARTIST_POPULARITY], **params)
```

[다음 단계](../step12-cross-validation/README.md)
