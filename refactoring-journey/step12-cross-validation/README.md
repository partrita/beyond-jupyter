# 단계 11: 교차 검증

이번 단계에서는 실행 가능한 스크립트를 확장하여 선택적으로 교차 검증을 적용하고, 한 번의 분할이 아닌 여러 분할에서 모델을 평가합니다. 평가 유틸리티 클래스가 직접 이를 지원하므로 변경 사항은 최소화됩니다.

```python
evaluator_params = VectorClassificationModelEvaluatorParams(fractional_split_test_fraction=0.3,
    binary_positive_label=dataset.class_positive)
cross_validator_params = VectorModelCrossValidatorParams(folds=3)
ev = ClassificationEvaluationUtil(io_data, evaluator_params=evaluator_params, cross_validator_params=cross_validator_params)
ev.compare_models(models, tracked_experiment=tracked_experiment, result_writer=result_writer, use_cross_validation=use_cross_validation)
```

그러나 교차 검증을 적용하는 것은 다른 실험 정의를 의미하므로 실험 이름의 정의를 이에 맞게 조정해야합니다.

```python
experiment_name = TagBuilder("popularity-classification", dataset.tag()) \
    .with_conditional(use_cross_validation, "CV").build()
```

특히, 이전 하이퍼파라미터 최적화 단계에서 검색한 XGBoost 모델의 성능에 관심이 있습니다. 해당 모델의 이름은 `XGBoost-meanPop-opt`입니다. 3-fold 교차 검증을 실행한 결과는 다음과 같습니다:

```
                     mean[MAE]  std[MAE]   mean[MSE]  std[MSE]  mean[R2]   std[R2]  mean[RMSE]  std[RMSE]  mean[RRSE]  std[RRSE]  mean[StdDevAE]  std[StdDevAE]
model_name                                                                                                                                                     
Linear                8.168991  0.005480  113.561908  0.114245  0.549987  0.001027   10.656542   0.005359    0.670830   0.000765        6.843204       0.005177
XGBoost               7.112810  0.012424   89.133103  0.260062  0.646790  0.001444    9.441023   0.013768    0.594313   0.001215        6.208118       0.012173
XGBoost-meanPop       5.833350  0.013290   66.385235  0.335167  0.736933  0.001612    8.147688   0.020553    0.512898   0.001570        5.688307       0.015839
XGBoost-meanPop-opt   5.722164  0.007521   64.473070  0.206048  0.744511  0.001108    8.029502   0.012825    0.505458   0.001095        5.632915       0.011055

```

위 결과를 통해 우리는 `XGBoost-meanPop`에서 조금의 성능 향상을 확인 할 수 있습니다.

[다음 단계](../step13-deployment/README.md)