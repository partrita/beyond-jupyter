# 단계 9: 실험 추적

우리가 이전 단계에서 소개한 고수준 평가 클래스는 다양한 로그를 통해 우리의 실험을 더 투명하게 만들었습니다.

이번 단계에서는 더 나아가서 우리의 실험 결과를 미래 참고를 위해 명시적으로 추적하는 메커니즘을 소개할 것입니다.

sensAI는 `MLFlowTrackedExperiment` 구현을 통해 mlflow를 지원합니다. 또한 `ResultWriter` 클래스를 통해 파일 시스템으로 직접 로깅을 지원합니다. 우리는 [main script](run_classifier_evaluation.py)에 이 두 가지 옵션을 (중복으로) 추가했습니다:

```python
    # set up (dual) tracking
    experiment_name = f"popularity-classification_{dataset.tag()}"
    run_id = datetime_tag()
    tracked_experiment = MLFlowExperiment(experiment_name, tracking_uri="", context_prefix=run_id + "_",
        add_log_to_all_contexts=True)
    result_writer = ResultWriter(os.path.join("results", experiment_name, run_id))
    logging.add_file_logger(result_writer.path("log.txt"))

    ...

    ev.compare_models(models, tracked_experiment=tracked_experiment, result_writer=result_writer)
```
결과의 의존성이 적절하게 고려되도록하기 위해, 관련된 데이터 집합 매개변수를 포함하는 간결한 태그를 실험 이름에 첨부합니다. 이를 위해 너무 많은 말을 하지 않고도 편리하게 구성할 수 있도록 `TagBuilder`를 사용합니다.

mlflow 수준에서 두 가지 개념을 구분합니다.
  - *실험*, 즉 결과를 비교할 목적으로 사용되는 결과의 컨테이너입니다. 즉, 우리의 경우 특정 예측 작업과 사용되는 데이터 집합에 의해 결정됩니다. 예를 들어, 데이터 집합을 10000개의 샘플과 기본이 아닌 랜덤 시드 23으로 구성하면 데이터 집합 태그는 `numSamples10000-seed23`이 됩니다.
  - *실행*, 특정 모델에 대한 개별 결과를 보유하는 것입니다. sensAI 용어로는 *추적 컨텍스트*라고합니다. 모델의 식별자로는 현재 날짜 및 시간과 모델 이름을 결합한 태그를 사용합니다 (이는 스크립트의 여러 실행에서 고유해야합니다). 예를 들어, 생성된 실행 이름은 `20230808-114244_LogisticRegression`일 수 있습니다.

메인 스크립트를 실행하면 다음이 발생합니다.
  - ![](res/results-folder.png)
  - 모델 설명, 메트릭, 이미지 (혼동 행렬을 보여주는 이미지) 및 로그를 `ResultWriter`에 지정된 폴더에 저장합니다.
  - ![](res/mlflow.png)
  - 동일한 메타 데이터, 메트릭, 로그 및 이미지를 로컬 mlflow 데이터 저장소에 저장합니다 (URI를 지정하지 않았으므로 서버 없음). `mlflow ui`를 실행하여 웹 인터페이스에서 결과를 편리하게 검사할 수 있도록 서버를 시작할 수 있습니다. 다음은 스크린샷입니다.

이제 스크립트의 임의의 실행에서 구체적인 실험 설정에 대한 결과를 추적함으로써, 과거에 달성한 성능을 잃지 않습니다.

**우리는 이제 자유롭게 실험**할 수 있습니다. 기존 모델의 다양한 매개변화 (또는 완전히 새로운 모델)를 실험해 보세요. 메인 스크립트에서 고려하는 모델 목록을 변경함으로써 이것을 언제든지 수행할 수 있으며, 언제든지 결과를 편리하게 검사하고 성능별로 정렬할 수 있습니다. 필요하면 모든 하이퍼파라미터를 쉽게 검사하고 개별 실행의 자세한 로그를 살펴볼 수 있습니다.

## 새롭게 얻은 실험의 자유를 활용하기

우리가 얻은 자유를 설명하기 위해, XGBoost 모델의 몇 가지 변형을 실험해 보겠습니다. 구체적으로는 다음을 수행합니다.
 - 오버피팅을 제어하는 매개변수를 조정하는 것이 모델에 미치는 영향을 평가합니다. 모델의 변형에서 매개변수 `min_child_weight`를 더 높은 값으로 설정하여 어떤 하위 샘플이든 어떤 자식 노드에 들어갈 수 있는지에 대한 하한을 설정합니다. 그리고, 그로 인해 트리의 어떤 리프에도 들어갈 수 있습니다.
 - 이전 단계에서 도입한 평균 아티스트 인기 특성의 중요성을 정량화하기 위한 실험을 수행합니다. 이 특성만 사용하는 모델의 품질은 어떤가요?

이를 위해 XGBoost 모델 공장 함수를 매개변화를 지원하도록 조정하여 특성 집합을 쉽게 변형하고 모델 매개변수를 조정할 수 있도록합니다.


```python
    @classmethod
    def create_xgb(cls, name_suffix="", features: Sequence[FeatureName] = DEFAULT_FEATURES, add_features: Sequence[FeatureName] = (),
            min_child_weight: Optional[float] = None, **kwargs):
        fc = FeatureCollector(*features, *add_features, registry=registry)
        return XGBGradientBoostedVectorClassificationModel(min_child_weight=min_child_weight, **kwargs) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder()) \
            .with_name(f"XGBoost{name_suffix}")
```

위에서 설명한대로, 다음과 같은 XGBoost 모델을 평가 모델 목록에 포함합니다.

```python
        ModelFactory.create_xgb(),
        ModelFactory.create_xgb("-meanArtistFreqPopular", add_features=[FeatureName.MEAN_ARTIST_FREQ_POPULAR]),
        ModelFactory.create_xgb("-meanArtistFreqPopularOnly", features=[FeatureName.MEAN_ARTIST_FREQ_POPULAR]),
```

기본 모델 "XGBoost"에 추가하여 기본 기능 및 매개변수를 사용하는 세 가지 변형을 추가로 정의합니다.
 - `XGBoost-meanArtistFreqPopular`는 기본 기능 위에 평균 아티스트 인기를 추가합니다.
 - `XGBoost-meanArtistFreqPopularOnly`는 특성 집합을 단일 정의된 특성으로 줄입니다.

우리는 매우 간결하게 모델 변형을 정의할 수 있었으며, 어떠한 코드 중복도 없습니다. 모델은 우리가 지정한 속성에 동적으로 적응합니다!

이제 의미있는 결과를 계산하고자 하며, 따라서 전체 데이터 집합으로 전환합니다. 랜덤 포레스트 모델은 학습 속도가 너무 느려 비활성화했습니다. XGBoost 모델의 변형에 대해 다음과 같이 얻을 수 있습니다:

```
INFO  2024-01-24 14:32:08,271 sensai.evaluation.eval_util:compare_models - Model comparison results:
                                   accuracy  balancedAccuracy  precision[popular]  recall[popular]  F1[popular]
model_name
XGBoost                            0.962409          0.645840            0.710075         0.297486     0.419305
XGBoost-meanArtistFreqPopular      0.964426          0.689429            0.698998         0.386820     0.498033
XGBoost-meanArtistFreqPopularOnly  0.957034          0.637985            0.556452         0.286902     0.378601
```

따라서 다음을 결론지을 수 있습니다.
  - 평균 아티스트 인기 특성은 매우 중요합니다. 이를 모델에 추가하면 F1 점수가 상당히 증가합니다.
  - 다른 모든 특성을 제거하면 결과가 악화되지만, 인기 특성만 사용하는 모델의 F1 점수는 여전히 매우 높습니다.

## 이 단계에서 해결된 원칙

- 실험 추적

[다음 단계](../step10-regression/README.md)