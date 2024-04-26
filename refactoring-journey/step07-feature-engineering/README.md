# 단계 7: 특성 엔지니어링

노래 인기의 현실을 살펴보면, 아티스트의 신원이 다른 모든 것보다 중요합니다:

인기 있는 아티스트의 노래는 그렇지 않은/알려지지 않은 아티스트의 노래보다 훨씬 더 인기가 있을 가능성이 높습니다. 그 노래의 특성이 그 외에 정확히 같다 하더라도 말이죠. 이론적으로 아티스트의 신원을 범주형 특성으로 사용할 수 있지만, 아주 많은 수의 아티스트로 인해 이는 실용적이지 않습니다.

따라서 이번 단계에서는 더 실용적인 방식으로 이 개념을 구현하는 사용자 지정 특성을 엔지니어링합니다: *동일한* 아티스트의 *다른* 노래가 인기 있는 경우의 상대적 빈도를 특성으로 추가합니다. `FeatureGeneratorMeanArtistPopularity`가 이러한 특성을 구현하고, 특성 생성기 레지스트리에 `FeatureName.MEAN_ARTIST_FREQ_POPULAR`로 등록됩니다 (갱신된 모듈 [features](songpop/features.py) 참조). 특정 의미론적 요구 사항으로 인해 학습 케이스와 추론 케이스 사이에 구별이 필요합니다:
  - 추론 케이스의 경우, 우리는 훈련 세트 전체에서 관측한 아티스트의 상대적 빈도를 단순히 사용할 수 있습니다.
  - 학습 케이스의 경우, 현재 데이터 포인트를 제외해야 합니다 (포함하면 명백한 데이터 누수가 발생합니다).

어느 경우에도 특성이 정의되지 않을 수 있습니다:
  - 추론 중에는 해당 아티스트가 훈련 세트에 나타나지 않았을 수 있습니다.
  - 학습 중에는 아티스트가 훈련 세트에 단 하나의 노래만 가지고 있는 경우가 있을 수 있습니다.

우리는 XGBoost의 그래디언트 부스팅 결정 트리를 사용하여 불완전한 데이터를 명시적으로 지원하는 모델의 일종으로 사용합니다.

```python
    @classmethod
    def create_xgb(cls, name_suffix="", features: Sequence[FeatureName] = DEFAULT_FEATURES, add_features: Sequence[FeatureName] = (),
            min_child_weight: Optional[float] = None, **kwargs):
        fc = FeatureCollector(*features, *add_features, registry=registry)
        return XGBGradientBoostedVectorClassificationModel(min_child_weight=min_child_weight, **kwargs)
            .with_feature_collector(fc)
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder())
            .with_name(f"XGBoost{name_suffix}")
```

우리가 새롭게 도입한 특성을 `add_features` 매개변수를 통해 추가할 수 있으며, 세 가지 구체적인 모델을 고려해 볼 수 있습니다: 
1. 이전 특성 집합을 사용하는 모델
2. 새로운 특성을 추가한 모델
3. 그리고 새로운 특성만 사용하는 모델.

```python
    models = [
        ...
        ModelFactory.create_xgb(),
        ModelFactory.create_xgb("-meanArtistFreqPopular", add_features=[FeatureName.MEAN_ARTIST_FREQ_POPULAR]),
        ModelFactory.create_xgb("-meanArtistFreqPopularOnly", features=[FeatureName.MEAN_ARTIST_FREQ_POPULAR]),
    ]
```

다른 모델들은 새로운 특성을 지원하려면 대체로 적합한 특성 변환을 적용할 수 있습니다. 

## 이번 단계에서 다루는 원칙

- 매개변수의 노출

[다음 단계](../step08-high-level-evaluation/README.md)
