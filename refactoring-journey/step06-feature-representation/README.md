## 단계 6: 특성 표현

이번 단계에서는 **특성에 대한 명시적인 표현을 생성**하여 특성에 대한 메타 정보를 표현하고 그에 따라 모델별 변환을 수행할 수 있도록 합니다. 아이디어는 특성이나 특성 집합에 관련된 관련 정보를 중앙에서 *한 번*만 등록한 다음 모델 구현에서 사용할 특성과 어떤 구체적인 변환을 적용할지를 결정하게 합니다.

이것은 **선언적 의미론**을 중요하게 합니다. 우리가 사용하려는 특성 집합을 간단히 선언하면 모든 모델별 측면이 자동으로 따라옵니다. 모델의 입력 파이프라인은 이렇게 조합 가능해집니다.


## 특성 레지스트리

우리는 [`features`](songpop/features.py)라는 새로운 모듈을 소개합니다. 여기에는 모든 특성/특성 집합을 단위로 참조하고자 하는 열거형 `FeatureName`을 사용합니다. 그런 다음 열거형 항목을 `FeatureGeneratorRegistry`의 키로 사용합니다. 각 특성 단위에 대해 중요한 속성을 정의하는 특성 생성기를 등록합니다.
  - 주요 측면은 물론 원본 입력 데이터에서 특성 값을 생성하는 방법입니다. 우리의 경우에는 특성을 단순히 입력에서 가져오고, 이전 단계와 같이 `FeatureGeneratorTakeColumns`를 사용합니다.
  - 중요한 것은 또한 메타 데이터를 지정하는 것입니다.
    - 특성 집합 중 어떤 부분이 범주형인지(있는 경우),
    - 숫자 특성을 어떻게 정규화할 수 있는지,
    - 메타 데이터가 특성 생성 자체에 영향을 미치지 않는다는 점에 유의하십시오.
  - 즉, 우리가 `categorical_feature_names` 또는 `normalisation_rule_template`에 전달하는 값은 특성 생성에 영향을 주지 않으며, 이는 나중에 특성 변환기에서 활용될 수 있는 정보를 제공하기 위한 것입니다 (다음 부분을 참조하세요).

```python
registry = FeatureGeneratorRegistry()

registry.register_factory(FeatureName.MUSICAL_DEGREES, lambda: FeatureGeneratorTakeColumns(COLS_MUSICAL_DEGREES,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(skip=True)))

registry.register_factory(FeatureName.MUSICAL_CATEGORIES, lambda: FeatureGeneratorTakeColumns(COLS_MUSICAL_CATEGORIES,
    categorical_feature_names=COLS_MUSICAL_CATEGORIES))

registry.register_factory(FeatureName.LOUDNESS, lambda: FeatureGeneratorTakeColumns(COL_LOUDNESS,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

registry.register_factory(FeatureName.TEMPO, lambda: FeatureGeneratorTakeColumns(COL_TEMPO,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

registry.register_factory(FeatureName.DURATION, lambda: FeatureGeneratorTakeColumns(COL_DURATION_MS,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))
```

해당되는 경우, 범주형 특성을 키워드 매개변수 `categorical_feature_names`를 통해 선언했습니다. 숫자 특성의 경우, 특성이 어떻게 정규화될지를 키워드 매개변수 `normalisation_rule_template`를 통해 지정했습니다 (한 번 더 강조하자면, 특성 생성기 자체가 정규화를 수행하는 것이 아니라 정보를 저장하는 역할만 한다는 것입니다):
  - 음악 학위는 이미 [0, 1] 범위로 정규화되어 있기 때문에 정규화 과정에서 무시할 수 있도록 지정했습니다 (`skip=True`).
  - 다른 특성의 경우 `StandardScaler`를 적용하는 것이 합리적이므로 해당 변환기 생성을 위한 팩토리를 지정했습니다.

특성 생성기에 등록된 특성 중 일부는 다르게 처리됩니다:
  - 원래 구현은 `mode`와 `key` 특성을 숫자 특성으로 취급했지만, 이제는 이를 범주형으로 취급합니다. 특히 노래의 음악 키에 대해서는 이것이 훨씬 합리적입니다.
  - 원래 구현은 `genre` 특성을 완전히 삭제했는데, 이는 숫자 표현이 없었기 때문입니다. 우리는 이를 또 다른 범주형 특성으로 포함시켰습니다.

레지스트리에서 키로 문자열 대신 `Enum` 항목을 사용하는 이유는 IDE에서 자동 완성 및 안전한 리팩터링을 가능하게 하기 위해서입니다.

## 적용된 모델 팩토리

새로 도입된 모듈 [model_factory](songpop/model_factory.py)의 모델 구현은 등록된 특성을 사용하기 위해 특성을 그들이 등록된 이름으로 참조하는 `FeatureCollector`를 사용합니다.
모델에 특성 수집기를 추가하면 수집된 모든 특성이 모델에 입력으로 제공됩니다.

또한 특성 수집기를 기반으로 특성 변환을 정의합니다. 특성 생성기가 필요한 메타 데이터를 나타내기 때문에, 특히 `FeatureCollector` 인스턴스에서 팩토리를 호출하여 사용되는 모든 범주형 특성에 대한 원핫 인코더를 만들 수 있습니다. 현재 모든 모델이 이를 필요로 하기 때문에, 모든 모델에 원핫 인코더를 추가했습니다.

```python
@classmethod
def create_logistic_regression(cls):
    fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
    return SkLearnLogisticRegressionVectorClassificationModel(solver='lbfgs', max_iter=1000) \
        .with_feature_collector(fc) \
        .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder(),
            fc.create_feature_transformer_normalisation()) \
        .with_name("LogisticRegression")
```

로지스틱 회귀 모델은 스케일링/정규화된 데이터와 가장 잘 작동하기 때문에, 특성 등록 중에 지정된 정규화를 수행하는 특성 변환기를 추가로 추가합니다. 이는 `FeatureCollector` 인스턴스의 두 번째 팩토리 메서드를 통해 수행됩니다.

KNN 모델의 경우, 특성이 의미 있는 거리 메트릭을 생성하기 위해 특성이 위치한 벡터 공간이 필요합니다. 이제 0과 1로 표현되는 불리언 특성을 추가했기 때문에, 표준화를 일부 사용하는 정규화 후에, 더 큰 척도를 사용하는 특성에 과도한 가중치를 주지 않기 위해 추가로 `MaxAbsScaler` 변환기를 추가합니다.

결과 변환은 개선될 것으로 예상되지만, 궁극적으로 철저히 설계된 거리 메트릭은 특성 공간의 하위 공간을 명시적으로 고려하고 이러한 개념을 지원하는 더 유연한 KNN 구현을 사용하여 서브 메트릭에서 메트릭을 구성해야 합니다.

```python
@classmethod
def create_knn(cls):
    fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
    return SkLearnKNeighborsVectorClassificationModel(n_neighbors=1) \
        .with_feature_collector(fc) \
        .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder(),
            fc.create_feature_transformer_normalisation(),
            DFTSkLearnTransformer(MaxAbsScaler())) \
        .with_name("KNeighbors")
```

알아두어야 할 사항:
  - 기존 모델과 새 모델을 함께 쉽게 지원할 수 있습니다 (`_orig` 팩토리 메서드는 변경되지 않음). 왜냐하면 차이가 있는 파이프라인 구성 요소를 실제 모델 사양으로 이동했기 때문입니다. 이는 우리가 예전 모델을 재평가할 수 있도록 하는 중요한 역할을 합니다. 예를 들어, 다른 데이터 집합에서도 가능합니다.
  - 모델 사양은 대부분 선언적입니다. 특히 모델이 사용할 특성 집합을 간단히 선언할 수 있으며, 모델에 따라 선택된 특성 집합이 모델에 적합한 방식으로 자동으로 변환됩니다. 즉, 필요한 경우 범주형 특성은 원핫 인코딩되고 숫자 특성은 적절하게 정규화됩니다.
  - 이번 단계에서 도입된 특성 표현 없이는 이러한 작업을 이렇게 간결하게 할 수 없었을 것입니다.


### 선언적이고 완전히 조합 가능한 특성 파이프라인

마지막으로, 이러한 점을 더 자세히 설명하기 위해 확장된 팩토리 정의를 사용하여 보여드립니다. 다음은 로지스틱 회귀 모델 팩토리의 확장된 정의입니다. 여기서 모델의 이름을 수정하고 특성 집합을 자유롭게 선택할 수 있는 두 개의 매개변수를 추가했습니다.

```python
    @classmethod
    def create_logistic_regression(cls, name_suffix="", features: Optional[List[FeatureName]] = None):
        if features is None:
            features = DEFAULT_FEATURES
        fc = FeatureCollector(features, registry=registry)
        return SkLearnLogisticRegressionVectorClassificationModel(solver='lbfgs', max_iter=1000) \
            .with_feature_collector(fc) \
            .with_feature_transformers(
                fc.create_feature_transformer_one_hot_encoder(),
                fc.create_feature_transformer_normalisation()) \
            .with_name("LogisticRegression" + name_suffix)
```

아래 코드와 같이 하면 로지스틱 회귀 모델의 변형을 실험할 수 있습니다.

```python
models = [
    ModelFactory.create_logistic_regression(),
    ModelFactory.create_logistic_regression("-only-cat", [FeatureName.MUSICAL_CATEGORIES]),
    ModelFactory.create_logistic_regression("-only-cat-deg", 
        [FeatureName.MUSICAL_CATEGORIES, FeatureName.MUSICAL_DEGREES]),
]
```

예를 들면 특성 집합을 선언함으로써 완전히 다른 입력 파이프라인을 사용하는 모델의 변형을 실험할 수 있습니다. 모델은 단순히 범주형 특성을 원핫 인코딩하고 숫자 특성을 정규화해야 한다고 선언했을 뿐이며, 우리가 실제로 사용하는 특성 집합이 무엇이든 하위 변환은 발생합니다. 우리가 해야 할 일은 특성 집합을 지정하는 것 뿐입니다.

특성 메타데이터의 표현은 이를 달성하는 데 중요했습니다! 만약 더 저수준의 데이터 처리 접근 방식을 사용했다면, 우리는 특성 변경을 처리하기 위해 하위 변환 코드를 명시적으로 적응시켜야 했을 것입니다.

## 이번 단계에서 다루는 원칙

- 특성을 파악하라
- 적절한 추상화를 찾는다.
- 선언적 의미론을 선호한다.


[다음 단계](../step07-feature-engineering/README.md)
