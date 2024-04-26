# 단계 5: sensAI 소개

이전 단계에서는 Scikit-learn 파이프라인 객체만을 의존하여 모델별 데이터 파이프라인을 생성하는 것이 조립 가능성 측면에서 부족함을 관찰했습니다.

우리는 Scikit-learn을 주요 프레임워크로 사용하는 것에서 [sensAI](https://github.com/aai-institute/sensAI)로 전환할 것입니다. 이는 합리적인 인공지능을 위한 파이썬 라이브러리입니다. sensAI는 다양한 머신러닝 라이브러리 (scikit-learn과 PyTorch가 가장 관련이 있는 것으로 생각됩니다)를 지원하는 편리한 추상화를 제공하는 고수준 프레임워크입니다. 다양한 고수준 함수 집합을 통해 중요한 부분을 제어하는 것을 포기하지 않으면서 보일러플레이트 코드를 최소화하는 데 도움이 됩니다.

이번 단계는 sensAI의 주요 추상화 중 일부를 소개하는 것을 목적으로 합니다. 이는 특성 생성 및 표현을 중심으로합니다. [이전 단계](../step04-model-specific-pipelines/README.md)의 Scikit-learn 파이프라인 객체와 비교하여 이러한 추상화는 'fit/transform' 객체를 연결하는 것보다 더 많은 의미론/메타 정보를 추가합니다.

[다음 단계](../step06-feature-representation/README.md)에서는 이러한 개념을 활용하여 모델 정의에서 완전한 선언적 코드 스타일을 얻을 것입니다.

## sensAI를 사용한 모델별 파이프라인

모델 명세에 관한 것으로써, sensAI는 모델별 입력 파이프라인 개념을 매우 명확하게 만듭니다. 이는 두 가지 중요한 추상화를 소개하여 다음과 같은 두 가지 중요한 모델링 측면을 다룹니다.

- **모델이 사용하는 데이터는 무엇인가요?**
  - 관련 추상화는 `FeatureGenerator`입니다. `FeatureGenerator` 인스턴스를 통해 모델은 사용할 특성 집합을 정의할 수 있습니다. 또한 이러한 인스턴스는 해당 특성에 대한 메타 데이터를 보유할 수 있으며 (우리는 다음 단계에서 이를 활용할 것입니다), sensAI에서 모든 특성 생성기 구현의 클래스 이름은 `FeatureGenerator` 접두사를 사용하여 IDE의 자동 완성 기능을 통해 편리하게 찾을 수 있습니다.

- **해당 데이터를 어떻게 표현해야 하나요?**
  - 다양한 모델은 동일한 데이터에 대해 다른 표현을 요구할 수 있습니다. 예를 들어 일부 모델은 모든 특성이 숫자여야 하므로 범주형 특성을 인코딩해야 할 수 있지만, 다른 모델은 원래의 표현을 사용하는 것이 더 나을 수 있습니다.
  - 더 나아가, 일부 모델은 숫자 특성이 특정한 방식으로 정규화되거나 스케일링되는 것이 더 좋을 수 있지만 다른 모델에는 차이가 없을 수 있습니다. 이러한 요구 사항은 모델별 변환을 추가함으로써 해결할 수 있습니다.
  - 관련 추상화는 `DataFrameTransformer`이며, sensAI에서 추상이 아닌 모든 구현은 `DFT` 클래스 이름 접두사를 사용합니다.

지금까지 우리가 정의한 모델은 모두 같은 기본 특성 생성기(`FeatureGeneratorTakeColumns`)를 사용하며, 달성하고자 하는 원래 표현을 얻기 위해 같은 변환기를 사용했습니다(`StandardScaler`와 함께 `DFTSkLearnTransformer`). 하지만 곧 상황을 바꿀 것입니다.

```python
class ModelFactory:
  COLS_USED_BY_ORIGINAL_MODELS = [COL_YEAR, *COLS_MUSICAL_DEGREES, COL_KEY, COL_MODE, COL_TEMPO, COL_TIME_SIGNATURE, COL_LOUDNESS,COL_DURATION_MS]

  @classmethod
  def create_logistic_regression_orig(cls):
      return SkLearnLogisticRegressionVectorClassificationModel(solver='lbfgs', max_iter=1000) \
          .with_feature_generator(FeatureGeneratorTakeColumns(cls.ORIGINAL_MODELS)) \
          .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
          .with_name("LogisticRegression-orig")
```

모델 정의는 우리가 원하는 것을 간결하게 정의한 것에 주목하세요. 이는 코드에서 선언적 의미론에 더 가까워져 있습니다. 게다가, 이제 모든 모델에 이름을 지정하여 모델별 결과를 보고할 수 있습니다.

## 무작위성

sensAI의 scikit-learn 클래스 래퍼는 재현 가능한 결과를 보장하기 위해 기본적으로 고정된 랜덤 시드를 사용할 것입니다. (원래 노트북 구현에서는 랜덤 포레스트 모델이 고정된 랜덤 시드를 사용하지 않았음에 유의하세요.)


## 이번 단계에서 다루는 원칙

- 선언적 의미론을 선호합니다.
- 적절한 추상화를 찾습니다.
- 제어되지 않은 무작위성을 피합니다.

[다음 단계](../step06-feature-representation/README.md)