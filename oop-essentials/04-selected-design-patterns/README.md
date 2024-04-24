# ML 프로젝트에 유용한 디자인 패턴

객체 지향 *디자인 패턴*은 소프트웨어 디자인에서 주로 발생하는 문제에 대한 재사용 가능한 해결책입니다. 디자인 패턴은 반복적으로 발생하는 디자인 문제를 해결하는 검증된 방법과 성공적인 접근 방식을 포착하여 디자인을 더 유연하고 우아하며 유지 관리 가능하게 만듭니다.

재미있게도 대부분의 프로그래머들은 이러한 패턴들을 공식적인 이름이나 설명을 알지 못해도 자연스럽게 도착하고 사용합니다. 이 패턴들은 공통적인 도전에 대응하기 위해 코드를 구조화하는 직관적인 방법들을 나타내기 때문입니다. 이러한 패턴들을 명명함으로써 소프트웨어 디자인에 대한 공통 언어(용어)를 정립합니다.

## 전략 패턴

알고리즘은 일반적으로 하위 알고리즘(또는 서브 루틴)의 집합으로 구성됩니다. 이러한 하위 알고리즘들이 수행하는 작업에는 여러 가지 방법이 있을 수 있습니다. *전략 패턴*은 상위 알고리즘이 하위 수준의 작업의 추상화를 나타내는 클래스의 객체를 받아들이는 방식으로 해당 하위 수준의 동작을 수정할 수 있도록 합니다.

Case Study에서의 예를 상기해보십시오: `ModelEvaluation`은 하나 이상의 `Metric` 인스턴스로 매개변수화될 수 있습니다.
이들 `Metric` 인스턴스는 평가지표를 계산하는 실제 작업을 수행합니다."

```python
class ModelEvaluation:
    def __init__(self, X: pd.DataFrame, y: pd.Series,
            metrics: List[Metric],
            test_size: float = 0.3, shuffle: bool = True, random_state: int = 42):
```

의존성 역전 원칙에 따라 `ModelEvaluation`은 추상화된 `Metric`에 의존하며, 특정 평가지표 값을 계산하는 로직을 캡슐화합니다.

```python
class Metric(ABC):
    @abstractmethod
    def compute_value(self, y_ground_truth: np.ndarray, y_predicted: np.ndarray) -> float:
        pass
```

그런 다음 다양한 방식으로 특화될 수 있으며, 그 후에는 해당 동작을 수정하기 위해 평가 객체에 삽입될 수 있습니다.

```python
class MetricMeanAbsError(Metric):
    def compute_value(self, y_ground_truth: np.ndarray, y_predicted: np.ndarray) -> float:
        return metrics.mean_absolute_error(y_ground_truth, y_predicted)

class MetricR2(Metric):
    def compute_value(self, y_ground_truth: np.ndarray, y_predicted: np.ndarray) -> float:
        return metrics.r2_score(y_ground_truth, y_predicted)
```

새로운 메트릭은 `Metric` 추상화의 새로운 구현을 통해 지원될 수 있으며, `ModelEvaluation`에 구현된 주요 알고리즘을 수정할 필요가 없습니다(개방-폐쇄(open-closed) 원칙).

## 팩토리 패턴

알고리즘은 종종 동적으로 객체를 생성해야 하며, 객체가 생성되는 방식을 사용자가 정의할 수 있도록 원할 때가 많습니다. *팩토리 패턴*은 이러한 필요성을 해결합니다.

예를 들어, 다른 국가/지역에 대해 별도의 모델을 생성하는 알고리즘에서 모델을 생성해야 할 수 있습니다. 알고리즘은 내부적으로 데이터를 분할하고 어느 시점에서는 특정 국가의 데이터만 사용하여 새로운 모델을 생성해야 합니다. 따라서 알고리즘에 모델 생성 메커니즘을 주입할 수 있는 방법이 필요합니다. 이는 팩토리 추상화를 도입함으로써 형식화될 수 있습니다.

```python
class ModelFactory(ABC):
    def create_model(self) -> Model:
        pass
```

그런 다음 우리가 사용하려는 구체적인 모델을 처리할 수 있도록 특수화될 수 있습니다.

```python
class MyModelFactory(ModelFactory):
    def __init__(self, config: MyModelConfig):
        self._config = config
    
    def create_model(self):
        ...
```

그런 다음 학습 프로세스에 전달될 수 있습니다.

```python
class RegionalLearner:
    def __init__(self, database: Database):
        self.database = database
        ...
    
    def train_models(self, model_factory: ModelFactory) -> List[Model]:
        ...
        for regional_data in self.database.split_regions():
            ...
            model = model_factory.create_model()
            model.fit(regional_data)
            ...
```

팩토리는 일반적으로 실제 객체 생성을 처리하는 하나 이상의 `create` 메서드를 갖습니다. 각 구체적인 객체가 어떻게 (즉, 어떤 매개변수로) 생성될지를 결정하는 구성은 일반적으로 팩토리의 속성에 저장됩니다 (우리의 예제에서는 `_config`).

호출자로부터 필수 매개변수를 받아야 하는 경우도 있습니다. 예를 들어, 팩토리를 구성할 때 이러한 매개변수를 지정할 수 없는 경우가 있습니다. 왜냐하면 이 매개변수는 아직 존재하지 않고 팩토리 생성 중에 생성되는 경우가 있기 때문입니다.

예를 들어, 강화 학습 에이전트를 훈련하는 사용 사례를 고려해 보겠습니다. 에이전트의 구성은 에이전트가 작동할 환경에 따라 달라집니다. 환경은 어딘가에서 강화 학습 과정으로 생성되고, 환경이 주어지면 특수화된 에이전트를 생성하고 훈련해야 합니다.
이런 경우 강화 학습 과정은 다음과 같은 서명의 팩토리를 받을 수 있습니다.

```python
class RLAgentFactory(ABC):
    def create_agent(self, env: Env) -> RLAgent:
        pass
```

그런 다음 훈련 메커니즘에 주입할 수 있습니다.

```python
class RLProcess:
    def __init__(self, env_factory: EnvFactory, agent_factory: Agent_factory, ...):
        ...

    def run(self) -> TrainingResult:
        ...
```

생성 메커니즘의 주입 외에도, 팩토리는 더 쉽게 저장되거나 전송할 수 있는 구성 객체로 사용될 수 있습니다. 예를 들어, 실제 객체를 기술적인 이유로 유지하기 어려운 경우, 불안정한 표현이거나 과도하게 큰 경우, 실제 객체 대신 팩토리를 사용하는 것이 유용할 수 있습니다.

또한 팩토리의 대체 개념도 있습니다. **함수를 팩토리로 사용**할 수도 있습니다. 이러한 함수는 주로 `OOP`에서 정적 메서드나 클래스 메서드로 구현됩니다. 특히 이러한 팩토리는 객체에 대한 대체 생성 메커니즘을 제공하고 `Class.from_something`과 같은 네이밍 스킴을 사용합니다. 예를 들어, 저장된 pickle 파일에서 `RLAgent`를 로드할 수 있도록 하고 클래스 메서드 `from_pickle_file`을 추가하여 클래스의 인스턴스를 얻을 수 있습니다:

```python
class RLAgent(ABC):
    @classmethod
    def from_pickle_file(cls, path: str) -> Self:
        pass
```

## 레지스트리 패턴

*레지스트리*는 이름으로 객체를 편리하게 검색할 수 있는 객체 컨테이너입니다. 이것을 팩토리 패턴과 결합하여 팩토리의 모음을 등록할 수 있습니다.
객체의 집합이 고정된 경우, 레지스트리로 열거형을 사용할 수 있습니다.

예를 들어, 모델에서 사용할 기능을 등록하여 모델이 이름으로 기능의 하위 집합을 사용할 수 있도록 할 수 있습니다.

```python
feature_registry = FeatureRegistry()
feature_registry.register_feature("age", AgeFeature(categorical=False))
feature_registry.register_feature("age_category", AgeFeature(categorical=True))
feature_registry.register_feature("height", HeightFeature(unit="inch"))
```

모델은 그냥 이름을 사용하여 등록된 기능을 가져올 수 있습니다.

```python
model = MyModelFactory(features=["age", "height"], feature_registry=feature_registry) \
    .create_model() 
```

그리고 모델 팩토리는 레지스트리를 사용하여 기능 객체를 내부적으로 해결합니다. 이렇게 하면 사용자가 기능 객체를 인스턴스화하는 방법을 알 필요 없이 모델의 기능 집합을 지정하는 프로세스가 간소화되며 사용자는 단순히 문자열을 지정할 수 있습니다.
이렇게 함으로써 모델의 *구성*이 용이해집니다. 또한, 서로 다른 모델 간에 기능 정의를 일관되게 유지하는 데 도움이 됩니다.
등록된 키의 집합이 고정된 경우, 문자열 대신 `Enum`을 사용하는 것이 유용할 수 있습니다. 이렇게 하면 IDE에서 자동 완성을 사용할 수 있고 잘못된 지정을 피할 수 있습니다.

[다음: IDE 기능 활용하기](../05-ide-features/README.md)