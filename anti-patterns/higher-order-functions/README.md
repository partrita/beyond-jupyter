# 전략 패턴 대신 고차 함수 사용하기

고차 함수, 즉 다른 함수를 매개변수로 사용하는 함수의 사용은 전략 패턴을 대체할 수 있습니다. 여기서 전략 패턴은 해당 함수를 지원하는 객체가 인수로 전달됩니다.

다음과 같은 회귀 모델 평가 사용 사례의 대체 구현을 고려해 보겠습니다. 여기서 사용할 메트릭을 매개변수화할 수 있어야 합니다.

- [함수형 구현](regressor_evaluation_functional.py): 우리는 메트릭을 올바르게 적용하기 위해 메트릭에 대한 추가 정보가 필요합니다. 특히
  - 보고를 위한 이름을 나타내는 문자열 및
  - 높은 값이 더 나은지를 나타내는 플래그,
  함수 자체만으로 충분하지 않으므로 세 개의 매개변수를 평가 함수의 인터페이스에 추가합니다:

  ```python
  metric_fn: Callable[[np.ndarray, np.ndarray], float],
  metric_name: str,
  higher_is_better: bool,
  ```

- [객체 지향 구현](regressor_evaluation_oop.py): 필요한 모든 측면을 처리하는 추상 클래스 `Metric`을 정의합니다.
    ```python
    class Metric(ABC):
      @abstractmethod
      def compute_value(self, y_ground_truth: np.ndarray, y_predicted: np.ndarray) -> float:
          pass

      @abstractmethod
      def get_name(self) -> str:
          pass

      @abstractmethod
      def is_larger_better(self) -> bool:
          pass
    ```

  - 평가 함수는 단일 인수를 취합니다:
    ```python
    metric: Metric
    ```

두 Python 파일을 자세히 살펴보고 결과에 대해 고민해 보세요.

## 주요 이슈들

객체 지향(OOP) 솔루션은 함수형 솔루션에 비해 상당한 이점을 가지고 있습니다:

- **캡슐화: 클래스 기반 접근 방식은 알고리즘 자체와 추가 정보/작업을 적절하게 그룹화할 수 있습니다.**
- 계산(메서드 `compute_value`)과 메타데이터(이름 및 "더 큰 값이 더 좋은 경우" 플래그)를 표현적 단위(추상화)에 그룹화함으로써 보다 번거로운 사용 패턴(1 대 3 개의 인수)을 줄이고, 더 나아가 오류를 방지하는 데 도움이 됩니다:
- 사용자는 적절한 매개변수 조합이 무엇인지 생각할 필요가 없으며, 의도된 조합은 클래스에 사전 구성되어 있습니다.
- 클래스는 단순히 함수보다 일반적인 개념이므로 복잡한 사용 사례에 더 적합합니다. 클래스는 알고리즘적 방법을 단일 함수로 합리적으로 축소할 수 없는 경우에 이에 대한 대안을 제공합니다.

- **검색 용이성: 추상 베이스 클래스는 (검색 가능한) 타입 바운드를 제공하며**, 이는 클래스 기반 인터페이스를 따르는 모든 구현을 발견하는 데 직접적으로 도움이 됩니다:

![계층 구조 보기](../../oop-essentials/05-ide-features/res/hierarchy_intellij.png)
 
- 반면에 함수형 인터페이스는 `Callable[[np.ndarray, np.ndarray], float]`를 지정하며 이러한 검색을 지원하지 않습니다. 이는 적용할 함수들이 적용 함수와 함께 동일한 위치에 있지 않을 때 심각한 사용성 제한이 발생할 수 있는 경우에 중대한 문제가 될 수 있습니다. 이러한 제한을 피하기 위해 매우 많은 문서화가 필요합니다. 
- 객체 지향(OOP) 솔루션은 이러한 문제 없이도 무방합니다. 타입 정보만으로 모든 구현을 발견할 수 있습니다.

- **매개변수화 가능성**: 객체는 속성을 통해 간편하게 동작을 매개변수화할 수 있습니다. 이는 불편한 *커링*(currying)을 피할 수 있으며, 예를 들어 메트릭의 임계값 매개변수를 지정하기 위해 `lambda` 함수, 로컬 함수(클로저) 또는 `functools.partial`을 사용합니다. 우리의 예에서는 lambda 함수를 사용하여 메트릭의 임계값 매개변수를 지정했습니다:

```python
lambda t, u: compute_metric_rel_freq_error_within(t, u, max_error)
```

반면에 객체지향적인 경우, 우리는 단순히 객체를 매개변수화합니다:

```python
MetricRelFreqErrorWithin(max_error)
```

- **로깅 및 지속성**: 객체는 더 쉽게 로그에 기록하고 저장할 수 있는 표현을 가지고 있습니다.
- `lambda` 함수 또는 익명 함수는 로깅에 적합한 표현을 갖추지 않았으며 직렬화할 수 있는 기능도 없습니다.
- `functools.partial`을 사용하면 (커리된 함수를 객체로 변환하여) 직렬화할 수 있지만 로깅에 대한 사용자 정의 가능한 표현이 없습니다.
- 클래스는 그와 대조적으로 완전한 제어를 제공합니다. 우리는 `__str__` 또는 `__repr__`을 원하는 대로 구현할 수 있으며, 선택적으로 `__getstate__` 및 `__setstate__`를 구현하여 지속성에 대한 세부적인 제어를 가질 수 있습니다.

- **명시적인 유형 관계**:
- 함수형 인터페이스 사양에서는 함수의 유형을 선언하기 위해 `Callable`을 사용하거나 키워드 인수를 사용하는 복잡한 함수 시그니처의 경우 이전에 정의된 `Protocol`을 사용합니다. 어느 경우에도 선언된 유형과 이를 구현하는 것 사이의 관계는 전형적인 *덕 타이핑*(duck typing) 방식으로 일반적으로 암시적입니다.
- 반면에 객체지향적인 해결책은 명시적인 유형 관계를 설정합니다. 명시적인 유형 관계는 확인 가능하므로, 정적 유형 검사기가 인터페이스가 실제로 올바르게 구현되었는지(유형적으로) 테스트할 수 있습니다. 반면에 덕 타이핑된 구현의 차이점은 무시됩니다: 함수 시그니처가 변경되면(어떤 이유로든) 정적 유형 검사는 함수가 그와 함께 사용될 의도로 고차 함수의 유형 주석과 더 이상 일치하지 않음을 나타내지 않을 것입니다.

## 캡슐화에 대한 더욱 강한 의지

이미 객체지향적인 해결책을 선호하는 중요한 이유들이 있습니다. 그러나 이제 우리가 [객체지향 핵심 모듈의 사례 연구](../../oop-essentials/02d-case-study-3-metric-abstraction/README.md)에서 다수의 메트릭을 지원하려는 경우를 고려해보겠습니다.
우리의 경우 함수만으로는 충분하지 않으므로 - 우리는 또한 메타 데이터(이름, 높은 값이 더 좋음)가 필요합니다 - 함수적인 경우 여러 메트릭을 제공하는 다음의 옵션이 있을 것입니다:

- 함수를 필요한 메타데이터와 함께 번들링한 튜플의 리스트: 튜플의 요소는 해당 인덱스로만 액세스할 수 있으며, 문서 문자열에서 이를 자세히 설명해야 합니다.
   ```python
   metric_tuples: List[Tuple[Callable[[np.ndarray, np.ndarray], float], str, bool]],
   ```

- "metric_fn", "metric_name", "higher_is_better"와 같은 키를 가진 사전의 리스트
   ```python
   metrics: List[Dict[str, Any]],
   ```
   
- 여기서 값에 대한 유형 정보가 완전히 손실되며, 정적 유형 검사가 진행될 수 없습니다.
- 다시 말하지만, 사전은 앞에서 언급한 단점을 가진 기본 객체이므로 이는 매우 의문스러운 선택일 것입니다.

- 세 개의 별도 리스트
   
   ```python
   metric_fns: List[Callable[[np.ndarray, np.ndarray], float],
   metric_names: List[str],
   higher_is_better_flags: List[bool],
   ```

   - 이를 `zip`을 사용하여 반복할 수 있지만, 이후 길이가 같은지 일관성을 확인해야 합니다.

이러한 옵션을 간단한 객체지향적인 해결책과 비교해보면, 단순한 다음 코드를 생각해보세요.

   ```python
   metrics: List[Metric],
   ```
   - 이 코드는 완전하게 문서화된 정적 유형 검사를 지원합니다.

## 결론

종합하자면, 기능적 대안을 선호할 많은 이유를 생각해낼 수 없습니다. 물론 함수가 간단하고 매개변수화될 필요가 없으며 의미 있는 로깅이 필요하지 않는 경우 등 - 이러한 경우에는 기능적 인터페이스를 사용하는 것이 가장 간결한 해결책이 될 수 있습니다. 그러나 일반적인 규칙으로는 객체지향적인 전략 패턴이 알고리즘을 주입하는 가장 우아한 해결책입니다.