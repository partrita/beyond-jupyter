# Step 3: 리팩터링

이제 코드 구조를 개선할 때입니다. `songpop` 패키지를 만들어서 [`data`](songpop/data.py) 모듈로 포함시킵니다. 이 모듈은 데이터셋과 관련된 모든 로직을 포함하고 있습니다. 다음 단계에서는 전용 목적을 가진 추가 모듈을 만들 것입니다.

업데이트된 [메인 스크립트](run_classifier_evaluation.py)는 주로 즉각적인 실험에만 중점을 둡니다.

## 이번에 다룬 원칙

- 재사용 가능한 구성 요소 만들기

[다음 단계](../step04-model-specific-pipelines/README.md)