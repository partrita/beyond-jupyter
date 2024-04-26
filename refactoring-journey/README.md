[![CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

# Spotify 노래 인기 예측: 리팩토링 하기

이 케이스 스터디에서는 Jupyter 노트북으로 구현된 머신러닝 사용 사례를 연이어 리팩터링하여 소프트웨어 디자인을 개선하는 방법을 보여드리겠습니다. 이 리팩터링을 통해 다음과 같은 이점을 얻을 수 있습니다.

- 실험의 유연성 향상
- 결과를 적절하게 추적
- 복잡한 솔루션을 직관적으로 배포

이 사용 사례는 Kaggle에서 가져온 대략 백만 곡의 메타데이터가 포함된 데이터 세트를 다룹니다(다운로드 지침은 아래에 있습니다). 이 데이터를 사용하여 다른 곡 특성(예: 템포, 출시 연도, 키, 음악 모드 등)이 주어졌을 때 노래 인기를 예측하는 모델을 학습하는 것이 목표입니다.

## 사전 준비 사항

Python 가상 환경을 생성하고 IDE에서 프로젝트를 설정하고 데이터를 다운로드했는지 확인하십시오([README 파일](../README.md#preliminaries)에 설명되어 있음).

## 이 패키지 사용 방법

이 패키지는 다음과 같이 구성됩니다.
 - 리팩터링 프로세스의 각 단계별로 하나의 폴더가 있으며 해당 단계에 대한 주요 측면을 설명하는 전용 README 파일이 있습니다.
 - 각 폴더에는 해당 README 파일과 함께 사용 사례의 독립적인 Python 구현이 있습니다.

이 패키지를 탐색하는 권장 방법은 리파지토리를 복제하고 선호하는 IDE에서 열어 익숙한 도구로 탐색하고 코드를 효율적으로 탐색하는 것입니다.

### 차이점 분석

단계 간의 구체적인 변경 사항을 더 명확하게 확인하려면 차이 도구를 사용할 수 있습니다. 이를 지원하려면 `generate_repository.py` Python 스크립트를 실행하여 각 단계를 별도의 태그로 참조하는 git 리포지토리를 `refactoring-repo` 폴더에 생성할 수 있습니다. 이 폴더에서 예를 들어 `git difftool step04-model-specific-pipelines step05-sensai`를 실행할 수 있습니다.

## 리팩토링 단계

1. [단일 노트북](step00-monolithic-notebook/README.md). 이것은 거의 구조화되지 않은 Jupyter 노트북에서 시작하는 지점입니다.  
2. [Python 스크립트](step01-python-script/README.md). 이 단계는 모델의 교육 및 평가와 엄격하게 관련된 코드를 추출합니다.
3. [데이터 세트 표현](step02-dataset-representation/README.md). 이 단계에서는 데이터 세트에 대한 명시적인 표현을 도입하여 변환을 명시적으로하고 선택적으로 수행합니다.
4. [리팩터링](step03-refactoring/README.md). 이 단계에서는 기능별 Python 모듈을 추가하여 코드 구조를 개선합니다.
5. [모델별 파이프라인](step04-model-specific-pipelines/README.md). 이 단계에서는 모든 변환 작업을 모델로 이동하여 서로 다른 모델이 완전히 다른 파이프라인을 사용할 수 있게합니다.
6. [sensAI](step05-sensai/README.md). 이 단계에서는 고수준 라이브러리 sensAI를 소개하며 이를 통해 나중에 더 유연하고 선언적인 모델 사양을 사용할 수 있습니다.
7. [특성 표현](step06-feature-representation/README.md). 이 단계에서는 모델이 사용하는 특성 및 특성 속성의 표현을 분리하여 모델 입력 파이프라인을 유연하게 구성할 수 있습니다.
8. [특성 엔지니어링](step07-feature-engineering/README.md). 이 단계에서는 조합된 특성에 엔지니어링된 특성을 추가합니다.
9. [고수준 평가](step08-high-level-evaluation/README.md). 이 단계에서는 sensAI의 고수준 추상화를 적용하여 모델 평가를 수행하고 로깅을 가능하게 합니다.
10. [실험 추적](step09-tracking-experiments/README.md). 이 단계에서는 sensAI의 mlflow 통합을 통해 추적 기능을 추가합니다(그리고 결과를 직접 파일 시스템에 저장하기도 합니다).
11. [회귀](step10-regression/README.md). 이 단계에서는 예측 문제를 회귀 문제로 더 자연스럽게 고려합니다.
12. [하이퍼파라미터 최적화](step11-hyperopt/README.md). 이 단계에서는 XGBoost 회귀 모델의 하이퍼파라미터 최적화를 추가합니다.
13. [교차 검증](step12-cross-validation/README.md). 이 단계에서는 교차 검증을 사용할 수 있는 옵션을 추가합니다.
14. [배포](step13-deployment/README.md). 이 단계에서는 도커 컨테이너에 패키지화된 추론용 웹 서비스를 추가합니다.