

<p align="left" style="text-align:left">
  <img src="resources/beyond-jupyter-logo.png#gh-light-mode-only" style="width:600px">
  <img src="resources/beyond-jupyter-logo-dark-mode.png#gh-dark-mode-only" style="width:600px">
  <br><br>
  <div align="left" style="text-align:left">
  <a href="https://creativecommons.org/licenses/by-sa/4.0/" style="text-decoration:none"><img src="https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg" alt="License"></a>
  </div>
</p>

*Beyond Jupyter* 프로젝트는 소프트웨어 디자인에 관한 리소스들의 모음입니다. 이 프로젝트는 머신 러닝 애플리케이션에 특히 중점을 두고 있습니다. 머신 러닝 컨텍스트에서 개발되는 소프트웨어는 종종 상당히 낮은 추상화 수준에 머무르며 소프트웨어 디자인과 소프트웨어 엔지니어링에서 잘 알려진 표준을 충족시키지 못할 수 있습니다. 심지어 Jupyter와 같은 개발 환경은 비구조적인 디자인을 적극적으로 장려한다고도 볼 수 있습니다. 따라서 우리는 해당 소프트웨어 개발 패턴을 포기하고 비유적으로 "Jupyter를 뛰어넘는" 것이 필요하다고 생각합니다.

이 자료들의 목표는 실무자들이 원칙에 기반한 소프트웨어 디자인 접근 방식이 머신 러닝 프로젝트의 모든 측면을 지원하며 개발과 실험을 모두 가속화하는 방법을 이해하는 데 있습니다.

좋은 디자인이 개발 속도를 늦춘다는 것은 흔한 오해입니다. 실제로는 그 반대가 사실입니다. 우리는 (비구조적인) 절차적 코드의 한계를 보여주고, 원칙에 기반한 디자인 접근 방식이 코드의 품질을 다양한 측면에서 극적으로 향상시킬 수 있는 방법을 설명하고 있습니다. 우리는 객체 지향 디자인 원칙을 지지하며, 이는 자연스럽게 모듈성을 장려하고 응용 프로그램 도메인에서 실제로나 추상적으로 표현되는 개념과 잘 매핑됩니다. 우리의 목표는 당신의 코드가 **유지 관리성**, **효율성**, **일반성** 및 **재현성**을 갖는 것입니다.

## 사전 준비 사항

강의 콘텐츠에는 실행에 필요한 데이터가 포함되어 있습니다. 따라서 Python 가상 환경을 설정하고 IDE 내에서 프로젝트를 구성하고 필요한 데이터셋을 다운로드하는 것이 필요합니다.

**Python 환경**

아래 [conda](https://docs.conda.io/projects/miniconda/en/latest/) 명령어를 사용해 [environment.yml](environment.yml)의 가상환경을 만듭니다.

```shell
conda env create -f environment.yml
```

혹은 [pixi]()를 사용해도 좋습니다.

```shell
pixi install
```

## IDE의 런타임 환경 구성

IDE에서 이 저장소를 프로젝트로 열고 이전 단계에서 생성한 `pop` 환경을 사용하도록 구성합니다.

## 데이터 다운로드

데이터를 다운로드하는 두 가지 방법이 있습니다:

1. 캐글에서 직접 [다운로드](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks).
   1. CSV 파일 `spotify_data.csv`를 이 저장소의 루트에 있는 `data` 폴더에 넣으세요.

![data_folder](resources/data_folder.png)

2. 다른 방법으로는 스크립트 [load_data.py](load_data.py)를 사용하여 원시 데이터 CSV 파일을 자동으로 다운로드하고 저장소의 최상위 수준에있는 하위 폴더 `data`에 저장할 수 있습니다. 이를 위해 Kaggle API 키가 필요하며 이는 `kaggle.json`에 설정되어야 합니다(자세한 내용은 [지침](https://www.kaggle.com/docs/api) 참조).

## 강의 모듈

 1. [객체지향 프로그래밍: 기본](oop-essentials/README.md)
    - 이 모듈은 후속 모듈의 기초를 제공하는 객체지향 프로그래밍(OOP)의 핵심 원칙을 설명합니다.OOP 개념 및 설계 원칙에 대한 친숙도가 낮거나 그 혜택이 아직 명확하지 않은 경우에는 특히 이 모듈부터 시작하는 것이 좋습니다.
    - 구조적으로 OOP는 복잡성을 추가하지만, 고급 개발 도구를 사용하여 이 복잡성을 완화할 수 있습니다. 따라서 이 섹션에는 OOP와 통합 개발 환경(IDE) 간의 상호 작용에 대한 섹션도 포함되어 있습니다.

 2. [지침 원칙](Guiding-Principles.md)
    - 이 모듈은 기계 학습 응용 프로그램의 소프트웨어 개발을위한 우리의 지침 원칙 집합을 제시합니다. 이런 원칙은 개발 중에 설계 결정을 중요하게 인식할 수 있습니다.

3. [Spotify 노래 인기 예측: 리팩터링 여정](refactoring-journey/README.md) 
    -이 모듈은 Jupyter에서 구현 된 노트북에서 실험 및 프로덕션 용도의 배포를 강력하게 용이하게 만들고 유지 관리하기 쉬운 구조화 된 솔루션으로 전환하는 전체 여정을 다룹니다. 우리는 구현을 단계적으로 변형하고, 달성 한 이점을 명확하게 설명하고, 그 과정에서 구현되는 관련 원칙을 명명합니다.

4. [안티 패턴](anti-patterns/README.md) 
    - 나머지 강의는 긍정적인 설계 패턴을 시연하는 데 중점을 두는 반면, 이 모듈은 일반적인 안티 패턴에 대해 설명합니다.