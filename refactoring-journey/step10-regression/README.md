# 단계 10: 회귀

기계 학습에서 가장 흔한 문제는 회귀 문제입니다. 이번 단계에서는 회귀 문제에 대해서 좀 더 알아봅니다.

다음 변경 사항이 있습니다.
  - 두 번째 모델 팩토리 `RegressionModelFactory`를 도입하고, 기존 것을 `ClassificationModelFactory`로 이름을 변경합니다. 이 새로운 팩토리에서도 일부 유형의 모델을 동일하게 구현합니다.
  - `FeatureGeneratorMeanArtistPopularity`를 수정하여 회귀 케이스도 지원하도록 하고, 두 케이스를 구분하는 생성자 매개변수를 추가하고, 추가로 회귀 특정 기능 `FeatureName.MEAN_ARTIST_POPULARITY`를 등록합니다.
  - 데이터 집합 표현을 확장하여 회귀 케이스를 지원하도록 하고, 대상 변수를 이에 맞게 수정합니다.
  - 회귀 모델을 사용하여 분류 문제를 처리하는 래퍼 클래스 `VectorClassificationModelFromVectorRegressionModel`를 구현합니다. 학습 중에 얻은 최상의 회귀 모델을 선택적으로 저장하고 해당 파일이 존재하는 경우 해당 래핑된 모델을 평가된 분류 모델 목록에 추가합니다.

새로운 [회귀용 실행 가능한 스크립트](run_regressor_evaluation.py)가 새로운 실험을 처리합니다.

이제 회귀 및 분류용 스크립트를 모두 실행하여 일부 예비 결과를 얻을 수 있습니다.

[다음 단계](../step11-hyperopt/README.md)