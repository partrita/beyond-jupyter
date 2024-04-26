# 단계 13: 도커로 배포하기

우리 여정의 최종 결과물은 최소한의 노력으로 배포할 수 있는 모델입니다. 모델 배포를 위해 도커 이미지 빌드를 위한 폴더 `app`을 새로 만듭니다.

- [main.py](app/main.py): fastAPI 애플리케이션의 정의, pydantic을 통한 입력 데이터 유효성 검사를 포함합니다.
- 이전 단계에서 보관된 회귀 모델을 수정하지 않고도 입력 데이터 클래스 인스턴스로부터 받은 입력 데이터 프레임에 직접 적용할 수 있음을 주목하세요.
- ```python
  @app.post("/predict/")
  def predict(input_data: PredictionInput):
      data = pd.DataFrame([dict(input_data)])
      prediction = model.predict(data)
      return prediction.to_dict(orient="records")
  ```

- [environment-prod.yml](app/environment-prod.yml): fastAPI 애플리케이션을 실행하는 데 필요한 종속성을 포함하며 모든 종속성을 완전히 고정합니다(`conda env export`를 사용하여 생성되었습니다).
- [Dockerfile](app/Dockerfile): 모델 추론을 실행하기 위한 최소한의 `Dockerfile`입니다.

도커 이미지는 이전 단계에서 `run_regressor_evaluation.py`에 의해 저장된 최적의 회귀 모델을 사용할 것입니다. 따라서 이미지가 작동하려면 이전 단계 디렉토리에서 해당 스크립트를 적어도 한 번 실행했는지 확인하세요.

이미지를 빌드하려면 다음을 최상위 폴더에서 실행하세요.

```shell
docker build -t spotify-popularity-estimator -f refactoring-journey/step10-deployment/app/Dockerfile .
```

컨테이너를 실행하려면 다음을 실행하세요.

```shell
docker run -p 80:80 spotify-popularity-estimator
```

[run_fastapi_test.py](run_fastapi_test.py) 스크립트를 사용하여 이미지를 빌드하고 컨테이너를 시작하고 샘플 데이터로 `GET` 및 `POST` 요청을 보낼 수 있습니다. 이로써 리팩터링 강의를 마칩니다.

## 이 단계에서 다룬 원칙

- 재사용 가능한 구성 요소 개발