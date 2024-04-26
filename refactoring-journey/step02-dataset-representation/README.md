# 단계 2: 데이터셋 표현

이번 단계에서는 데이터셋을 위한 표현을 소개합니다. **데이터를 결정하는 매개변수를 명시적으로 표현합니다.** 새롭게 도입된 `Dataset` 클래스의 속성들에 데이터를 결정하는 매개변수를 명시적으로 표현하여 선택 사항을 명확하게합니다.

- 우리는 실제로 인기가 없는 노래를 삭제하는 개념에 동의하지 않습니다. 이것은 사실상 사기에 해당합니다. 실제로 많은 노래가 인기가 없고, 우리의 모델은 이를 처리할 수 있어야 하며 이상적으로는 예측할 수도 있어야 합니다. 따라서 이것을 기본적으로 비활성화된 옵션으로 만들었습니다.
- 또한 노래가 인기 있는 것으로 간주되는 임계값을 명시적 매개변수로 추가했습니다. 나중에 이를 변경하고자 할 수 있으며, 결국 어떤 하드 임계값이든 어느 정도 임의적이라는 점을 고려하면 됩니다. 이 문제를 완전히 피하려고 이를 회귀 문제로 취급하기로 결정할 수도 있습니다.
- 또한 개발 속도를 높이기 위해 데이터를 샘플링할 수 있도록 허용합니다. 개발의 여러 단계에서는 가능한 많은 데이터를 고려하는 아주 좋은 모델을 학습할 필요가 없습니다. 따라서 코드가 작동하는지 확인하고 어느 정도 작동하는지 대략적인 추정을 얻고 싶은 실험에서는 데이터의 작은 샘플을 사용할 것입니다.

이 프로젝트에서는 하나의 데이터셋만 사용할 것입니다. 그러나 여전히 여러 가지 변형을 실험할 것입니다. 따라서 해당 매개변수를 명시적으로 설정해야 합니다. 이 매개변수들은 데이터의 구체적인 표현을 결정하며, 이를 기록함으로써 데이터 버전 관리와 유사한 역할을 수행할 것입니다.



 ```python
class Dataset:
  def __init__(self, num_samples: Optional[int] = None, drop_zero_popularity: bool = False, threshold_popular: int = 50,
          random_seed: int = 42):
      """
      :param num_samples: 데이터 프레임에서 추출할 샘플의 수; None이면 모든 샘플 사용
      :param drop_zero_popularity: 인기도가 0인 데이터 포인트를 삭제할지 여부
      :param threshold_popular: 인기 있는 곡으로 간주되는 임계값
      :param random_seed: 데이터 포인트를 샘플링할 때 사용할 난수 시드
      """
      self.num_samples = num_samples
      self.threshold_popular = threshold_popular
      self.drop_zero_popularity = drop_zero_popularity
      self.random_seed = random_seed

  def load_data_frame(self) -> pd.DataFrame:
      """
      :return: 이 데이터셋의 전체 데이터 프레임(클래스 열 포함)
      """
      df = pd.read_csv(config.csv_data_path()).dropna()
      if self.num_samples is not None:
          df = df.sample(self.num_samples, random_state=self.random_seed)
      df[COL_GEN_POPULARITY_CLASS] = df[COL_POPULARITY].apply(lambda x: CLASS_POPULAR if x >= self.threshold_popular else CLASS_UNPOPULAR)
      return df

  def load_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
      """
      :return: X는 모든 속성을 포함하는 데이터 프레임이고 y는 해당 클래스 값의 시리즈입니다.
      """
      df = self.load_data_frame()
      return df.drop(columns=COL_GEN_POPULARITY_CLASS), df[COL_GEN_POPULARITY_CLASS]
 ```

- **데이터를 건드리지 않고 모델이 데이터를 처리할 방법을 결정하도록 하겠습니다**. 처음부터 데이터 열을 삭제하거나 수정할 필요는 없습니다. 모델은 데이터의 일부만 사용하기를 선택할 수 있습니다. 서로 다른 모델은 완전히 다른 작업을 수행할 수 있어야 합니다.
- **코드에서 상수 리터럴을 사용하지 않고 명명된 상수를 사용할 것입니다.**
  - 문자열 리터럴을 통한 데이터 열 참조는 오류가 발생하기 쉽습니다. 오타가 발생할 수 있습니다.
  - 문자열을 사용하면 IDE에서 열 이름이 어떻게 되는지 확실하지 않을 때 도움을 받을 수 없습니다. 잘 정의된 식별자 체계로 코드에 상수를 추가하면 IDE에서 최적의 지원을 받을 수 있습니다. `COL_`로 시작하는 모든 식별자에 대해 자동 완성을 요청하면 가능한 옵션 목록이 표시됩니다.
  - 열 이름이 변경되더라도 업데이트해야 할 곳이 하나뿐입니다(물론 이 장난감 프로젝트에서는 문제가 되지 않습니다).
    - 원본 열 외에 추가한 열(예: 클래스 열)에 대한 상수를 추가하고, 나중에 유용할 수 있는 의미적 그룹(전용 접두사 `COLS_`)을 도입합니다.
      ```python
      # 범주형 특성
      COL_GENRE = "genre"
      COL_KEY = "key"
      COL_MODE = "mode"  # 이진
      COLS_MUSICAL_CATEGORIES = [COL_GENRE, COL_KEY, COL_MODE]
      ```
- `apply` 함수를 사용하여 클래스 열 생성을 단순화했습니다.
  - 이전:
     ```python
     popularity_verdict['verdict'] = ''
     for i, row in popularity_verdict.iterrows():
     score = 'low'
     if row.popularity >= 50:
     score = 'popular'
     popularity_verdict.at[i, 'verdict'] = score
     ```
  - 이후:
     ```python
     df[COL_GEN_POPULARITY_CLASS] = df[COL_POPULARITY].apply(lambda x: CLASS_POPULAR if x >= self.threshold_popular else CLASS_UNPOPULAR)
     ```
- 모델이 실제로 사용하는 특성 집합을 명시했습니다(변수 `cols_used_by_models`). 이전에는 몇 가지 삭제 및 투영 후에 남은 숫자 열의 집합이었습니다. 이미 실제로 범주형인 열(예: 키 및 모드)에 `StandardScaler`를 적용하는 것이 적어도 의문스러울 정도입니다. 이에 대해 후속 단계에서 무언가를 처리할 것입니다.
  - 이전:
     ```python
     pop_ver_att = popularity_verdict[['year', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_mins']]
     X = pop_ver_att.select_dtypes(include='number')
     ```
  - 이후:
     ```python
     cols_used_by_models = [COL_YEAR, *COLS_MUSICAL_DEGREES, COL_KEY, COL_MODE, COL_TEMPO, COL_TIME_SIGNATURE, COL_LOUDNESS, COL_DURATION_MS]
     X = X[cols_used_by_models]
     ```
  
# 이번 단계에서 다룬 원칙

- 적절한 추상화 찾기
- 매개변수화 노출
- 데이터 버전 관리

[다음 단계](../step03-refactoring/README.md)
