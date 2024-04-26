# 사례 연구 - 단계 1: 평가 함수 추출

먼저 수정된 스크립트 [run_regressor_evaluation.py](run_regressor_evaluation.py)를 살펴봅니다.

첫 번째 단계에 평가 로직을 함수로 선언했습니다.

```python
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    print(f"{model}: MAE={mae:.1f}")
```

이를 통해 이 스크립트 내에서 함수를 재사용할 수 있게 되어 코드 중복을 제거하고 유지보수성을 높일 수 있습니다.

```python
# 모델 평가하기
evaluate_model(LogisticRegression(solver='lbfgs', max_iter=1000), X_train, y_train, X_test, y_test)
evaluate_model(KNeighborsRegressor(n_neighbors=1), X_train, y_train, X_test, y_test)
evaluate_model(RandomForestRegressor(n_estimators=100), X_train, y_train, X_test, y_test)
evaluate_model(DecisionTreeRegressor(random_state=42, max_depth=2), X_train, y_train, X_test, y_test)
```

[다음 단계](../02c-case-study-2-evaluation-abstraction/)