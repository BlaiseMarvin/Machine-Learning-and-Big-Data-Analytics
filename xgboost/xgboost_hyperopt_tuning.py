import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import numpy as np

# Generate synthetic regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter search space
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'gamma': hp.uniform('gamma', 0, 5),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'rate_drop': hp.uniform('rate_drop', 0.0, 0.5),  # DART-specific
    'sample_type': hp.choice('sample_type', ['uniform', 'weighted']),  # DART-specific
    'normalize_type': hp.choice('normalize_type', ['tree', 'forest']),  # DART-specific
    'skip_drop': hp.uniform('skip_drop', 0.0, 0.5),  # DART-specific
    'one_drop': hp.choice('one_drop', [True, False])  # DART-specific
}

# Define objective function for Hyperopt
def objective(params):
    # Convert max_depth to int (required by XGBoost)
    params['max_depth'] = int(params['max_depth'])
    
    # Initialize XGBRegressor with DART booster
    model = xgb.XGBRegressor(
        booster='dart',
        objective='reg:squarederror',
        eval_metric='rmse',
        n_estimators=500,  # Max boosting rounds
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        gamma=params['gamma'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        rate_drop=params['rate_drop'],
        sample_type=params['sample_type'],
        normalize_type=params['normalize_type'],
        skip_drop=params['skip_drop'],
        one_drop=params['one_drop'],
        seed=42
    )
    
    # Train with early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Get validation RMSE
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    # Return loss (RMSE) and status
    return {'loss': rmse, 'status': STATUS_OK}

# Run Hyperopt optimization
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,  # Tree-structured Parzen Estimator
    max_evals=50,      # Number of trials
    trials=trials,
    rstate=np.random.default_rng(42)  # For reproducibility
)

# Convert max_depth to int for final model
best['max_depth'] = int(best['max_depth'])
# Map choice indices to actual values
best['sample_type'] = ['uniform', 'weighted'][best['sample_type']]
best['normalize_type'] = ['tree', 'forest'][best['normalize_type']]
best['one_drop'] = [True, False][best['one_drop']]

# Train final model with best hyperparameters
final_model = xgb.XGBRegressor(
    booster='dart',
    objective='reg:squarederror',
    eval_metric='rmse',
    n_estimators=500,
    learning_rate=best['learning_rate'],
    max_depth=best['max_depth'],
    gamma=best['gamma'],
    subsample=best['subsample'],
    colsample_bytree=best['colsample_bytree'],
    rate_drop=best['rate_drop'],
    sample_type=best['sample_type'],
    normalize_type=best['normalize_type'],
    skip_drop=best['skip_drop'],
    one_drop=best['one_drop'],
    seed=42
)

final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    early_stopping_rounds=10,
    verbose=True
)

# Evaluate final model
y_pred = final_model.predict(X_val)
final_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Final Validation RMSE: {final_rmse:.4f}")
print(f"Best Hyperparameters: {best}")
print(f"Best Iteration: {final_model.best_iteration}")
print(f"Best Score: {final_model.best_score}")