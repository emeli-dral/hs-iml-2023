import pandas as pd

from sklearn.metrics import mean_squared_error

import xgboost as xgb

def looad_data(path):
    data = pd.read_parquet(path)
    data.lpep_dropoff_datetime = pd.to_datetime(data.lpep_dropoff_datetime)
    data.lpep_pickup_datetime = pd.to_datetime(data.lpep_pickup_datetime)

    data['duration'] = data.lpep_dropoff_datetime - data.lpep_pickup_datetime
    data.duration = data.duration.apply(lambda td: td.total_seconds() / 60)
    data = data[(data.duration >= 1) & (data.duration <= 60)]
    
    data['PULocationID'].astype(str, copy=False)
    data['DOLocationID'].astype(str, copy=False)
    return data

def generate_datasets(train_frame, val_frame):
    num_features = ['trip_distance', 'extra', 'fare_amount']
    cat_features = ['PULocationID', 'DOLocationID']

    X_train = train_frame[num_features + cat_features]
    X_val = val_frame[num_features + cat_features] 

    y_train = train_frame['duration']
    y_val = val_frame['duration'] 
    return X_train, X_val, y_train, y_val

def train_model(X_train, y_train, X_val, y_val):
    best_params = {
        'max_depth': 5,
        'min_child': 19.345653147972058,
        'objective': 'reg:linear',
        'reg_alpha': 0.031009193638004067,
        'reg_lambda': 0.013053945835415701,
        'seed': 111
    }

    train = xgb.DMatrix(X_train, label=y_train)
    validation = xgb.DMatrix(X_val, label=y_val)

    booster = xgb.train(
        params = best_params,
        dtrain = train,
        evals = [(validation, "validation")],
        num_boost_round = 500,
        early_stopping_rounds = 50,
    )

    return booster

def estimate_quality(model, X_val, y_val):
    validation = xgb.DMatrix(X_val, label=y_val)
    y_pred = model.predict(validation)
    return mean_squared_error(y_pred, y_val, squared=False)

if __name__ == '__main__':
    train_frame = looad_data('data/green_tripdata_2022-01.parquet')
    val_frame = looad_data('data/green_tripdata_2022-02.parquet')
    print(f"data loaded")

    X_train, X_val, y_train, y_val = generate_datasets(train_frame, val_frame)
    print(f"datsets are generate")

    model = train_model(X_train, y_train, X_val, y_val)
    print(f"model trained")

    rmse = estimate_quality(model, X_val, y_val)
    print(f"rmse: {rmse}")

