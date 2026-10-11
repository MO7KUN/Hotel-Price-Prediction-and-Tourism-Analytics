"""Shared inference and preprocessing contracts."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .prepare import NUMERIC, CATEGORICAL, FEATURES

def positioning_weights(values):
    """Price-led market positioning, not an official quality or star rating."""
    return np.asarray(values)*np.array([2.0,0.5,0.25,0.5])

def estimator(base):
    numeric=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())])
    category=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    prep=ColumnTransformer([('numeric',numeric,NUMERIC),('category',category,CATEGORICAL)])
    return TransformedTargetRegressor(regressor=Pipeline([('prepare',prep),('model',base)]),func=np.log1p,inverse_func=np.expm1)

def metrics(y,pred):
    y=np.asarray(y);pred=np.maximum(np.asarray(pred),0)
    return {'n':len(y),'mae_eur':float(mean_absolute_error(y,pred)),
        'rmse_eur':float(np.sqrt(mean_squared_error(y,pred))),'r2':float(r2_score(y,pred)),
        'median_ape_percent':float(100*np.median(np.abs(y-pred)/y)),
        'log_mae':float(mean_absolute_error(np.log1p(y),np.log1p(pred)))}

def validate_input(frame):
    missing=sorted(set(FEATURES)-set(frame.columns))
    if missing:raise ValueError(f'Missing input columns: {missing}')
    if not pd.to_numeric(frame.adults,errors='raise').between(1,3).all():
        raise ValueError('Supported observed range: 1–3 adults, one child, one room, one night')
    for feature in NUMERIC:
        values=pd.to_numeric(frame[feature],errors='raise')
        if not np.isfinite(values.dropna()).all():raise ValueError(f'Nonfinite input: {feature}')
        if (values.dropna()<0).any():raise ValueError(f'Negative input: {feature}')
        if (feature.startswith('mentions_') or feature in ['beach_distance_missing','sustainability_certification','description_missing']) and not values.dropna().isin([0,1]).all():
            raise ValueError(f'Expected binary indicator: {feature}')
    return frame
