"""Batch price prediction and optional observed-profile segmentation."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .model import validate_input

def predict(frame,artifacts='artifacts'):
    validate_input(frame)
    bundle=joblib.load(Path(artifacts)/'room_price.joblib')
    value=np.maximum(bundle['model'].predict(frame[bundle['features']]),0)
    result=frame.copy();result['prediction_eur']=value
    radius=bundle['log_interval_radius']
    result['lower90_eur']=np.maximum(np.expm1(np.log1p(value)-radius),0)
    result['upper90_eur']=np.expm1(np.log1p(value)+radius)
    return result

def segment(profiles,artifacts='artifacts'):
    bundle=joblib.load(Path(artifacts)/'positioning.joblib')
    missing=set(bundle['features'])-set(profiles.columns)
    if missing:raise ValueError(f'Missing observed property profile inputs: {sorted(missing)}')
    result=profiles.copy()
    result['cluster']=bundle['kmeans'].predict(bundle['preprocessing'].transform(profiles[bundle['features']]))
    result['positioning']=result.cluster.map(bundle['labels'])
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',default='predictions.csv')
    parser.add_argument('--artifacts',default='artifacts');args=parser.parse_args()
    result=predict(pd.read_csv(args.input),args.artifacts);result.to_csv(args.output,index=False)
    print(result[['prediction_eur','lower90_eur','upper90_eur']].to_string(index=False))
