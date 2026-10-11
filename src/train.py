"""Property-disjoint evaluation of RF, gradient boosting and XGBoost."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from xgboost import XGBRegressor
from .model import estimator,metrics,positioning_weights
from .prepare import FEATURES,CLUSTER_FEATURES

SEED=42

def assign_splits(frame):
    groups=np.array(sorted(frame.property_id.unique()))
    np.random.default_rng(SEED).shuffle(groups)
    n=len(groups); a=int(.6*n); b=int(.75*n); c=int(.85*n)
    mapping={g:role for role,ids in [('train',groups[:a]),('validation',groups[a:b]),('calibration',groups[b:c]),('test',groups[c:])] for g in ids}
    labels=frame.property_id.map(mapping)
    assert labels.notna().all()
    return labels

def hotel_profiles(frame):
    return frame.groupby('property_id').agg(name=('name','first'),district=('district','first'),
        canonical_url=('canonical_url','first'),median_price_eur=('price_per_night_eur','median'),
        distance_center_km=('distance_center_km','median'),log_review_count=('log_review_count','median'),
        amenity_mentions_count=('amenity_mentions_count','median'),split=('split','first'),quotes=('adults','size')).assign(
        log_median_price=lambda d:np.log1p(d.median_price_eur)).reset_index()

def train(path='data/processed/hotel_quotes.csv'):
    Path('reports').mkdir(exist_ok=True);Path('artifacts').mkdir(exist_ok=True)
    data=pd.read_csv(path);data['split']=assign_splits(data)
    data[['property_id','split']].drop_duplicates().to_csv('reports/property_splits.csv',index=False)
    subsets={role:data.loc[data.split==role].copy() for role in ['train','validation','calibration','test']}
    train,valid,cal,test=[subsets[k] for k in subsets]
    configurations={
        'random_forest':(RandomForestRegressor(n_estimators=200,random_state=SEED,n_jobs=2),
            {'regressor__model__max_depth':[6,None],'regressor__model__min_samples_leaf':[2,5]}),
        'gradient_boosting':(HistGradientBoostingRegressor(max_iter=180,min_samples_leaf=8,early_stopping=False,random_state=SEED),
            {'regressor__model__max_leaf_nodes':[7,15],'regressor__model__l2_regularization':[1,10]}),
        'xgboost':(XGBRegressor(n_estimators=200,objective='reg:squarederror',tree_method='hist',n_jobs=2,random_state=SEED,learning_rate=.05),
            {'regressor__model__max_depth':[2,4],'regressor__model__reg_lambda':[1,10]})}
    models,searches,validation={},{},{}
    for name,(base,grid) in configurations.items():
        print(f'Tuning {name}',flush=True)
        search=GridSearchCV(estimator(base),grid,cv=GroupKFold(3),scoring='neg_mean_absolute_error',n_jobs=1)
        search.fit(train[FEATURES],train.price_per_night_eur,groups=train.property_id)
        models[name]=search.best_estimator_
        searches[name]={'best_params':search.best_params_,'cv_mae_eur':-float(search.best_score_)}
        validation[name]=metrics(valid.price_per_night_eur,search.predict(valid[FEATURES]))
        pd.DataFrame(search.cv_results_).to_csv(f'reports/search_{name}.csv',index=False)
    selected=min(validation,key=lambda k:validation[k]['mae_eur'])
    fit=data.loc[data.split.isin(['train','validation'])]
    model=clone(models[selected]).fit(fit[FEATURES],fit.price_per_night_eur)
    prediction=np.maximum(model.predict(test[FEATURES]),0)
    cal_prediction=np.maximum(model.predict(cal[FEATURES]),0)
    residual=np.abs(np.log1p(cal.price_per_night_eur)-np.log1p(cal_prediction))
    group_scores=pd.DataFrame({'property_id':cal.property_id.to_numpy(),'residual':residual.to_numpy()}).groupby('property_id').residual.max()
    rank=min(int(np.ceil((len(group_scores)+1)*.9)),len(group_scores))
    radius=float(np.sort(group_scores)[rank-1])
    lower=np.maximum(np.expm1(np.log1p(prediction)-radius),0);upper=np.expm1(np.log1p(prediction)+radius)
    median=float(fit.price_per_night_eur.median())
    district=fit.groupby(['district','adults']).price_per_night_eur.median()
    district_prediction=np.array([district.get((row.district,row.adults),median) for row in test.itertuples()])
    result={'selected_model':selected,'features':FEATURES,'search':searches,'validation':validation,
        'test':{'selected':metrics(test.price_per_night_eur,prediction),
                'global_median':metrics(test.price_per_night_eur,np.full(len(test),median)),
                'district_occupancy_median':metrics(test.price_per_night_eur,district_prediction)},
        'splits':{k:{'quotes':len(v),'properties':int(v.property_id.nunique())} for k,v in subsets.items()},
        'interval':{'nominal_coverage':.9,'calibration_quotes':len(cal),'calibration_properties':int(cal.property_id.nunique()),
            'log_radius':radius,'test_coverage':float(np.mean((test.price_per_night_eur>=lower)&(test.price_per_night_eur<=upper))),
            'median_width_eur':float(np.median(upper-lower))},'seed':SEED}
    result['interval']['calibration_unit']='maximum absolute log residual per property'
    result['interval']['test_property_coverage']=float(pd.DataFrame({'property_id':test.property_id.to_numpy(),
        'covered':((test.price_per_night_eur>=lower)&(test.price_per_night_eur<=upper)).to_numpy()}).groupby('property_id').covered.all().mean())
    test['prediction_eur']=prediction;test['lower90_eur']=lower;test['upper90_eur']=upper
    test.to_csv('reports/test_predictions.csv',index=False)
    # Resample complete held-out properties, not individual quote rows.
    rng=np.random.default_rng(SEED);ids=test.property_id.unique()
    errors={g:np.abs(v.prediction_eur-v.price_per_night_eur).to_numpy() for g,v in test.groupby('property_id')}
    boot=[float(np.concatenate([errors[g] for g in rng.choice(ids,size=len(ids),replace=True)]).mean()) for _ in range(2000)]
    result['test_mae_property_bootstrap_95ci_eur']=np.quantile(boot,[.025,.975]).tolist()
    imp=permutation_importance(model,cal[FEATURES],cal.price_per_night_eur,n_repeats=10,
        scoring='neg_mean_absolute_error',random_state=SEED,n_jobs=1)
    pd.DataFrame({'feature':FEATURES,'mae_increase_eur':imp.importances_mean,'std_eur':imp.importances_std}).sort_values(
        'mae_increase_eur',ascending=False).to_csv('reports/importance.csv',index=False)
    breakdown=[]
    for district_name,group in test.groupby('district'):
        if len(group)>1:breakdown.append({'district':district_name,**metrics(group.price_per_night_eur,group.prediction_eur)})
    pd.DataFrame(breakdown).to_csv('reports/errors_by_district.csv',index=False)
    joblib.dump({'model':model,'features':FEATURES,'log_interval_radius':radius,
        'supported_stay':'2026-01-24 to 2026-01-25','children':1,'rooms':1},'artifacts/room_price.joblib',compress=3)
    data['prediction_eur']=np.maximum(model.predict(data[FEATURES]),0)
    data.to_csv('reports/all_quotes_predictions.csv',index=False)
    profiles=hotel_profiles(data)
    train_profiles=profiles.loc[profiles.split.isin(['train','validation'])]
    transform=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),
        ('positioning_weights',FunctionTransformer(positioning_weights))])
    z=transform.fit_transform(train_profiles[CLUSTER_FEATURES]);scores={};candidates={}
    for k in range(2,7):
        candidate=KMeans(n_clusters=k,n_init=30,random_state=SEED).fit(z)
        candidates[k]=candidate;scores[k]=float(silhouette_score(z,candidate.labels_))
    # Three business labels are requested; score alternatives without changing that taxonomy.
    cluster=candidates[3]
    centers=train_profiles.assign(cluster=cluster.labels_).groupby('cluster').median_price_eur.median().sort_values()
    label_map=dict(zip(centers.index.tolist(),['Economy','Mid-range','Upscale']))
    profiles['cluster']=cluster.predict(transform.transform(profiles[CLUSTER_FEATURES]))
    profiles['positioning']=profiles.cluster.map(label_map)
    pca=PCA(n_components=2,random_state=SEED).fit(z)
    xy=pca.transform(transform.transform(profiles[CLUSTER_FEATURES]));profiles['pc1']=xy[:,0];profiles['pc2']=xy[:,1]
    profiles.to_csv('reports/property_profiles.csv',index=False)
    summary=profiles.groupby('positioning').agg(properties=('property_id','size'),median_price_eur=('median_price_eur','median'),
        center_km=('distance_center_km','median'),amenity_mentions=('amenity_mentions_count','median')).reindex(['Economy','Mid-range','Upscale'])
    summary.to_csv('reports/cluster_summary.csv')
    result['clustering']={'k':3,'silhouette_by_k':scores,'best_diagnostic_k':max(scores,key=scores.get),
        'standardized_feature_weights':dict(zip(CLUSTER_FEATURES,[2.0,.5,.25,.5])),
        'label_map':label_map,'features':CLUSTER_FEATURES,'training_properties':len(train_profiles),
        'pca_explained_variance':pca.explained_variance_ratio_.tolist(),'counts':profiles.positioning.value_counts().to_dict()}
    joblib.dump({'preprocessing':transform,'kmeans':cluster,'pca':pca,'labels':label_map,'features':CLUSTER_FEATURES},'artifacts/positioning.joblib',compress=3)
    Path('reports/metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':train()
