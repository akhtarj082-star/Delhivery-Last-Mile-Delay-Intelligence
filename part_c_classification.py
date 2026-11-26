import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from src.shipment_prep import FEATURES,NOISE_COLUMNS,write_csv

def split_data(df):
    X,y=df[FEATURES],df.is_delayed
    xt,xp,yt,yp=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
    xv,xs,yv,ys=train_test_split(xp,yp,test_size=.5,random_state=42,stratify=yp)
    return xt,xv,xs,yt,yv,ys

def forest(params=None): return RandomForestClassifier(random_state=42,**(params or {}))
def pipeline(params=None,k=7): return Pipeline([('selector',SelectKBest(f_classif,k=k)),('model',forest(params))])
def metrics(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return dict(TN=tn,FP=fp,FN=fn,TP=tp,accuracy=accuracy_score(y,p),precision=precision_score(y,p,zero_division=0),recall=recall_score(y,p,zero_division=0),f1=f1_score(y,p,zero_division=0))
def main():
    xt,xv,xs,yt,yv,ys=split_data(pd.read_csv('data/modelling_table.csv'))
    assert [len(xt),len(xv),len(xs)]==[840,180,180] and [yt.sum(),yv.sum(),ys.sum()]==[195,42,41]
    rows=[dict(model='blind_all_on_time',**metrics(yv,np.zeros(len(yv),dtype=int)))]
    log=Pipeline([('scaler',StandardScaler()),('model',LogisticRegression())]).fit(xt,yt)
    rows.append(dict(model='logistic_regression',**metrics(yv,log.predict(xv))))
    tree_rows=[]
    for d in [2,3,4,5,6,8,10]:
        t=DecisionTreeClassifier(criterion='gini',max_depth=d,random_state=42).fit(xt,yt)
        tree_rows.append(dict(model=f'decision_tree_depth_{d}',**metrics(yv,t.predict(xv))))
    rows.append(max(tree_rows,key=lambda r:r['f1']))
    rf=forest({'n_estimators':100}).fit(xt,yt)
    rows.append(dict(model='random_forest_default',**metrics(yv,rf.predict(xv))))
    write_csv(pd.DataFrame(rows),'outputs/part_c_models.csv')
    fs=SelectKBest(f_classif,k='all').fit(xt,yt)
    feature=pd.DataFrame(dict(feature=FEATURES,f_score=fs.scores_,rf_importance=rf.feature_importances_,is_noise=[f in NOISE_COLUMNS for f in FEATURES]))
    feature['f_rank']=feature.f_score.rank(ascending=False,method='first').astype(int)
    write_csv(feature[['feature','f_score','f_rank','rf_importance','is_noise']],'outputs/part_c_feature_scores.csv')
    write_csv(pd.DataFrame([dict(k=k,val_f1=f1_score(yv,pipeline(k=k).fit(xt,yt).predict(xv))) for k in [5,7,10,12,17]]),'outputs/part_c_selectk.csv')
    selector=SelectKBest(f_classif,k=7).fit(xt,yt)
    a,b=selector.transform(xt),selector.transform(xv)
    grid={'n_estimators':[50,100,150],'max_depth':[4,6,8],'min_samples_leaf':[1,5,10]}
    searches=[('grid_search',GridSearchCV(forest(),grid,cv=5,n_jobs=2),27),('randomized_search',RandomizedSearchCV(forest(),grid,n_iter=9,cv=5,random_state=42,n_jobs=2),9)]
    tuning=[];params={}
    for name,search,n in searches:
        search.fit(a,yt);params[name]=search.best_params_
        tuning.append(dict(search=name,candidates=n,cv_folds=5,total_fits=n*5,best_params=json.dumps(search.best_params_,sort_keys=True),best_cv_accuracy=search.best_score_,val_f1=f1_score(yv,search.predict(b))))
    default='default_forest_same_7_features';params[default]={'n_estimators':100,'max_depth':None,'min_samples_leaf':1}
    tuning.append(dict(search=default,candidates=1,cv_folds=0,total_fits=1,best_params='defaults',best_cv_accuracy=np.nan,val_f1=f1_score(yv,forest(params[default]).fit(a,yt).predict(b))))
    write_csv(pd.DataFrame(tuning),'outputs/part_c_tuning.csv')
    priority={default:0,'grid_search':1,'randomized_search':2}
    chosen=max(tuning,key=lambda r:(r['val_f1'],-priority[r['search']]))['search'];p=params[chosen]
    model=pipeline(p).fit(xt,yt);prob=model.predict_proba(xv)[:,1]
    thresholds=[dict(threshold=t,**metrics(yv,(prob>=t).astype(int))) for t in [.1,.2,.3,.4,.5,.6,.7,.8,.9]]
    write_csv(pd.DataFrame(thresholds),'outputs/part_c_thresholds.csv')
    t=max((r['threshold'] for r in thresholds if r['recall']>=.8),default=.1)
    test=model.predict_proba(xs)[:,1]
    write_csv(pd.DataFrame([dict(threshold=t,**metrics(ys,(test>=t).astype(int)))]),'outputs/part_c_final_test.csv')
    write_csv(pd.DataFrame([dict(selector_k=7,n_estimators=p['n_estimators'],max_depth=p['max_depth'] if p['max_depth'] is not None else 'unlimited',min_samples_leaf=p['min_samples_leaf'],threshold=t,chosen_from=chosen)]),'outputs/part_c_final_config.csv')
if __name__=='__main__': main()
