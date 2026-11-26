import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from src.shipment_prep import FEATURES,write_csv

df=pd.read_csv('data/modelling_table.csv'); X,y=df[FEATURES],df.delivery_hours
xt,xp,yt,yp=train_test_split(X,y,test_size=.3,random_state=42)
xv,xs,yv,ys=train_test_split(xp,yp,test_size=.5,random_state=42)
scaler=StandardScaler().fit(xt)
checks={'train_rows':len(xt),'val_rows':len(xv),'test_rows':len(xs),'train_scaled_max_abs_mean':np.abs(scaler.transform(xt).mean(axis=0)).max(),'test_scaled_max_abs_mean':np.abs(scaler.transform(xs).mean(axis=0)).max()}
write_csv(pd.DataFrame(checks.items(),columns=['metric','value']),'outputs/part_b_checks.csv')
def rmse(y,p): return np.sqrt(mean_squared_error(y,p))
def metrics(name,split,y,p):
    mse=mean_squared_error(y,p)
    return dict(model=name,split=split,MAE=mean_absolute_error(y,p),MSE=mse,RMSE=np.sqrt(mse),R2=r2_score(y,p))
depths=[1,2,3,4,6,8,12,None]; diagnostic=[]
for d in depths:
    m=DecisionTreeRegressor(max_depth=d,random_state=42).fit(xt,yt)
    diagnostic.append(dict(max_depth=d if d is not None else 'unlimited',train_rmse=rmse(yt,m.predict(xt)),val_rmse=rmse(yv,m.predict(xv))))
write_csv(pd.DataFrame(diagnostic),'outputs/part_b_tree_depth.csv')
best=depths[min(range(len(depths)),key=lambda i:diagnostic[i]['val_rmse'])]
models={'linear_regression':Pipeline([('scaler',StandardScaler()),('model',LinearRegression())]),'decision_tree':DecisionTreeRegressor(max_depth=best,random_state=42),'random_forest':RandomForestRegressor(n_estimators=100,random_state=42),'gradient_boosting':GradientBoostingRegressor(n_estimators=100,learning_rate=.1,random_state=42)}
rows=[metrics('mean_baseline','val',yv,np.full(len(yv),yt.mean()))]
for name,m in models.items():
    m.fit(xt,yt);rows.append(metrics(name,'val',yv,m.predict(xv)))
learning=[]
for lr in [.01,.1,1.]:
    p=GradientBoostingRegressor(n_estimators=100,learning_rate=lr,random_state=42).fit(xt,yt).predict(xv)
    learning.append(dict(learning_rate=lr,val_rmse=rmse(yv,p),val_r2=r2_score(yv,p)))
write_csv(pd.DataFrame(learning),'outputs/part_b_gb_learning_rate.csv')
rank=sorted(rows[1:],key=lambda r:r['RMSE']);cv=KFold(n_splits=5,shuffle=True,random_state=42);cv_rows=[]
for r in rank[:2]:
    scores=cross_val_score(models[r['model']],pd.concat([xt,xv]),pd.concat([yt,yv]),cv=cv)
    cv_rows.append(dict(model=r['model'],mean_r2=scores.mean(),std_r2=scores.std()))
write_csv(pd.DataFrame(cv_rows),'outputs/part_b_cv.csv')
name=rank[0]['model'];rows.append(metrics(name,'test',ys,models[name].predict(xs)))
write_csv(pd.DataFrame(rows),'outputs/part_b_metrics.csv')
