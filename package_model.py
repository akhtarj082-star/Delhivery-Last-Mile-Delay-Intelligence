from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from part_c_classification import split_data,pipeline
from src.shipment_prep import clean_shipments,FEATURES,write_csv
Path('models').mkdir(exist_ok=True)
cfg=pd.read_csv('outputs/part_c_final_config.csv').iloc[0]
params=dict(n_estimators=int(cfg.n_estimators),max_depth=None if cfg.max_depth=='unlimited' else int(cfg.max_depth),min_samples_leaf=int(cfg.min_samples_leaf))
xt,xv,xs,yt,yv,ys=split_data(pd.read_csv('data/modelling_table.csv'))
model=pipeline(params,k=int(cfg.selector_k)).fit(xt,yt)
joblib.dump(model,'models/delay_model.pkl',compress=3)
write_csv(pd.DataFrame({'threshold':[cfg.threshold]}),'models/delay_threshold.csv')
write_csv(pd.DataFrame({'feature':FEATURES}),'models/feature_order.csv')
_,medians=clean_shipments(pd.read_csv('data/raw_shipments.csv'))
write_csv(pd.DataFrame(medians.items(),columns=['column','median']),'models/impute_values.csv')
loaded=joblib.load('models/delay_model.pkl');diff=np.abs(loaded.predict_proba(xs)-model.predict_proba(xs)).max()
write_csv(pd.DataFrame({'metric':['test_rows_scored','reload_max_abs_probability_diff','threshold'],'value':[len(xs),diff,cfg.threshold]}),'outputs/part_d_reload_check.csv')
