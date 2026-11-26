import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments,build_features,write_csv
raw=pd.read_csv('data/new_shipments.csv');hubs=pd.read_csv('data/hubs.csv')
medians=pd.read_csv('models/impute_values.csv').set_index('column')['median'].to_dict()
features=pd.read_csv('models/feature_order.csv').feature.tolist()
threshold=float(pd.read_csv('models/delay_threshold.csv').threshold.iloc[0])
model=joblib.load('models/delay_model.pkl')
clean,_=clean_shipments(raw,medians);df=build_features(clean,hubs)
assert not df[features].isna().any().any()
df['delay_probability']=model.predict_proba(df[features])[:,1].round(4)
df['flagged']=df.delay_probability.ge(threshold).astype(int)
df=df.merge(pd.read_csv('outputs/hub_segments.csv')[['hub_id','segment']],on='hub_id',how='left',validate='many_to_one',sort=False)
df['action']=np.where(df.flagged.eq(0),'No action',np.where(df.segment.eq('Escalate'),'Escalate to hub manager','Proactive customer call'))
write_csv(df[['shipment_id','hub_id','segment','delay_probability','flagged','action']],'outputs/new_shipment_actions.csv')
assert len(df)==48 and df.shipment_id.is_unique
