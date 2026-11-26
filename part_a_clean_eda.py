from pathlib import Path
import pandas as pd
from src.shipment_prep import FEATURES,clean_shipments,build_features,write_csv
Path('outputs').mkdir(exist_ok=True)
raw=pd.read_csv('data/raw_shipments.csv')
hubs=pd.read_csv('data/hubs.csv')
dedup=raw.drop_duplicates()
clean,medians=clean_shipments(raw)
df=build_features(clean,hubs)
df['is_delayed']=df.delivery_hours.gt(df.promised_hours).astype(int)
df.to_csv('data/modelling_table.csv',index=False)
q1,q3=df.delivery_hours.quantile([.25,.75]); lo,hi=q1-1.5*(q3-q1),q3+1.5*(q3-q1)
checks={'raw_rows':len(raw),'duplicate_rows_removed':len(raw)-len(dedup),'rows_after_dedup':len(dedup),'missing_distance_km_before':dedup.distance_km.isna().sum(),'missing_parcel_weight_before':dedup.parcel_weight_kg.isna().sum(),'sentinel_weight_rows':dedup.parcel_weight_kg.gt(100).sum(),'missing_declared_value_before':dedup.declared_value.isna().sum(),'missing_payment_mode_before':dedup.payment_mode.isna().sum(),'service_type_spellings_before':dedup.service_type.nunique(),'service_type_spellings_after':clean.service_type.nunique(),**{'median_'+k:v for k,v in medians.items()},'rows_after_hub_merge':len(df),'nulls_in_features_and_target':df[FEATURES+['delivery_hours','is_delayed']].isna().sum().sum(),'delayed_count':df.is_delayed.sum(),'delay_rate':df.is_delayed.mean(),'hours_q1':q1,'hours_q3':q3,'hours_lower_bound':lo,'hours_upper_bound':hi,'hours_outlier_count':((df.delivery_hours<lo)|(df.delivery_hours>hi)).sum(),'declared_value_mean_inr':df.declared_value_inr.mean(),'declared_value_median_inr':df.declared_value_inr.median(),'log_declared_value_mean':df.log_declared_value.mean(),'log_declared_value_median':df.log_declared_value.median()}
write_csv(pd.DataFrame(checks.items(),columns=['metric','value']),'outputs/part_a_checks.csv')
df['distance_band']=pd.cut(df.distance_km,[0,200,400,600,2000],labels=['0-200','201-400','401-600','601+'])
rows=[]
for dim in ['service_type','hub_tier','distance_band']:
    g=df.groupby(dim,observed=True).agg(shipments=('shipment_id','size'),delayed=('is_delayed','sum'),delay_rate=('is_delayed','mean'),mean_hours=('delivery_hours','mean')).reset_index().rename(columns={dim:'group'})
    g.insert(0,'dimension',dim);rows.append(g)
write_csv(pd.concat(rows,ignore_index=True),'outputs/part_a_delay_by_group.csv')
write_csv(df.select_dtypes('number').corr(),'outputs/part_a_correlations.csv',index=True)
assert len(df)==1200 and checks['nulls_in_features_and_target']==0
