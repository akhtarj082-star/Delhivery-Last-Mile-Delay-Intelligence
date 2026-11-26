from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.shipment_prep import write_csv

df=pd.read_csv('data/modelling_table.csv');hubs=pd.read_csv('data/hubs.csv')
profile=df.groupby('hub_id',sort=True).agg(shipments=('shipment_id','size'),mean_delivery_hours=('delivery_hours','mean'),delay_rate=('is_delayed','mean'),mean_distance_km=('distance_km','mean'),mean_n_stops=('n_stops','mean')).reset_index()
write_csv(profile,'outputs/hub_profile.csv')
cols=['mean_delivery_hours','delay_rate','mean_distance_km','mean_n_stops'];X=StandardScaler().fit_transform(profile[cols]);rows=[]
for k in [2,3,4,5]:
    labels=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(X)
    if k==3: profile['cluster']=labels
    across=[];within=[]
    for i,j in combinations(range(len(X)),2):
        (within if labels[i]==labels[j] else across).append(np.linalg.norm(X[i]-X[j]))
    rows.append(dict(K=k,across_over_within=np.mean(across)/np.mean(within),cluster_sizes='-'.join(map(str,sorted(np.bincount(labels),reverse=True)))))
write_csv(pd.DataFrame(rows),'outputs/part_d_separation.csv')
order=profile.groupby('cluster').delay_rate.mean().sort_values().index
profile['segment']=profile.cluster.map(dict(zip(order,['Standard','Watch','Escalate'])))
segments=profile[['hub_id','segment']].merge(hubs[['hub_id','hub_city','hub_tier']],on='hub_id',validate='one_to_one')
write_csv(segments,'outputs/hub_segments.csv')
s=profile.groupby('segment').agg(hubs=('hub_id','size'),**{c:(c,'mean') for c in cols}).reindex(['Standard','Watch','Escalate']).reset_index()
write_csv(s,'outputs/segment_profile.csv')
write_csv(segments[segments.segment!=segments.hub_tier.map({1:'Standard',2:'Watch',3:'Escalate'})],'outputs/tier_vs_segment_mismatch.csv')
