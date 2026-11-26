"""Shared preprocessing for training and scoring."""
import numpy as np
import pandas as pd
PROMISED_HOURS = {'Express':36,'Standard':60,'Economy':96}
FEATURES = ['promised_hours','distance_km','parcel_weight_kg','log_declared_value','n_stops','rain_flag','sale_event_flag','is_cod','is_weekend','dispatch_hour','hub_tier','hub_daily_capacity','label_print_batch','scanner_build_no','tape_roll_no','sticker_serial_mod','shelf_slot_no']
NOISE_COLUMNS = FEATURES[-5:]

def clean_shipments(raw, medians=None):
    df=raw.drop_duplicates().copy()
    df['service_type']=df.service_type.str.strip().str.title()
    df['payment_mode']=df.payment_mode.fillna('Unknown')
    df['declared_value_inr']=pd.to_numeric(df.pop('declared_value').str.replace(r'[₹,]','',regex=True))
    df['dispatch_time']=pd.to_datetime(df.dispatch_time)
    df.loc[df.parcel_weight_kg>100,'parcel_weight_kg']=np.nan
    if medians is None:
        medians={c:float(df[c].median()) for c in ['distance_km','parcel_weight_kg','declared_value_inr']}
    return df.fillna(medians),medians

def build_features(clean,hubs):
    df=clean.merge(hubs,on='hub_id',how='left',validate='many_to_one')
    df['promised_hours']=df.service_type.map(PROMISED_HOURS).astype(int)
    df['is_cod']=df.payment_mode.eq('COD').astype(int)
    df['is_weekend']=df.dispatch_time.dt.dayofweek.ge(5).astype(int)
    df['dispatch_hour']=df.dispatch_time.dt.hour.astype(int)
    df['log_declared_value']=np.log(df.declared_value_inr)
    return df

def write_csv(df,path,index=False):
    df.round(4).to_csv(path,index=index)
