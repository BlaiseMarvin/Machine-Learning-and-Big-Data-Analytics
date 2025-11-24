import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def fetch_timeseries_data():
    path = '/Users/blaise/Documents/ML/Machine-Learning-and-Big-Data-Analytics/data/CTA_-_Ridership_-_Daily_Boarding_Totals_20251110.csv'
    df = pd.read_csv(path, parse_dates=['service_date'])
    df.columns = ['date', 'day_type','bus', 'rail', 'total']
    df = df.sort_values(by='date').set_index('date')
    df = df.drop('total', axis=1)
    df = df.drop_duplicates()
    df['bus'] = df['bus'].str.replace(',','')
    df['rail'] = df['rail'].str.replace(',','')
    df['bus'] = df['bus'].astype(np.int64)
    df['rail'] = df['rail'].astype(np.int64)
    return df


def create_splits(df, attr='rail', train_ran=['2016-01','2018-12'],val_ran=['2019-01','2019-05'],test_ran=['2019-06']):
    rail_train = df[attr][train_ran[0]:train_ran[1]]/1e6
    rail_valid = df[attr][val_ran[0]:val_ran[1]]/1e6
    rail_test = df[attr][test_ran[0]:]/1e6
    return rail_train, rail_valid, rail_test

def create_oned_chunks(seq_length, ds):
    chunk_size = seq_length+1
    return [ds.values.tolist()[i:i+chunk_size] for i in range(len(ds.values.tolist())-chunk_size+1)]

def create_multidim_chunks2(seq_length, df, ts):
    chunk_size = seq_length+ts
    return [df.iloc[i:i+chunk_size].values for i in range(df.shape[0]-chunk_size+1)]

def create_splits_mulvar(df, train_ran=['2016-01','2018-12'],val_ran=['2019-01','2019-05'],test_ran=['2019-06']):
    df_mulvar = (df[['bus','rail']]/1e6).copy()
    df_mulvar['next_day_type'] = df['day_type'].shift(-1)
    mulvar_train = df_mulvar.iloc[:-1,:][train_ran[0]:train_ran[1]]
    mulvar_valid = df_mulvar.iloc[:-1,:][val_ran[0]:val_ran[1]]
    mulvar_test =  df_mulvar.iloc[:-1,:][test_ran[0]:]
    return mulvar_train, mulvar_valid, mulvar_test

def train_encoder(df):
    onehotencoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    onehotencoder.fit(df[['next_day_type']])
    return onehotencoder

def transform_split(df, encoder):
    df = df.copy()
    inter_df = encoder.transform(df[['next_day_type']])
    inter_df = pd.DataFrame(
        inter_df,
        columns=list(encoder.get_feature_names_out()),
        index=df.index
    )
    df.loc[:,'next_day_type_A'] = inter_df['next_day_type_A']
    df.loc[:, 'next_day_type_U'] = inter_df['next_day_type_U']
    df.loc[:, 'next_day_type_W'] = inter_df['next_day_type_W']
    df = df.reindex(columns=['bus','rail','next_day_type_A','next_day_type_U','next_day_type_W'])
    return df