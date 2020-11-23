###############
# IMPORT LIBS
###############

import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

################
# SET CONSTANTS
################
COLS_TO_DROP = ['target','card_id']
MODELS_PATH = 'models/'

################
# SET PARAMS
################

eval_strategy = 'kfold'
MODEL_NAME = 'lgb'
NSEED = 8
BASE_SEED = 1000
NFOLD = 5
PCA_COMPONENTS = 30

SEEDS = [948, 534, 432, 597, 103, 21, 2242, 17, 20, 29]

###############
# DEFINE FUNCS
###############

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


#####################
# PREPROCESSING FUNCS
#####################

def preprocess_data(data):
    for col in data.columns[2:]:
        if data[col].dtype == 'object':
            #data[col] = data[col].astype('category')
            #data[col] = data[col].fillna('NaN')
            le = joblib.load(MODELS_PATH + f'le_{col}.pkl', 'rb')
            featureDict = dict(zip(le.classes_, le.transform(le.classes_)))
            data[col] = data[col].apply(lambda x: featureDict.get(x, -1))


        # if data[col].dtype != 'object':
        #     data[col] = data[col].fillna(0)
            
    return data

def feature_generation(data):
    
    '''
    Generate new features
    '''
    
    df = data.copy()
    
    def diff_features(df, ft1, ft2):
        df['diff-{}-{}'.format(ft1, ft2)] = df[ft1] - df[ft2]
        return df

    def ratio_features(df, ft2, ft1):
        df['ratio-{}-{}'.format(ft1, ft2)] = df[ft1] / (df[ft2] + 1e-12)
        return df

    def match_features(df, ft1, ft2):
        df['match-{}-{}'.format(ft1, ft2)] = (df[ft1] == df[ft2]).astype(int)
        return df

    def common_regions(df, ft1, ft2):
        missed_regions = []
        used_regions = []
        for reg, count in df[ft1].value_counts().items():
            if count in df[ft2].value_counts().values:
                used_regions.append(reg)
            else:
                missed_regions.append(reg)
        df['common_regions-{}-{}'.format(ft1, ft2)] = df[ft1].apply(lambda x: int(x in used_regions))
        return df
    
    def comb_features(df, ft1, ft2):
        df['comb-{}-{}'.format(ft1, ft2)] = df[ft1].astype(str) + '-' + df[ft2].astype(str)
        return df
    
    
    ##### v01 #####

#     for col in [
#         'addr_region_fact_encoding1',
#         'addr_region_reg_encoding1',
#         'app_addr_region_reg_encoding1',
#         'app_addr_region_fact_encoding1',
#     ]:
#         df[col] = np.round(df[col] * 0.0083, 6).astype(int)
#     df['app_addr_region_sale_encoding1'] = np.round(df['app_addr_region_sale_encoding1'] * 0.0039).astype(int)
#     for col in [
#         'addr_region_fact_encoding2',
#         'addr_region_reg_encoding2',
#         'app_addr_region_reg_encoding2',
#         'app_addr_region_fact_encoding2',
#     ]:
#         df[col] = np.round(df[col] * 1.1).astype(int)
#     df['app_addr_region_sale_encoding2'] = np.round(df['app_addr_region_sale_encoding2'] * 0.007).astype(int)

#     df = diff_features(df, 'sas_limit_after_003_amt', 'sas_limit_last_amt')

#     df = match_features(df, 'sas_limit_after_003_amt', 'sas_limit_last_amt')

#     df = match_features(df, 'addr_region_fact_encoding1', 'addr_region_reg_encoding1')
#     df = match_features(df, 'addr_region_fact_encoding1', 'app_addr_region_reg_encoding1')
#     df = match_features(df, 'addr_region_fact_encoding1', 'app_addr_region_fact_encoding1')
#     df = match_features(df, 'addr_region_reg_encoding1', 'app_addr_region_reg_encoding1')
#     df = match_features(df, 'addr_region_reg_encoding1', 'app_addr_region_fact_encoding1')
#     df = match_features(df, 'app_addr_region_reg_encoding1', 'app_addr_region_fact_encoding1')

#     df = match_features(df, 'addr_region_fact_encoding2', 'addr_region_reg_encoding2')
#     df = match_features(df, 'addr_region_fact_encoding2', 'app_addr_region_reg_encoding2')
#     df = match_features(df, 'addr_region_fact_encoding2', 'app_addr_region_fact_encoding2')
#     df = match_features(df, 'addr_region_reg_encoding2', 'app_addr_region_reg_encoding2')
#     df = match_features(df, 'addr_region_reg_encoding2', 'app_addr_region_fact_encoding2')
#     df = match_features(df, 'app_addr_region_reg_encoding2', 'app_addr_region_fact_encoding2')

#     df = common_regions(df, 'addr_region_fact', 'addr_region_fact_encoding1')
#     df = common_regions(df, 'addr_region_reg', 'addr_region_reg_encoding1')
#     df = common_regions(df, 'app_addr_region_reg', 'app_addr_region_reg_encoding1')
#     df = common_regions(df, 'app_addr_region_fact', 'app_addr_region_fact_encoding1')
#     df = common_regions(df, 'app_addr_region_sale', 'app_addr_region_sale_encoding1')

    ##### v02 #####

#     df = diff_features(df, 'first_loan_date', 'last_loan_date')
#     df = ratio_features(df, 'first_loan_date', 'last_loan_date')
#     df = match_features(df, 'first_loan_date', 'last_loan_date')

#     df = diff_features(df, 'clnt_experience_cur_mnth', 'clnt_experience_total_mnth')

#     df = diff_features(df, 'ttl_inquiries', 'inquiry_1_week')
#     df = diff_features(df, 'inquiry_12_month', 'inquiry_1_week')
#     df = diff_features(df, 'inquiry_9_month', 'inquiry_1_week')
#     df = diff_features(df, 'inquiry_6_month', 'inquiry_1_week')
#     df = diff_features(df, 'inquiry_3_month', 'inquiry_1_week')
#     df = diff_features(df, 'ttl_inquiries', 'inquiry_3_month')
#     df = diff_features(df, 'inquiry_12_month', 'inquiry_3_month')
#     df = diff_features(df, 'inquiry_9_month', 'inquiry_3_month')
#     df = diff_features(df, 'inquiry_6_month', 'inquiry_3_month')
#     df = diff_features(df, 'ttl_inquiries', 'inquiry_6_month')
#     df = diff_features(df, 'inquiry_12_month', 'inquiry_6_month')
#     df = diff_features(df, 'inquiry_9_month', 'inquiry_6_month')
#     df = diff_features(df, 'ttl_inquiries', 'inquiry_9_month')
#     df = diff_features(df, 'inquiry_12_month', 'inquiry_9_month')
#     df = diff_features(df, 'ttl_inquiries', 'inquiry_12_month')

#     df = ratio_features(df, 'ttl_inquiries', 'inquiry_1_week')
#     df = ratio_features(df, 'inquiry_12_month', 'inquiry_1_week')
#     df = ratio_features(df, 'inquiry_9_month', 'inquiry_1_week')
#     df = ratio_features(df, 'inquiry_6_month', 'inquiry_1_week')
#     df = ratio_features(df, 'inquiry_3_month', 'inquiry_1_week')
#     df = ratio_features(df, 'ttl_inquiries', 'inquiry_3_month')
#     df = ratio_features(df, 'inquiry_12_month', 'inquiry_3_month')
#     df = ratio_features(df, 'inquiry_9_month', 'inquiry_3_month')
#     df = ratio_features(df, 'inquiry_6_month', 'inquiry_3_month')
#     df = ratio_features(df, 'ttl_inquiries', 'inquiry_6_month')
#     df = ratio_features(df, 'inquiry_12_month', 'inquiry_6_month')
#     df = ratio_features(df, 'inquiry_9_month', 'inquiry_6_month')
#     df = ratio_features(df, 'ttl_inquiries', 'inquiry_9_month')
#     df = ratio_features(df, 'inquiry_12_month', 'inquiry_9_month')
#     df = ratio_features(df, 'ttl_inquiries', 'inquiry_12_month')
    
#     ##### v03 #####

    df = comb_features(df, 'prt_name', 'channel_name_2')
    df = comb_features(df, 'prt_name', 'clnt_income_month_avg_net_amt') # 132 categories
    df = comb_features(df, 'prt_name', 'clnt_birth_year') # 503 categories
    df = comb_features(df, 'prt_name', 'inquiry_1_week') # 222 categories
    df = comb_features(df, 'prt_name', 'addr_region_fact') # 464 categories
    df = comb_features(df, 'prt_name', 'sas_limit_last_amt') # 142 categories
    df = comb_features(df, 'prt_name', 'clnt_speciality_sphere_name') # 250 categories
    df = comb_features(df, 'prt_name', 'addr_region_fact_encoding1') # 234 categories
    
    
    
    ########## v04 #############
#     # df['fe_col01'] = df.apply(lambda x: 1 if
#     #          (x['inquiry_1_week'] == x['inquiry_1_month']) & \
#     #          (x['inquiry_1_month'] == x['inquiry_3_month']) & \
#     #          (x['inquiry_3_month'] == x['inquiry_6_month']) & \
#     #          (x['inquiry_6_month'] == x['inquiry_9_month']) & \
#     #          (x['inquiry_9_month'] == x['inquiry_12_month'])
#     #         else 0, axis=1)
#     df['fe_feature_nans'] = sum([df[col].isna() for col in ['feature_10'] + ['feature_{}'.format(x) for x in range(12, 30)]])
#     df['fe_inquiry_mean_diff'] = df['inquiry_1_week'] / 7 + \
#             (df['inquiry_1_month'] - df['inquiry_1_week']) / 23 + \
#             (df['inquiry_3_month'] - df['inquiry_1_month']) / 60 + \
#             (df['inquiry_6_month'] - df['inquiry_3_month']) / 90 + \
#             (df['inquiry_9_month'] - df['inquiry_6_month']) / 90 + \
#             (df['inquiry_12_month'] - df['inquiry_9_month']) / 90
#     df['fe_inquiry_anomaly_day'] = df.apply(lambda x: 1 if
#                                             ((x['inquiry_recent_period'] <= 30) & (x['inquiry_1_month'] == 0)) | \
#                                             ((x['inquiry_recent_period'] <= 89) & (x['inquiry_3_month'] == 0)) | \
#                                             ((x['inquiry_recent_period'] <= 181) & (x['inquiry_6_month'] == 0)) | \
#                                             ((x['inquiry_recent_period'] <= 273) & (x['inquiry_9_month'] == 0)) | \
#                                             ((x['inquiry_recent_period'] <= 273) & (x['inquiry_12_month'] == 0)) | \
#                                             ((x['inquiry_recent_period'] <= 365) & (x['inquiry_12_month'] == 0))
#                                            else 0, axis=1)
#     df['fe_inquiry_mean_before_12_month'] = (df['ttl_inquiries'] - df['inquiry_12_month']) / (df['first_loan_date'] + 366)

#     df = diff_features(df, 'loans_main_borrower', 'loans_active')
#     df = match_features(df, 'addr_region_reg', 'addr_region_fact')

    # df = diff_features(df, 'inquiry_1_month', 'inquiry_1_week')
    # df = diff_features(df, 'inquiry_3_month', 'inquiry_1_month')
    # df = diff_features(df, 'inquiry_6_month', 'inquiry_3_month')
    # df = diff_features(df, 'inquiry_9_month', 'inquiry_6_month')
    # df = diff_features(df, 'inquiry_12_month', 'inquiry_9_month')
    # df = diff_features(df, 'ttl_inquiries', 'inquiry_12_month')

    return df

def add_pca(data):
    '''
    Add pca features to data
    '''

    pca = joblib.load(MODELS_PATH + f'pca.pkl')
    pcas = pca.transform(data.iloc[:, 2:])
    pcas = pd.DataFrame(pcas, columns=[f'pca_{i}' for i in range(PCA_COMPONENTS)])
    data_all = pd.concat([data, pcas], axis=1)
    
    return data_all

def postprocess_predictions(predictions, rate = 0.1):
    thresh = 0.049
    order = np.argsort(-predictions)  # -> 5%: лучшие (единицы) | 90% занулили (все что между) | 5% оставили (нули)
    
    predictions[order[int(thresh * len(predictions)):int((1 - rate + thresh) * len(predictions))]] = -1
    
    return predictions



####################
# PREDICTION FUNCS
####################


# kfold
def prediction_kfold_model(df):
    X_test = df.drop(columns = COLS_TO_DROP).values

    prediction = np.zeros(X_test.shape[0]) * 0.0

    for i_seed in range(NSEED):
        seed = SEEDS[i_seed] # grab those seeds from train
        seed_everything(seed)
        print('Seed: {}, {}/{}'.format(seed, i_seed + 1, NSEED))
        for i_fold in range(NFOLD):
        
            model = joblib.load(MODELS_PATH + f'{MODEL_NAME}_{i_fold}_{seed}.pkl')
            prediction += model.predict_proba(X_test)[:, 1] / (NSEED*NFOLD)
            print('  done.')
    
    prediction = postprocess_predictions(prediction)
    
    return prediction


# single model

def prediction_single_model(df):
    X_test = df.drop(columns = COLS_TO_DROP).values
    prediction = np.zeros(X_test.shape[0]) * 0.0
    
    # load model
    model = joblib.load(MODELS_PATH + f'{MODEL_NAME}_{0}_{0}.pkl')

    prediction = model.predict_proba(X_test)[:, 1]

    prediction = postprocess_predictions(prediction)
    return prediction

if __name__ == '__main__':

    # read test
    test = pd.read_csv("test.csv")

    # preprocess test
    test = feature_generation(test)
    test = preprocess_data(test)

    
    # obtain predictions
    prediction = test[["card_id", "target"]].copy(deep=True)
    if eval_strategy == 'kfold':
        prediction['target'] = prediction_kfold_model(test)
    elif eval_strategy == 'single':
        prediction['target'] = prediction_single_model(test)

    # save predictions
    prediction.to_csv("prediction.csv", index=False)