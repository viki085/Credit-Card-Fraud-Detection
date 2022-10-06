
import pandas as pd
import numpy as np
import datetime
from  sklearn.preprocessing import minmax_scale

def feature_generator(df):

    # Create Date features
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    df["Date"] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

    df['_Weekdays'] = df['Date'].dt.dayofweek
    df['_Hours'] = df['Date'].dt.hour
    df['_Days'] = df['Date'].dt.day

    # Generalize feature values for P_emaildomain
    df.loc[df['P_emaildomain'].isin(['gmail.com', 'gmail']),'P_emaildomain'] = 'Google'
    df.loc[df['P_emaildomain'].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk',
                                    'yahoo.co.jp', 'yahoo.de', 'yahoo.fr',
                                    'yahoo.es']), 'P_emaildomain'] = 'Yahoo Mail'
    df.loc[df['P_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 
                                    'hotmail.es','hotmail.co.uk', 'hotmail.de',
                                    'outlook.es', 'live.com', 'live.fr',
                                    'hotmail.fr']), 'P_emaildomain'] = 'Microsoft'
    df.loc[df.P_emaildomain.isin(df.P_emaildomain\
                                .value_counts()[df.P_emaildomain.value_counts() <= 500 ]\
                                .index), 'P_emaildomain'] = "Others"
    df.P_emaildomain.fillna("NoInf", inplace=True)

    # Generalize feature values for R_emaildomain
    df.loc[df['R_emaildomain'].isin(['gmail.com', 'gmail']),'R_emaildomain'] = 'Google'
    df.loc[df['R_emaildomain'].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk',
                                    'yahoo.co.jp', 'yahoo.de', 'yahoo.fr',
                                    'yahoo.es']), 'R_emaildomain'] = 'Yahoo Mail'
    df.loc[df['R_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx','hotmail.es', 
                                    'hotmail.co.uk', 'hotmail.de','outlook.es','live.com', 'live.fr',
                                    'hotmail.fr']), 'R_emaildomain'] = 'Microsoft'
    df.loc[df.R_emaildomain.isin(df.R_emaildomain\
                                .value_counts()[df.R_emaildomain.value_counts() <= 300 ]\
                                .index), 'R_emaildomain'] = "Others"
    df.R_emaildomain.fillna("NoInf", inplace=True)





    ################################# Domain Specific Features ############################################
    # Transaction amount minus mean of transaction 
    df['Trans_min_mean'] = df['TransactionAmt'] - np.nanmean(df['TransactionAmt'],dtype="float64")
    df['Trans_min_std']  = df['Trans_min_mean'] / np.nanstd(df['TransactionAmt'].astype("float64"),dtype="float64")

    # Features for transaction amount and card 
    df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_mean_card4'] = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_std_card1']  = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('std')
    df['TransactionAmt_to_std_card4']  = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('std')

    # initialize function to perform PCA
    def perform_PCA(df, cols, n_components, prefix='PCA_', rand_seed=4):
        pca = PCA(n_components=n_components, random_state=rand_seed)
        principalComponents = pca.fit_transform(df[cols])
        principalDf = pd.DataFrame(principalComponents)
        df.drop(cols, axis=1, inplace=True)

        principalDf.rename(columns=lambda x: str(prefix)+str(x), inplace=True)
        df = pd.concat([df, principalDf], axis=1)
        return df

    # Fill na values and scale V columns

    # Columns starting from V1 to V339
    filter_col = df.columns[53:392] 
    #filter_col = col for col in df.columns if col.str.startswith('V')
    for col in filter_col:
        df[col] = df[col].fillna((df[col].min() - 2))
        df[col] = (minmax_scale(df[col], feature_range=(0,1)))

    # Perform PCA    
    df = perform_PCA(df, filter_col, prefix='PCA_V_', n_components=30)