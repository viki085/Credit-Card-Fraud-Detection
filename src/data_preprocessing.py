import pandas as pd


def data_preprocessing(df):

    # Add flag column for missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col+"_missing_flag"] = df[col].isnull()

    df[col+"_missing_flag"] = [df[col].isnull() \
                            for col in df.columns \
                            if df[col].isna().sum() > 0]

    # Drop the columns where one category contains more than 90% values
    drop_cols = {}
    for col in df.columns:
        missing_share = df[col].isnull().sum()/df.shape[0]
        if missing_share > 0.9:
            drop_cols.add(col)

    # Drop the columns which have only one unique value    
    for col in good_cols:
        unique_value = df[col].nunique()
        if unique_value == 1:
            drop_cols.add(col)

    good_cols = [col for col in df.columns if col not in drop_cols]

    return df[good_cols]

if __name__ == '___main__':

    print("Starting Preprocessing")
    df_id = pd.read_csv("input/train_identity.csv")
    df_tran = pd.read_csv("input/train_transaction.csv")
    df = df_tran.merge(df_id, how='left', on='TransactionID')
    data = data_preprocessing(df)

    print("Preprocessing Successfull!!!")
