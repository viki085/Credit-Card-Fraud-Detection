import pandas as pd
from sklearn import model_selection



class CrossValidationSplit:


    def __init__(self, data, target, shuffle=True, num_folds=5, train_test_cv_split=False, random_state=42) -> None:
        
        self.data = data
        self.target = target
        self.shuffle = shuffle
        self.num_folds = num_folds
        self.train_test_cv_split = train_test_cv_split
        self.random_state = random_state
        
        if self.shuffle is True:
            self.data = self.data.sample(frac=1).reset_index(drop=True)


    def K_fold_split(self):

        self.data["kfold"] = -1

        kf = model_selection.StratifiedKFold(n_splits=self.num_folds, 
                                             shuffle=False, 
                                             )

        for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.data, y=self.data[self.target].values)):
            self.data.loc[val_idx, "kfold"] = fold

        return self.data


    def train_test_cv_split(self):

        X = self.data.drop(self.target)
        y = self.data[self.target]

        X_train, x_test, y_train, y_test = model_selection.train_test_split(X,y,
                                                                            test_size=0.15,
                                                                            stratify=self.data[self.target].values,
                                                                            random_state=self.random_state)

        x_train, x_cv, y_train, y_cv = model_selection.train_test_split(X_train,y_train,
                                                                            test_size=0.15,
                                                                            stratify=self.data[self.target].values,
                                                                            random_state=self.random_state)

        return x_train, y_train, x_cv, y_cv, x_test, y_test


if __name__ == "__main__":

    df_id = pd.read_csv("input/train_identity.csv")
    df_tran = pd.read_csv("input/train_transaction.csv")
    df = df_tran.merge(df_id, how='left', on='TransactionID')
    print(df.head())
    cv = CrossValidationSplit(df, ['isFraud'], shuffle = True, num_folds = 5, train_test_cv_split = False, random_state = 42)
    df_split = cv.K_fold_split()
    print(df_split.head())
    print(df_split.kfold.value_counts())

                                                                            

    

