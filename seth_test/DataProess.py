from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from DataFrameProcess import DataFrameProcess
import pandas as pd

class DataProcess:

    def scale_dataset(self, dataframe):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(dataframe)
        dataframe = pd.DataFrame(scaled_data, columns=dataframe.columns)
        return dataframe
    
    def get_cv_score_y_predit_important_feature(self,X_train,Y_train,X_test):

        clf = RandomForestClassifier(n_estimators=100,max_depth= 15, min_samples_split= 5, max_features= 'log2', random_state=99)
        # Perform cross-validation on the resampled data
        cv_scores = cross_val_score(clf, X_train, Y_train, cv=5)
        clf.fit(X_train, Y_train)   
        Y_pred = clf.predict(X_test)

        return cv_scores,Y_pred, clf.feature_importances_
    

    def remove_less_important_feature(self,important_feature,X_train,X_test,feature_drop = 5):
        dataframe_process = DataFrameProcess()
         # Get feature importances
        importances = important_feature
        # Get corresponding feature labels
        feature_labels = X_train
        # Pair feature importances with their labels
        feature_importances = dict(zip(feature_labels, importances))

        # Sort features based on importances in ascending order
        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1])

        X_train_final = X_train
        X_test_final = X_test
        for feature, importance in sorted_features:
            if feature_drop > 0:
                feature_drop -=1
                X_train_final = dataframe_process.drop_column(X_train_final ,feature)
                X_test_final = dataframe_process.drop_column(X_test_final ,feature)
        
        return X_train_final,X_test_final            