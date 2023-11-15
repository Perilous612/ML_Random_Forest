from CSVHandler import CSVHandler
from DataFrameProcess import DataFrameProcess
from DataProess import DataProcess
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE

if __name__ == "__main__":
    
    #get csv to dataframe and remove empty rows 
    input_csv_path = "seth_test/train.csv"
    csv_handler = CSVHandler(input_csv_path)
    df = csv_handler.get_dataframe()
    TargetDF = df['label']

    #chategorize the feature into numerical, categorical and special attention (which has '|')
    dataframe_process = DataFrameProcess()
    label_name = 'label'
    df = dataframe_process.drop_column(df,label_name)
    numerical_feature_list, non_numerical_feature_list = dataframe_process.get_numerical_and_non_numerical_list(df)
    categorical_feature_list, special_attention_feature_list,useless_feature_list = dataframe_process.get_categorical_and_special_list(df,non_numerical_feature_list,50)

    #derive useable dataset frame for categorical feature label
    categorical_df = dataframe_process.get_useable_categorical_dataframe(df,categorical_feature_list)
    numerical_df = dataframe_process.get_useable_numerical_dataframe(df,numerical_feature_list)
    special_df = dataframe_process.get_useable_special_dataframe(df,special_attention_feature_list)

    categorical_df.to_csv('categorical_df.csv', index=False)
    numerical_df.to_csv('numerical_df.csv', index=False)
    special_df.to_csv('special_df.csv', index=False)
    
    #scale numerical_df for 
    data_process = DataProcess()
    numerical_df = data_process.scale_dataset(numerical_df)
    categorical_df.reset_index(drop=True, inplace=True)
    numerical_df.reset_index(drop=True, inplace=True)
    special_df.reset_index(drop=True, inplace=True)

    #Get the final dataframe after data has been process 
    finalDF = pd.concat([categorical_df,numerical_df,special_df],axis=1, ignore_index=False)

    X = finalDF
    Y = TargetDF

    #Split data set to traing to test
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=20)

    # Apply SMOTE for oversampling only on the training set to balance dataset for 1 and 0 
    smote = SMOTE(random_state=30)
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)










    cv_scores, Y_pred , important_feature = data_process.get_cv_score_y_predit_important_feature(X_train_resampled, Y_train_resampled,X_test)


    print('accuracy before droping less important feature: ',accuracy_score(Y_test,Y_pred))


    # Drop less relevant feature from the dataframe 
    feature_drop = 50
    X_final_train,X_final_test = data_process.remove_less_important_feature(important_feature,X_train_resampled,X_test,feature_drop )
    cv_scores, Y_pred , important_feature = data_process.get_cv_score_y_predit_important_feature(X_final_train, Y_train_resampled,X_final_test)


    print('accuracy after droping less important feature: ',accuracy_score(Y_test,Y_pred))
    







    # param_grid = {
    # 'n_estimators': [25, 50, 75, 100],
    # 'max_depth': [6, 10, 12, 15],
    # 'min_samples_split': [5, 10, 15 , 20],
    # 'max_features': [ 'log2']
    # }

    #  #  Create the grid search model
    # grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=99),
    #                         param_grid=param_grid,
    #                         cv=5,  # 5-fold cross-validation
    #                         scoring='accuracy', # or other appropriate scoring metric
    #                         verbose=3) 
    # # Fit the grid search to the data
    # grid_search.fit(X_train_resampled, Y_train_resampled)

    # # Get the best parameters
    # best_params = grid_search.best_params_
    # print("Best Hyperparameters:", best_params)