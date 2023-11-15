import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

class DataFrameProcess:
    
    def drop_column(self,dataframe, column_label):
        # Drop the column with the specified label
        df = dataframe.drop(column_label, axis=1)
        return df

    def get_numerical_and_non_numerical_list(self,dataframe):
        numerical_label_list = []
        non_numerical_label_list = []
        for column_label in dataframe:
            try:
                pd.to_numeric(dataframe[column_label])
                numerical_label_list.append(column_label)
            except (ValueError,TypeError):
                non_numerical_label_list.append(column_label)
                continue
            
        return numerical_label_list,non_numerical_label_list


    def get_categorical_and_special_list(self,dataframe,non_numerical_label_list,threshold):
        special_attention_feature_list = []
        categorical_feature_list = []
        useless_feature_list = []
        for label in non_numerical_label_list:
            try:
                result = dataframe[label].str.contains('\|').any()
                if result:
                    special_attention_feature_list.append(label)
                else:
              
                    if dataframe[label].nunique() <= threshold:
                        categorical_feature_list.append(label)
                    else:
                        useless_feature_list.append(label)
                    
            except (ValueError,TypeError):
                continue
        return categorical_feature_list , special_attention_feature_list, useless_feature_list
    
    def get_useable_categorical_dataframe(self,dataframe,categorical_feature_list):
        categorical_df = pd.DataFrame()
        for label in categorical_feature_list:
            
            df_encoded = pd.get_dummies(dataframe[label], columns=[label],prefix=label)
            categorical_df = pd.concat([categorical_df,df_encoded.astype(int)],axis=1)
        return categorical_df
    


    def get_useable_numerical_dataframe(self,dataframe,numerical_feature_list):
        numerical_df = pd.DataFrame()
        for label in numerical_feature_list:
            numerical_df = pd.concat([numerical_df,dataframe[label]],axis=1)
        return numerical_df
    


    def get_useable_special_dataframe(self,dataframe,special_attention_feature_list):
        special_numerical_feature = []
        special_categorical_feature = []
        special_numerical_df = pd.DataFrame()
        special_categorical_df = pd.DataFrame()

        mlb = MultiLabelBinarizer()


        for label in special_attention_feature_list:
            value = dataframe[label].str.split('|').str[1][0]
            try:
                int(value)
                special_numerical_feature.append(label)
            except ValueError:
                special_categorical_feature.append(label)
        for label in special_numerical_feature:
            # Split the 'feature' column by '|', convert to integers, and create separate columns
            split_features = dataframe[label].str.split('|').apply(lambda x: pd.Series(x, dtype=int))
            # Determine the maximum number of variables
            max_vars = split_features.shape[1]
            # Rename the columns to 'var1', 'var2', ..., 'varN'
            split_features.columns = [f'{label}var{i}' for i in range(1, max_vars + 1)]
            special_numerical_df =  pd.concat([special_numerical_df,split_features],axis=1)


        # for label in special_categorical_feature:
        #     # Replace NaN values with an empty string
        #     dataframe[label] = dataframe[label].fillna('')
        #     # Convert the integers to strings, split by '|', and create a list of integers
        #     dataframe[label] = dataframe[label].apply(lambda x: x.split('|'))
        #     encoded_features = mlb.fit_transform(dataframe[label])
        #     print(label , encoded_features.shape)
        #     # Create a new DataFrame with the encoded features
        #     encoded_df = pd.DataFrame(encoded_features, columns=[str(i) for i in mlb.classes_])
            

        return special_numerical_df 
    


