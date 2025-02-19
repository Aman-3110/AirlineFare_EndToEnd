import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

class EncodingAndScalingClass:
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path)
        self.df = self.df.drop('Unnamed: 0', axis=1, errors='ignore')
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pipe = self.encoding_and_scaling()

    def read_file(self):
        self.df = self.df.drop('Unnamed: 0', axis=1, errors='ignore')
        return self.df
    
    def split_df_to_X_y(self):
        x = self.df.drop(['Price'], axis=1)
        y = self.df['Price']
        return x, y
    
    def train_test_split(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def encoding_and_scaling(self):
        trf1 = ColumnTransformer([
            ('OneHot', OneHotEncoder(drop='first', handle_unknown='ignore'),[0,1,2])
        ], remainder='passthrough')

        trf2 = ColumnTransformer([
            ('Ordinal', OrdinalEncoder(categories=[['non-stop', '1 stop', '2 stops', '3 stops', '4 stops']]),[16])
        ], remainder='passthrough')

        trf3 = ColumnTransformer([
            ('scale', StandardScaler(), slice(25))
        ])

        pipe = make_pipeline(trf1, trf2, trf3)
        return pipe
                        
    def save_X_train(self):
        x_train_transformed = self.pipe.fit_transform(self.X_train)
        x_train_transformed = pd.DataFrame(x_train_transformed)
        x_train_transformed.to_csv('Data/04_Encoded_Data/X_train.csv', index=False)

    def save_X_test(self):
        x_test_transformed = self.pipe.transform(self.X_test)
        x_test_transformed = pd.DataFrame(x_test_transformed)
        x_test_transformed.to_csv('Data/04_Encoded_Data/X_test.csv', index=False)

    def save_y_train(self):
        self.y_train.to_csv('Data/04_Encoded_Data/y_train.csv', index=False)

    def save_y_test(self):
        self.y_test.to_csv('Data/04_Encoded_Data/y_test.csv', index=False)


if __name__ == "__main__":
    file_path = 'Data/03_noOutlierData/noOutlierDataFile.csv'
    encoding_and_scaling_obj = EncodingAndScalingClass(file_path)

    encoding_and_scaling_obj.read_file()

    x, y = encoding_and_scaling_obj.split_df_to_X_y()

    encoding_and_scaling_obj.train_test_split(x, y)

    encoding_and_scaling_obj.save_X_train()
    encoding_and_scaling_obj.save_X_test()
    encoding_and_scaling_obj.save_y_train()
    encoding_and_scaling_obj.save_y_test() 

    




    