import pandas as pd
import yaml

class RemoveOutliersClass:
    def __init__(self, file_path: str, yaml_path: str):
        self.df = pd.read_csv(file_path)
        self.yaml_path = yaml_path
        
        self.cleaned_df = pd.DataFrame(columns=list(self.df.columns))

    def load_yaml(self):
        with open(self.yaml_path, 'r') as file:
            data = yaml.safe_load(file) 
            print(data['airlineName'])
            return data['airlineName']

    def remove_outliers(self,airlineName):
        print(airlineName)
        for airline, quartiles in airlineName.items():
            airDataset = self.df[self.df['Airline'] == airline]

            q1 = airDataset['Price'].quantile(quartiles[0])
            q3 = airDataset['Price'].quantile(quartiles[1])
            IQR = q3 - q1
            lowerLimit = q1 - IQR * 1.5
            upperLimit = q3 + IQR * 1.5

            lowerLimitIndex = airDataset[airDataset['Price'] <= lowerLimit].index
            upperLimitIndex = airDataset[airDataset['Price'] >= upperLimit].index

            if airDataset.shape[0] > 5:
                airDataset.drop(lowerLimitIndex, axis=0, inplace=True)
                airDataset.drop(upperLimitIndex, axis=0, inplace=True)

            self.cleaned_df = pd.concat([self.cleaned_df, airDataset], axis=0)
        return self.cleaned_df 
    
    def save_file(self, noOutlierDataFilePath):
        self.cleaned_df.to_csv(noOutlierDataFilePath, index=False)

if __name__ == "__main__":
    file_path = 'Data/02_CleanedData/CleanedData.csv'
    yaml_path = "constants.yaml"
    noOutlierDataFilePath = 'Data/03_noOutlierData/noOutlierDataFile.csv'

    RemoveOutliersObj = RemoveOutliersClass(file_path, yaml_path)
    airlineName = RemoveOutliersObj.load_yaml() 
    RemoveOutliersObj.remove_outliers(airlineName)
    RemoveOutliersObj.save_file(noOutlierDataFilePath)