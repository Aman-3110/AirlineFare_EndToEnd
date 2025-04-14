import pandas as pd 
import os
from src.Utils.exception import CustomException
import sys

class DataCleaningClass:
    def __init__(self):
        pass

    def standardize_datetime(value):
        if len(value.split()) == 1:  # Time-only string
            return f"01-01-2000 {value}"  # Add a placeholder date
        return value

    def read_csv(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        return df
    

    def read_csv_as_dataframe(self,raw_file_dir,raw_file) -> pd.DataFrame:

        # Combine the folder and file path into a full path
        # full_path = os.path.join(folder, file)
        
        file_path = os.path.join(raw_file_dir,raw_file)

        print(file_path)
        # Read the CSV file into a DataFrame
        try:
            df = pd.read_csv(file_path)
            
            return df
        
        except Exception as e:
            print(CustomException(e,sys))
            return None
        except Exception as e:
            print(CustomException(e,sys))
            return None
        

    
    def clean_total_stops(self, df: pd.DataFrame) -> pd.DataFrame:
        mode_of_total_stops = df['Total_Stops'].mode()[0]
        df['Total_Stops'].fillna(mode_of_total_stops, inplace=True)
        print("Missing Values in 'Total_Stops' filled with mode.")
        return df
    
    def clean_airline_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Airline'].replace("Multiple carriers Premium economy", "Multiple carriers", inplace=True)
        df['Airline'].replace("Jet Airways Business", "Jet Airways", inplace=True)
        df['Airline'].replace("Vistara Premium economy", "Vistara", inplace=True)    
        print("Cleaned Airlines Names.")
        return df
    
    def clean_destination_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Destination'].replace("New Delhi", "Delhi", inplace=True)
        print("Cleaned Destination Names.")
        return df
        
    def create_duration_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df["hoursMinutes"] = 0
        for i in df.index:
            if " " in df.loc[i, "Duration"]:
                column1 = df.loc[i, "Duration"].split(" ")[0]
                column2 = df.loc[i, "Duration"].split(" ")[1]

                if "h" in column1:
                    column1 = (int(column1.replace("h", "")) * 60)
                elif "m" in column1:
                    column1 = (int(column1.replace("m", "")))

                if "h" in column2:
                    column2 = (int(column2.replace("h", "")) * 60)
                elif "m" in column2:
                    column2 = (int(column2.replace("m", "")))

                df.loc[i, "hoursMinutes"] = column1 + column2
            else:
                column1 = df.loc[i, "Duration"]

                if "h" in column1:
                    column1 = (int(column1.replace("h", "")) * 60)
                elif "m" in column1:
                    column1 = (int(column1.replace("m", "")))

                df.loc[i, "hoursMinutes"] = column1

        print("'hoursMinutes' column created.")
        return df
    
    def process_date_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Day'] = pd.to_datetime(df['Date_of_Journey'], format="%d-%m-%Y").dt.day
        df['Month'] = pd.to_datetime(df['Date_of_Journey'], format="%d-%m-%Y").dt.month
        df['Year'] = pd.to_datetime(df['Date_of_Journey'], format="%d-%m-%Y").dt.year

        df['Dept_Hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
        df['Dept_Minute'] = pd.to_datetime(df['Dep_Time']).dt.minute

        df['Arrival_Time'] = df['Arrival_Time'].apply(DataCleaningClass.standardize_datetime)

        df['Arr_Hour'] = pd.to_datetime(df['Arrival_Time'], format="%d-%m-%Y %H:%M").dt.hour
        df['Arr_Minute'] = pd.to_datetime(df['Arrival_Time']).dt.minute

        print("Date and Time Columns Processed.")
        return df
    
    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration', 'Route', 'Additional_Info'], axis=1)
        print("Unnecessary Columns Dropped.")
        return df
    
    def reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[['Airline', 'Source', 'Destination', 'Total_Stops', 'Day', 'Month', 'Year', 'Dept_Hour', 'Dept_Minute', 'Arr_Hour', 'Arr_Minute', 'hoursMinutes', 'Price']]
        print("Columns Reordered.")
        return df
    
    def save_file(self, df: pd.DataFrame, directory: str, filename: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' was created.")
        else:
            print(f"Directory '{directory}' already exists.")
        
        file_path = os.path.join(directory, filename)
        df.to_csv(file_path, index=False)
        print(f"Cleaned Data Saved at {file_path}")


if __name__ == "__main__":
    #raw_file_path = 'Data/01_RawData/Airline.csv'
    #directory = "Data/02_CleanedData/"
    #filename = "CleanedData.csv"

    raw_file_dir = "./Data/01_RawData/"
    raw_file = "Airline.csv"
    directory = "./Data/02_CleanedData/"
    filename = "CleanedData.csv"


    data_cleaning_obj = DataCleaningClass()
    df = data_cleaning_obj.read_csv_as_dataframe(raw_file_dir,raw_file)

    #df = data_cleaning_obj.read_csv(raw_file_path)

    df = data_cleaning_obj.clean_total_stops(df)
    df = data_cleaning_obj.clean_airline_column(df)
    df = data_cleaning_obj.clean_destination_column(df)
    df = data_cleaning_obj.create_duration_column(df)
    df = data_cleaning_obj.process_date_time_columns(df)
    df = data_cleaning_obj.drop_unnecessary_columns(df)
    df = data_cleaning_obj.reorder_columns(df)

    data_cleaning_obj.save_file(df, directory, filename)    
