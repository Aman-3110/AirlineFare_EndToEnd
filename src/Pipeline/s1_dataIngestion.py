# import pandas as pd

# class DataIngestionClass:
#     def __init__(self, source_path:str, destination_path:str):
#         self.source_path = source_path
#         self.destination_path = destination_path

#     def read_csv(self):
#         df = pd.read_csv(self.source_path)
#         print(df)
#         return df
    
#     def save_csv(self, df):
#         df.to_csv(self.destination_path, index=False)

# if __name__ == "__main__":
#     source_path = 'E:\\Codes\\Juypter Notebook\\The XL Academy\\ML\\Airline.csv'
#     destination_path = 'Data\\01_RawData\\Airline.csv'
#     file_handler = DataIngestionClass(source_path, destination_path)
#     df = file_handler.read_csv()
#     file_handler.save_csv(df)


import os
import pandas as pd

class DataIngestionClass:
    def read_csv(source_path):
        # Read the CSV file and return the DataFrame
        df = pd.read_csv(source_path)
        print(df)
        return df

    def save_file(df, directory, filename):
        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' was created.")
        else:
            print(f"Directory '{directory}' already exists.")

        # Construct the file path
        file_path = os.path.join(directory, filename)
        
        # Save the DataFrame to the file
        df.to_csv(file_path, index=False)  # index=False to avoid writing row indices
        print(f"File has been saved to {file_path}")

# This block will only execute if this script is run directly
if __name__ == "__main__":
    source_path = 'E:\\Codes\\Juypter Notebook\\The XL Academy\\ML\\Airline.csv'
    directory = "Data/01_RawData/"
    filename = "Airline.csv"

    df = DataIngestionClass.read_csv(source_path)  # Read the CSV file
    DataIngestionClass.save_file(df, directory, filename)  # Save the DataFrame to the destination