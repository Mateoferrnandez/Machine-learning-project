import logging 
import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd

# Setup logging configuration
#logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#Define an abstract method class for data ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path:str) -> pd.DataFrame:
        """ Abstract method to ingest data from a given file """
        pass

#Implement a concret class for zip ingestion
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path:str) -> pd.DataFrame:
        """ Extracts a zip file and returns the content as a pandas dataframe """
        #Ensure the file is a .zip
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")

        # Extract the zip file
        with zipfile.ZipFile(file_path,"r") as zip_ref:
            zip_ref.extractall("extracted_data")
        # Find the extracted CSV file (assuming theres one inside the zip) 
        extracted_files=os.listdir("extracted_data")
        csv_files=[f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found. Please specify which one to use.")
        
        #read the csv into a dataframe
        csv_file_path=os.path.join("extracted_data",csv_files[0])
        df=pd.read_csv(csv_file_path)
        #Return the Dataframe
        return df
    
class CsvDataIngestor(DataIngestor):
    def ingest(self, file_path:str) -> pd.DataFrame:

        #Ensure the file is a .csv
        if not file_path.endswith(".csv"):
            raise ValueError("The provided file is not a .csv file")
        #read the csv into a dataframe
        df=pd.read_csv(file_path,sep=";")
        #df=df.drop(columns="CANTIDADES_ESTIMADAS")
        logging.info(f"df columns:{df.head()})")
        return df
  

#Implement a factory to create data ingestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension:str) -> DataIngestor:
        """Returns the appropiate DataIngestor base on the file extension."""
        if file_extension == ".zip":
            return ZipDataIngestor()
        elif file_extension ==".csv":
            return CsvDataIngestor()
        else:
            raise ValueError(f" No ingestor available for file extension:{file_extension}")
            
#Example of usage

if __name__ == "__main__":
    # # specify the file path
    #file path = example.zip

    ##Determine the file extension
    #file_extension = os.path.splitext(filepath)[1]

    ##Get the appropiate DataIngestor
    #data_ingestor=DataIngestorFactory.get_data_ingestor(file_extension)

    ## Ingest the data and load it into a dataframe
    # df = data_ingestor.ingest(file_path)

    ## Now df contains the dataframe from the extracted csv
    # print(df.head())
    pass