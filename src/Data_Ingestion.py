import os
import sys
import pandas as pd
from pathlib import Path
from Common_Utils.sqlite_handler import SQLiteStrategy
from Common_Utils.file_reader import FileReaderManager
from Common_Utils import CustomException, track_performance, setup_logger,load_yaml

# Setup logger
logger = setup_logger(filename="logs")
config = load_yaml("Config_Yaml/config_path.yaml")
schema_config = load_yaml("Config_Yaml/schema.yaml")


"""
db_path (Path): SQLite DB path.
raw_data_dir (Path): Directory to store raw CSV.
zip_path (Path): path to store downloaded zip.
"""
db_path: Path = Path(config["DataIngestion"]["path"]["db_path"])
extracted_data_dir: Path = Path(config["DataIngestion"]["path"]["extracted_data_dir"])
zip_path: Path = Path(config["DataIngestion"]["path"]["zip_path"])



class DataIngestion:
    """
    Handles ingestion of external datasets into local SQLite database and raw storage.
    """

    def __init__(self):
        """
        Initialize data ingestion pipeline with required paths.

        Args:
            
            db_path (str): SQLite DB path.
            extracted_data_dir (str): Directory to store raw CSV.
            zip_dir (str): Directory to store downloaded zip.
           
        """
        self.db_path = db_path
        self.zip_path = zip_path
        self.extracted_data_dir = extracted_data_dir


    def upload_to_db(self, df: pd.DataFrame, table_name: str = "raw_data"):
        """
        Uploads a DataFrame to a SQLite table.

        Args:
            df (pd.DataFrame): The data to upload.
            table_name (str): Table name (default = "raw_data").
        """
        try:
            db = SQLiteStrategy(self.db_path)
            db.connect()

            # Optionally drop existing table or append
            if db.table_exists(table_name):
                logger.info(f"Table '{table_name}' exists. Overwriting...")
                db.write_df(df, table_name)
            else:
                logger.info(f"Creating and writing to new table '{table_name}'...")
                db.write_df(df, table_name)

            db.close()

        except Exception as e:
            logger.error(f"Failed to upload data to table '{table_name}'")
            raise CustomException(e, sys)



    @track_performance
    def ingest(self):
        """
        Full ingestion pipeline: download, extract, filter columns by schema, save to DB and CSV.
        """
        try:
            # Step 1: Read dataset using FileReaderManager
            logger.info("Starting data ingestion...")
            df = FileReaderManager.load_dataset(self.zip_path, self.extracted_data_dir)
            print(df.columns)
            if df.empty:
                raise ValueError("Loaded DataFrame is empty.")

            # Step 2: Load schema.yaml and filter columns
            expected_columns = list(schema_config["columns"].keys())
            logger.info(f"Filtering DataFrame columns to match schema: {expected_columns}")
            
            df_filtered = df[expected_columns]

            # Step 3: Save cleaned CSV to extracted_data_dir as 'bank_churners.csv'
            output_csv_path = self.extracted_data_dir / "bank_churners.csv"
            df_filtered.to_csv(output_csv_path, index=False)
            logger.info(f"Filtered DataFrame saved to: {output_csv_path}")

            # Step 4: Upload to SQLite
            self.upload_to_db(df_filtered)

            logger.info("Data ingestion completed successfully.")

        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise CustomException(e, sys)


@track_performance
def execute_data_ingestion():
    try:

        logger.info(f"Starting Data_Ingestion process..")
        context = DataIngestion()
        context.ingest()
        logger.info(f"Data_Ingestion completed successfully")
    
    except CustomException as e :
        logger.error(f"Data_Ingestion process failed : {e}")


if __name__ == "__main__":
    execute_data_ingestion()