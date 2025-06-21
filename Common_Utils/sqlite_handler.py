import os
import sys
import sqlite3
import pandas as pd
from typing import Optional
from pathlib import Path
from Common_Utils import CustomException, track_performance, setup_logger,load_yaml

# Setup logger
logger = setup_logger(filename="logs")
config = load_yaml("Config_Yaml/config_path.yaml")


class SQLiteStrategy:
    """
    Strategy class to encapsulate SQLite operations like connect, read, write, etc.
    """

    def __init__(self, db_path: str):
        """
        Initialize the strategy with a SQLite database path.

        Args:
            db_path (str): Path to the SQLite DB file.
        """
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None

    def connect(self):
        """
        Establishes a connection to the SQLite database.
        """
        try:
            if not self.connection:
                self.connection = sqlite3.connect(self.db_path)
                logger.info(f"Connected to SQLite DB at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to DB: {e}")
            raise CustomException(e, sys)

    def close(self):
        """
        Closes the connection to the SQLite database.
        """
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                logger.info(f"Closed connection to SQLite DB.")
        except Exception as e:
            logger.warning(f"Failed to close DB: {e}")
            raise CustomException(e, sys)

    def read_table_from_db(db_path: str, table_name: str) -> pd.DataFrame:
        """
        Reads an entire table from a SQLite database and returns it as a DataFrame.

        Args:
            db_path (str): Path to the SQLite database file.
            table_name (str): Name of the table to read.

        Returns:
            pd.DataFrame: Table contents.
        """
        try:
            db_path = Path(db_path)
            if not db_path.exists():
                raise FileNotFoundError(f"Database file not found at: {db_path}")
            
            conn = sqlite3.connect(str(db_path))
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            conn.close()

            logger.info(f"Read table '{table_name}' from '{db_path}'. Rows fetched: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"Failed to read table '{table_name}' from '{db_path}': {e}")
            raise CustomException(e, sys)

    def read_query(self, query: str) -> pd.DataFrame:
        """
        Executes a read query and returns a DataFrame.

        Args:
            query (str): SQL SELECT query.

        Returns:
            pd.DataFrame: Query result.
        """
        try:
            if not self.connection:
                self.connect()
            df = pd.read_sql_query(query, self.connection)
            logger.info(f"Executed read query. Rows fetched: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"Read query failed: {e}")
            raise CustomException(e, sys)

    def write_df(self, df: pd.DataFrame, table_name: str):
        """
        Writes a DataFrame to a table, replacing it.

        Args:
            df (pd.DataFrame): Data to write.
            table_name (str): Name of the target table.
        """
        try:
            if not self.connection:
                self.connect()
            df.to_sql(table_name, self.connection, if_exists="replace", index=False)
            logger.info(f"Wrote DataFrame to table '{table_name}' (overwrite). Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to write DataFrame to table '{table_name}': {e}")
            raise CustomException(e, sys)

    def append_df(self, df: pd.DataFrame, table_name: str):
        """
        Appends a DataFrame to an existing table.

        Args:
            df (pd.DataFrame): Data to append.
            table_name (str): Target table.
        """
        try:
            if not self.connection:
                self.connect()
            df.to_sql(table_name, self.connection, if_exists="append", index=False)
            logger.info(f"Appended DataFrame to table '{table_name}'. Rows added: {len(df)}")
        except Exception as e:
            logger.error(f"Failed to append DataFrame to table '{table_name}': {e}")
            raise CustomException(e, sys)

    def table_exists(self, table_name: str) -> bool:
        """
        Checks if a table exists in the database.

        Args:
            table_name (str): Table name.

        Returns:
            bool: True if exists, else False.
        """
        try:
            if not self.connection:
                self.connect()
            cursor = self.connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
            exists = cursor.fetchone() is not None
            logger.info(f"Table '{table_name}' exists: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            raise CustomException(e, sys)

    def execute(self, sql: str):
        """
        Executes a generic SQL command.

        Args:
            sql (str): SQL command.
        """
        try:
            if not self.connection:
                self.connect()
            cursor = self.connection.cursor()
            cursor.execute(sql)
            self.connection.commit()
            logger.info(f"Executed SQL: {sql}")
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise CustomException(e, sys)
