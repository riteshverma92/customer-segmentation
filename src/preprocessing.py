import pandas as pd
from typing import Union
import logging

# Configure logging (better than using print statements in production)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_and_preprocess(file_path: Union[str, bytes]) -> pd.DataFrame:
    """
    Load a CSV file, clean it by removing unused columns, 
    and handle missing values with column medians.

    Parameters
    ----------
    file_path : str or bytes
        Path to the CSV file to be loaded.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with missing values filled with median and 
        unnecessary columns removed.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info("CSV file loaded successfully with shape %s", df.shape)
    except FileNotFoundError:
        logging.error("File not found at path: %s", file_path)
        raise
    except Exception as e:
        logging.error("Error reading CSV file: %s", e)
        raise

    # Drop unused columns if present
    columns_to_drop = ["CUST_ID"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors="ignore")
    logging.info("Dropped unused columns: %s", columns_to_drop)

    # Fill missing values with median for selected numeric columns
    columns_to_fill = ["MINIMUM_PAYMENTS", "CREDIT_LIMIT"]
    for col in columns_to_fill:
        if col in df.columns:
            median_value = df[col].median()
            missing_count = df[col].isna().sum()
            df[col] = df[col].fillna(median_value)
            logging.info("Filled %d missing values in '%s' with median: %f", missing_count, col, median_value)

    logging.info("Preprocessing completed. Final shape: %s", df.shape)
    return df
