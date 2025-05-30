import pandas as pd

def read(file_path):
    """
    Loads a CSV file into a pandas DataFrame.

    Parameters:
        file_path (str): The path to the file to be loaded.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV file.")
        
        print(f"File successfully loaded: {file_path}")
        return df

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

