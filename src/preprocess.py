import pandas as pd
from typing import Union, Tuple


def count_result(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Count the number of occurences of postive and negative outcomes in the dataset.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the dataset.
    Returns:
        pd.DataFrame: DataFrame with the counts of positive and negative
        outcomes instead of the column 'Result'.
    '''
    result_counts = dataframe['Result'].value_counts().reset_index()
    result_counts.columns = ['Result', 'Count']
    return result_counts


def filter_data(dataframe: pd.DataFrame, column_name: str, filter_value: Union[str, int, float, Tuple[Union[int, float, None], Union[int, float, None]]]) -> pd.DataFrame:
    '''
    Filtre un DataFrame sur une colonne spécifique, soit par une valeur unique, soit par un intervalle.

    Args:
        dataframe (pd.DataFrame): Le DataFrame à filtrer.
        column_name (str): Le nom de la colonne sur laquelle appliquer le filtre.
        filter_value (Union[str, int, float, Tuple[Union[int, float, None], Union[int, float, None]]]): 
            - La valeur unique pour le filtre (ex: 'positive', 1, 25.5).
            - Un tuple (min, max) pour un filtre d'intervalle. 
              Utilisez None pour une limite ouverte (ex: (50, None) pour >= 50).

    Returns:
        pd.DataFrame: Un nouveau DataFrame contenant les données filtrées.

    Raises:
        ValueError: Si le nom de la colonne n'existe pas ou si le format du filtre est invalide.
    '''
    # Check if the column exists in the DataFrame, not supposed to happen
    if column_name not in dataframe.columns:
        raise ValueError(
            f"The column'{column_name}' does not exist in the DataFrame.")

    # We create a copy to avoid modifying the original DataFrame
    df_filtered = dataframe.copy()

    if isinstance(filter_value, tuple) and len(filter_value) == 2:
        # Filter by range
        min_val, max_val = filter_value
        if min_val is not None and max_val is not None:
            mask = (df_filtered[column_name] >= min_val) & (
                df_filtered[column_name] <= max_val)
        elif min_val is not None:
            mask = df_filtered[column_name] >= min_val
        elif max_val is not None:
            mask = df_filtered[column_name] <= max_val
        else:
            # If both are None, return the original DataFrame
            return df_filtered

        return df_filtered[mask]

    elif isinstance(filter_value, (str, int, float)):
        # Filter by single value
        return df_filtered[df_filtered[column_name] == filter_value]

    else:
        raise ValueError(
                "The format of 'filter_value' is invalid. Use a unique value or a tuple (min, max).")
