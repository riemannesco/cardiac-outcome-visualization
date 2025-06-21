import pandas as pd
import utils


# Commentaire à supprimer: Permet de faire le graphe de la première vis
def get_results_by_age_range(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Count the number of patients in different age ranges.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        pd.DataFrame: DataFrame with age ranges and their respective counts.
    '''
    return utils.get_results_by_range(dataframe, 'Age')


def get_heart_rate_median(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate the median heart rate for each outcome.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        pd.DataFrame: DataFrame with the median heart rate for each outcome.
    '''
    return utils.get_median_value(dataframe, 'Heart rate')


def get_blood_sugar_median(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate the median blood sugar for each outcome.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        pd.DataFrame: DataFrame with the median blood sugar for each outcome.
    '''
    return utils.get_median_value(dataframe, 'Blood sugar')


# Commentaire à supprimer: Permet de faire le graphe de la deuxième vis
def filter_systolic_diastolic_blood_pressure(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Filter the columns of the dataframe to only include systolic and diastolic blood pressure and result.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        pd.DataFrame: A new DataFrame containing the filtered dataframe.
    '''
    filtered_df = dataframe[['Systolic blood pressure', 'Diastolic blood pressure', 'Result']].copy()
    filtered_df.columns = ['Systolic BP', 'Diastolic BP', 'Result']
    filtered_df['Result'] = filtered_df['Result'].astype('category')
    return filtered_df


# Commentaire à supprimer: Permet de faire le graphe de la troisième vis
def get_ckmb_median(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate the median CK-MB for each outcome.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        pd.DataFrame: DataFrame with the median CK-MB for each outcome.
    '''
    return utils.get_median_value(dataframe, 'CK-MB')


def get_troponin_median(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate the median Troponin for each outcome.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        pd.DataFrame: DataFrame with the median Troponin for each outcome.
    '''
    return utils.get_median_value(dataframe, 'Troponin')


# Commentaire à supprimer: Permet de faire le graphe de la quatrième vis
def get_results_by_systolic_blood_pressure_range(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Count the number of patients in different ranges of systolic blood pressure.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        pd.DataFrame: DataFrame with blood pressure ranges and its counts.
    '''
    return utils.get_results_by_range(dataframe, 'Systolic blood pressure')


def get_results_by_blood_sugar_range(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Count the number of patients in different ranges of blood sugar.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        pd.DataFrame: DataFrame with blood sugar ranges and their respective counts.
    '''
    return utils.get_results_by_range(dataframe, 'Blood sugar')


def get_bloodsugar_systolic_heatmap_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Count the number of patients in different ranges of 'Systolic blood pressure' and 'Blood sugar'.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        pd.DataFrame: DataFrame with counts for each combination of blood pressure and blood sugar ranges.
    '''
    return utils.get_heatmap_data(dataframe, 'Systolic blood pressure', 'Blood sugar')
