import pandas as pd
from typing import Union, Tuple

from data_ranges import (
    AGE_BINS, AGE_LABELS,
    HEART_RATE_BINS, HEART_RATE_LABELS,
    SYSTOLIC_BP_BINS, SYSTOLIC_BP_LABELS,
    DIASTOLIC_BP_BINS, DIASTOLIC_BP_LABELS,
    BLOOD_SUGAR_BINS, BLOOD_SUGAR_LABELS,
    CK_MB_BINS, CK_MB_LABELS,
    TROPONIN_BINS, TROPONIN_LABELS
)


bins_labels = {
    'Age': (AGE_BINS, AGE_LABELS),
    'Heart rate': (HEART_RATE_BINS, HEART_RATE_LABELS),
    'Systolic blood pressure': (SYSTOLIC_BP_BINS, SYSTOLIC_BP_LABELS),
    'Diastolic blood pressure': (DIASTOLIC_BP_BINS, DIASTOLIC_BP_LABELS),
    'Blood sugar': (BLOOD_SUGAR_BINS, BLOOD_SUGAR_LABELS),
    'CK-MB': (CK_MB_BINS, CK_MB_LABELS),
    'Troponin': (TROPONIN_BINS, TROPONIN_LABELS)
}


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
    Filters a DataFrame on a specific column, either by a unique value or by an interval.

    Args:
        dataframe (pd.DataFrame): DataFrame to filter.
        column_name (str): The name of the column to filter on.
        filter_value (Union[str, int, float, Tuple[Union[int, float, None], Union[int, float, None]]]): 
            - The unique value for the filter (ex: 'positive', 1, 25.5).
            - A tuple (min, max) for an interval filter.
              Use None for an open limit (e.g., (50, None) for >= 50).

    Returns:
        pd.DataFrame: A new DataFrame containing the filtered data.

    Raises:
        ValueError: If the column does not exist in the DataFrame or if the filter format is invalid.
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


def get_mean_value(dataframe: pd.DataFrame, column_name: str) -> Union[float, None]:
    '''
    Calculate the average value of a specified column in the DataFrame for each outcome.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the dataset.
        column_name (str): The name of the column for which to calculate the median.

    Returns:
        Union[float, None]: The average value of the specified column for each outcome,
        or None if the column does not exist or if there are no valid values.
    '''
    if column_name not in dataframe.columns:
        raise ValueError(
            f"The column '{column_name}' does not exist in the DataFrame.")
    # Calculate the median for each outcome
    median_values = dataframe.groupby(
        'Result')[column_name].mean().reset_index()
    median_values.columns = ['Result', 'Median']
    return median_values


def get_results_by_range(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    '''
    Count the number of patients in different ranges of a specified column.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the dataset.
        column (str): The column to filter by range.

    Returns:
        pd.DataFrame: DataFrame with ranges and their respective counts.
    '''
    # Define the bins and labels based on the column
    if column not in bins_labels:
        raise ValueError(f"Column '{column}' is not supported for range filtering.")

    bins, labels = bins_labels[column]
    results_summary = []

    for i, label in enumerate(labels):
        min_val, max_val = bins[i]
        filtered_df = filter_data(dataframe, column, (min_val, max_val))
        positive_count = 0
        negative_count = 0

        if not filtered_df.empty:
            result_counts = count_result(filtered_df)
            positive_series = result_counts[result_counts['Result'] == 'positive']['Count']
            if not positive_series.empty:
                positive_count = positive_series.iloc[0]
            negative_series = result_counts[result_counts['Result'] == 'negative']['Count']
            if not negative_series.empty:
                negative_count = negative_series.iloc[0]

        results_summary.append({
            'Range': label,
            'Positive': positive_count,
            'Negative': negative_count,
            'Total': positive_count + negative_count
        })

    return pd.DataFrame(results_summary)


def get_heatmap_data(dataframe: pd.DataFrame, column_x: str, column_y: str) -> pd.DataFrame:
    '''
    Count the number of patients in different ranges of 'Systolic blood pressure' and 'Blood sugar'.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        pd.DataFrame: DataFrame with counts for each combination of ranges.
    '''
    heatmap_data = []

    if column_x not in bins_labels:
        raise ValueError(f"Columns '{column_x}' are not supported for heatmap data.")

    if column_y not in bins_labels:
        raise ValueError(f"Columns '{column_y}' are not supported for heatmap data.")

    bins_x, labels_x = bins_labels[column_x]
    bins_y, labels_y = bins_labels[column_y]

    for i, x_label in enumerate(labels_x):
        min_x, max_x = bins_x[i]

        for j, y_label in enumerate(labels_y):
            min_y, max_y = bins_y[j]

            # Filter data for the current range combination
            column_x_filtered_df = filter_data(dataframe, column_x, (min_x, max_x))
            combined_filtered_df = filter_data(column_x_filtered_df, column_y, (min_y, max_y))

            positive_count = 0
            negative_count = 0

            if not combined_filtered_df.empty:
                result_counts = count_result(combined_filtered_df)
                positive_series = result_counts[result_counts['Result'] == 'positive']['Count']
                if not positive_series.empty:
                    positive_count = positive_series.iloc[0]
                negative_series = result_counts[result_counts['Result'] == 'negative']['Count']
                if not negative_series.empty:
                    negative_count = negative_series.iloc[0]

            heatmap_data.append({
                f'{column_x} Range': x_label,
                f'{column_y} Range': y_label,
                'Positive': positive_count,
                'Negative': negative_count,
                'Total': positive_count + negative_count
            })

    return pd.DataFrame(heatmap_data)
