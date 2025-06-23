"""
Module Utilitaire pour le Traitement des Données

Ce module fournit des fonctions de bas niveau pour filtrer, compter
et agréger les données du dataset médical. Il est utilisé par preprocess.py.
"""
import pandas as pd
from typing import Union, Tuple, List
from data_ranges import *

bins_labels = {
    'age': (AGE_BINS, AGE_LABELS),
    'heart_rate': (HEART_RATE_BINS, HEART_RATE_LABELS),
    'systolic_bp': (SYSTOLIC_BP_BINS, SYSTOLIC_BP_LABELS),
    'diastolic_bp': (DIASTOLIC_BP_BINS, DIASTOLIC_BP_LABELS),
    'blood_sugar': (BLOOD_SUGAR_BINS, BLOOD_SUGAR_LABELS),
    'ck_mb': (CK_MB_BINS, CK_MB_LABELS),
    'troponin': (TROPONIN_BINS, TROPONIN_LABELS)
}

def filter_data(dataframe: pd.DataFrame, column_name: str, filter_value: Union[str, int, float, Tuple, List]) -> pd.DataFrame:
    """Filtre un DataFrame sur une colonne, par valeur unique ou par intervalle."""
    if column_name not in dataframe.columns:
        raise ValueError(f"La colonne '{column_name}' n'existe pas dans le DataFrame.")
    df_filtered = dataframe.copy()
    if isinstance(filter_value, (tuple, list)) and len(filter_value) == 2:
        min_val, max_val = filter_value
        mask = True
        if min_val is not None: mask &= (df_filtered[column_name] >= min_val)
        if max_val is not None: mask &= (df_filtered[column_name] <= max_val)
        return df_filtered[mask]
    elif isinstance(filter_value, (str, int, float)):
        return df_filtered[df_filtered[column_name] == filter_value]
    else:
        raise ValueError("Format de 'filter_value' invalide.")

def get_heatmap_data(dataframe: pd.DataFrame, column_x: str, column_y: str) -> pd.DataFrame:
    """Génère les données pour une heatmap en comptant les occurrences par intervalle croisé."""
    if column_x not in bins_labels or column_y not in bins_labels:
        raise ValueError("Colonnes non supportées pour la heatmap.")
    bins_x, labels_x = bins_labels[column_x]
    bins_y, labels_y = bins_labels[column_y]
    heatmap_data = []
    for i, x_label in enumerate(labels_x):
        min_x, max_x = bins_x[i]
        for j, y_label in enumerate(labels_y):
            min_y, max_y = bins_y[j]
            filtered_df_x = filter_data(dataframe, column_x, (min_x, max_x))
            combined_filtered_df = filter_data(filtered_df_x, column_y, (min_y, max_y))
            heatmap_data.append({
                f'{column_x} Range': x_label,
                f'{column_y} Range': y_label,
                'Total': len(combined_filtered_df)
            })
    return pd.DataFrame(heatmap_data)
