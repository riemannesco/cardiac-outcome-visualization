"""
Module de Prétraitement

Ce module fournit des fonctions de haut niveau pour préparer les données
spécifiques à chaque visualisation du tableau de bord. Il utilise les
fonctions de bas niveau définies dans utils.py.
"""
import pandas as pd
import utils

def get_mean_values_by_result(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calcule le profil moyen (valeurs moyennes) pour chaque type de résultat."""
    mean_values_dict = {}
    for column_name in dataframe.columns:
        if dataframe[column_name].dtype not in ['int64', 'float64']:
            continue
        mean_values = dataframe.groupby('outcome')[column_name].mean()
        mean_values_dict[column_name] = mean_values
    mean_values_df = pd.DataFrame(mean_values_dict)
    return mean_values_df.reset_index()

def get_bloodsugar_systolic_heatmap_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Prépare les données pour la heatmap Pression Systolique vs. Glycémie."""
    return utils.get_heatmap_data(dataframe, 'systolic_bp', 'blood_sugar')
