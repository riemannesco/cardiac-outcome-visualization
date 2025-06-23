"""
Application Principale du Tableau de Bord Dash

Ce fichier contient la logique de l'application, la mise en page (layout)
et les callbacks qui gèrent l'interactivité.
"""
import dash
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update, clientside_callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os

# --- Importation des modules locaux ---
try:
    from utils import filter_data
    from preprocess import get_bloodsugar_systolic_heatmap_data, get_mean_values_by_result
except ImportError:
    print("ERREUR CRITIQUE: Impossible d'importer les modules `preprocess.py` ou `utils.py`. Assurez-vous qu'ils sont dans le même dossier.")
    # Fallback pour éviter un crash au démarrage
    def filter_data(df, col, val): return pd.DataFrame()
    def get_bloodsugar_systolic_heatmap_data(df): return pd.DataFrame()
    def get_mean_values_by_result(df): return pd.DataFrame()

# ==============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==============================================================================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'assets', 'data', 'Medicaldataset.csv')
    df_full = pd.read_csv(csv_path)
    df_full = df_full.rename(columns={'Age': 'age', 'Gender': 'gender', 'Heart rate': 'heart_rate', 'Systolic blood pressure': 'systolic_bp', 'Diastolic blood pressure': 'diastolic_bp', 'Blood sugar': 'blood_sugar', 'CK-MB': 'ck_mb', 'Troponin': 'troponin', 'Result': 'outcome'})
    df_full = df_full[df_full['heart_rate'] < 1000]
    if 'gender' in df_full.columns and pd.api.types.is_numeric_dtype(df_full['gender']):
        df_full['gender'] = df_full['gender'].map({0: 'Female', 1: 'Male'})
    df_full['outcome'] = df_full['outcome'].str.capitalize()
except FileNotFoundError:
    print("ERREUR: Le fichier CSV 'Medicaldataset.csv' est introuvable dans le dossier 'assets/data'.")
    df_full = pd.DataFrame()

initial_age_range = [df_full['age'].min(), df_full['age'].max()] if not df_full.empty else [20, 80]
initial_gender = 'All'
graph_config = {'displaylogo': False, 'toImageButtonOptions': {'format': 'png', 'filename': 'chart_export', 'height': 600, 'width': 800, 'scale': 2}}

# ==============================================================================
# 2. Fonctions de Création de Composants
# ==============================================================================
def tooltip(text):
    """Génère une infobulle d'aide."""
    return html.Span(className="tooltip-container", children=[html.I(className="fas fa-info-circle text-slate-400 ml-2 cursor-pointer"), html.Div(className="tooltip-text", children=text)])

def create_visualization_card(card_id, title, tooltip_text, graph_id, insight_id, age_slider_id, gender_dropdown_id):
    """Génère une carte de visualisation complète pour éviter la répétition."""
    return html.Section(id=card_id, className='card', children=[
        html.H2(children=[title, tooltip(tooltip_text)], className='card-title'),
        html.Div(className='card-filters', children=[
            html.Div([html.Label('Age Range'), dcc.RangeSlider(id=age_slider_id, min=initial_age_range[0], max=initial_age_range[1], value=initial_age_range, marks=None, tooltip={"placement": "bottom", "always_visible": True})]),
            html.Div([html.Label('Gender'), dcc.Dropdown(id=gender_dropdown_id, options=['All', 'Male', 'Female'], value=initial_gender, clearable=False, searchable=False)]),
        ]),
        dcc.Loading(children=dcc.Graph(id=graph_id, config=graph_config)),
        html.P(id=insight_id, className='card-insight')
    ])

# ==============================================================================
# 3. FONCTIONS DE CRÉATION DE GRAPHIQUES
# ==============================================================================
def create_age_gender_chart(df):
    """Génère le graphique de distribution par âge et genre."""
    if df.empty: return go.Figure().update_layout(title='No data to display', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    bins, labels = [14, 30, 45, 60, 75, 90, float('inf')], ['14-29', '30-44', '45-59', '60-74', '75-89', '90+']
    df_copy = df.copy(); df_copy.loc[:, 'age_group'] = pd.cut(df_copy['age'], bins=bins, labels=labels, right=False)
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Male', 'Female'), x_title='Age Group')
    legend_added = {'Positive': False, 'Negative': False}
    for i, gender in enumerate(['Male', 'Female']):
        col, gender_df = i + 1, df_copy[df_copy['gender'] == gender]
        if gender_df.empty: continue
        counts = gender_df.groupby(['age_group', 'outcome'], observed=True).size().reset_index(name='count')
        positive_data, negative_data = counts[counts['outcome'] == 'Positive'], counts[counts['outcome'] == 'Negative']
        if not positive_data.empty:
            fig.add_trace(go.Bar(x=positive_data['age_group'], y=positive_data['count'], name='Positive', marker_color='#f97316', legendgroup='Positive', showlegend=not legend_added['Positive']), row=1, col=col)
            legend_added['Positive'] = True
        if not negative_data.empty:
            fig.add_trace(go.Bar(x=negative_data['age_group'], y=negative_data['count'], name='Negative', marker_color='#0ea5e9', legendgroup='Negative', showlegend=not legend_added['Negative']), row=1, col=col)
            legend_added['Negative'] = True
    fig.update_layout(barmode='stack', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=10, l=10, r=10), legend_title_text='Outcome', yaxis_title='Number of Patients')
    return fig

def create_blood_pressure_chart(df):
    """Génère le nuage de points pour la pression artérielle."""
    if df.empty: return go.Figure().update_layout(title='No data to display', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig = px.scatter(df, x='systolic_bp', y='diastolic_bp', color='outcome', labels={'systolic_bp': 'Systolic BP (mmHg)', 'diastolic_bp': 'Diastolic BP (mmHg)'}, color_discrete_map={'Positive': '#f97316', 'Negative': '#0ea5e9'})
    fig.add_hline(y=90, line_dash="dash", line_color="gray"); fig.add_vline(x=140, line_dash="dash", line_color="gray")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20), legend_title_text='Outcome'); return fig

def create_biomarkers_chart(df):
    """Génère le graphique en barres des biomarqueurs."""
    if df.empty: return go.Figure().update_layout(title='No data to display', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    biomarkers_median = df.groupby('outcome')[['ck_mb', 'troponin']].median().reset_index()
    biomarkers_melted = biomarkers_median.melt(id_vars='outcome', value_vars=['ck_mb', 'troponin'], var_name='biomarker', value_name='median_value')
    fig = px.bar(biomarkers_melted, x='biomarker', y='median_value', color='outcome', barmode='group', labels={'median_value': 'Median Level (log scale)', 'biomarker': 'Biomarker'}, color_discrete_map={'Positive': '#f97316', 'Negative': '#0ea5e9'}, log_y=True, text_auto='.2s')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20), legend_title_text='Outcome'); return fig

def create_heatmap_chart(df):
    """Génère la heatmap Pression Systolique vs. Glycémie."""
    heatmap_long_df = get_bloodsugar_systolic_heatmap_data(df)
    if heatmap_long_df.empty: return go.Figure().update_layout(title='No data for this selection', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    try:
        heatmap_pivot = heatmap_long_df.pivot(index='systolic_bp Range', columns='blood_sugar Range', values='Total')
        fig = px.imshow(heatmap_pivot, text_auto=True, aspect="auto", color_continuous_scale='OrRd', labels=dict(x="Blood Sugar Range", y="Systolic BP Range", color="Patient Count"))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20)); return fig
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        return go.Figure().update_layout(title='Error processing heatmap data', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

def create_radar_chart(df):
    """Génère le graphique radar comparant les profils moyens."""
    if df.empty or len(df['outcome'].unique()) < 2:
        return go.Figure().update_layout(title='Not enough data for comparison (requires both outcomes)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    mean_df = get_mean_values_by_result(df)
    if mean_df.empty or len(mean_df) < 2:
        return go.Figure().update_layout(title='Not enough data for comparison', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    numeric_cols = [col for col in mean_df.columns if col != 'outcome']
    scaled_df = mean_df.copy()
    
    for col in numeric_cols:
        min_val, max_val = df_full[col].min(), df_full[col].max()
        range_val = max_val - min_val
        if range_val > 0: scaled_df[col] = (scaled_df[col] - min_val) / range_val
        else: scaled_df[col] = 0.5
    
    categories = [col.replace('_', ' ').title() for col in numeric_cols]
    fig = go.Figure()
    colors = {'Positive': '#f97316', 'Negative': '#0ea5e9'}
    fill_colors = {'Positive': 'rgba(249, 115, 22, 0.4)', 'Negative': 'rgba(14, 165, 233, 0.4)'}

    for index, row in scaled_df.iterrows():
        outcome = row['outcome']
        values = row[numeric_cols].values.tolist()
        closed_values = values + [values[0]]
        closed_categories = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(r=closed_values, theta=closed_categories, fill='toself', name=outcome, line=dict(color=colors.get(outcome)), fillcolor=fill_colors.get(outcome)))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=80, b=40, l=40, r=40), legend_title_text='Outcome')
    return fig

# ==============================================================================
# 4. INITIALISATION ET MISE EN PAGE DE L'APPLICATION
# ==============================================================================
app = Dash(__name__, external_scripts=['https://cdn.tailwindcss.com?plugins=forms', 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css'])
app.title = "Cardiac Risk Dashboard"

app.layout = html.Div(className='bg-slate-50 text-slate-800', children=[
    dcc.Store(id='filter-store', storage_type='session', data={'age_range': initial_age_range, 'gender': initial_gender}),
    html.Header(className='bg-white shadow-md sticky top-0 z-50', children=[
        html.Div(className='container mx-auto px-4 py-3', children=[
            html.H1('Heart Attack Risk Factor Dashboard', className='text-xl md:text-2xl font-bold'),
            html.Nav(className='flex justify-between items-center mt-2', children=[
                html.Div(className='flex flex-wrap space-x-4', children=[
                    html.A('Demographics', href='#demographics-card', className='text-sm text-slate-600 hover:text-sky-500 font-semibold'),
                    html.A('Blood Pressure', href='#bp-card', className='text-sm text-slate-600 hover:text-sky-500 font-semibold'),
                    html.A('Biomarkers', href='#biomarkers-card', className='text-sm text-slate-600 hover:text-sky-500 font-semibold'),
                    html.A('Heatmap', href='#heatmap-card', className='text-sm text-slate-600 hover:text-sky-500 font-semibold'),
                    html.A('Profiles', href='#profiles-card', className='text-sm text-slate-600 hover:text-sky-500 font-semibold'),
                ]),
                html.Div(className='flex items-center space-x-4', children=[
                    html.Button("Save Filters", id='save-button', className='text-xs bg-slate-200 px-2 py-1 rounded-md hover:bg-slate-300'),
                    html.Button("Load Filters", id='load-button', className='text-xs bg-slate-200 px-2 py-1 rounded-md hover:bg-slate-300'),
                    dcc.Checklist(options=[{'label': ' Sync', 'value': 'sync'}], value=[], id='sync-switch', className='text-sm', inputClassName='mr-1'),
                ])
            ])
        ])
    ]),
    html.Main(className='container mx-auto p-4 md:p-8', children=[
        create_visualization_card('demographics-card', 'Outcome Distribution', 'Number of patients by age group and gender.', 'age-gender-chart', 'insight-demographics', 'age-slider-1', 'gender-dropdown-1'),
        create_visualization_card('bp-card', 'Blood Pressure vs. Outcome', 'Systolic vs. Diastolic BP. Lines indicate hypertension thresholds (140/90).', 'blood-pressure-chart', 'insight-bp', 'age-slider-2', 'gender-dropdown-2'),
        create_visualization_card('biomarkers-card', 'Cardiac Biomarker Analysis', 'Median levels of CK-MB and Troponin.', 'biomarkers-chart', 'insight-biomarkers', 'age-slider-3', 'gender-dropdown-3'),
        create_visualization_card('heatmap-card', 'BP vs. Blood Sugar Heatmap', 'Patient concentration by blood sugar and systolic pressure.', 'heatmap-chart', 'insight-heatmap', 'age-slider-4', 'gender-dropdown-4'),
        create_visualization_card('profiles-card', 'Average Patient Profiles', 'Compares the "average" patient profile for each outcome.', 'radar-chart', 'insight-profiles', 'age-slider-5', 'gender-dropdown-5'),
    ])
])

# ==============================================================================
# 5. CALLBACKS
# ==============================================================================
clientside_callback("function(n,data){ if(n>0) localStorage.setItem('cardiac_dashboard_filters', JSON.stringify(data)); return ''; }", Output('save-button', 'data-saved'), Input('save-button', 'n_clicks'), State('filter-store', 'data'), prevent_initial_call=True)
clientside_callback("function(n){ if(n>0){ const d=localStorage.getItem('cardiac_dashboard_filters'); if(d) return JSON.parse(d); } return window.dash_clientside.no_update; }", Output('filter-store', 'data', allow_duplicate=True), Input('load-button', 'n_clicks'), prevent_initial_call=True)

@app.callback(
    Output('filter-store', 'data', allow_duplicate=True),
    [Output(f'age-slider-{i}', 'value') for i in range(1, 6)],
    [Output(f'gender-dropdown-{i}', 'value') for i in range(1, 6)],
    Input('filter-store', 'data'),
    [Input(f'age-slider-{i}', 'value') for i in range(1, 6)],
    [Input(f'gender-dropdown-{i}', 'value') for i in range(1, 6)],
    State('sync-switch', 'value'),
    prevent_initial_call=True
)
def sync_filters(store_data, age1, age2, age3, age4, age5, gen1, gen2, gen3, gen4, gen5, sync_on):
    """Gère la synchronisation des filtres entre toutes les cartes."""
    ctx = callback_context
    if not ctx.triggered: return (no_update,) * 11
    trigger_id_str = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id_str == 'filter-store':
        age = store_data.get('age_range', initial_age_range)
        gender = store_data.get('gender', initial_gender)
        return (no_update, *([age]*5), *([gender]*5))

    if 'sync' in sync_on:
        all_ages = [age1, age2, age3, age4, age5]; all_genders = [gen1, gen2, gen3, gen4, gen5]
        try:
            trigger_type, trigger_index_str = trigger_id_str.rsplit('-', 1); idx = int(trigger_index_str) - 1
            new_age = all_ages[idx]; new_gender = all_genders[idx]
            new_store_data = {'age_range': new_age, 'gender': new_gender}
            return (new_store_data, *([new_age]*5), *([new_gender]*5))
        except (ValueError, IndexError):
            return (no_update,) * 11
    
    new_store_data = {'age_range': age1, 'gender': gen1}
    return (new_store_data,) + (no_update,) * 10

@app.callback(Output('age-gender-chart', 'figure'), Output('insight-demographics', 'children'), Input('age-slider-1', 'value'), Input('gender-dropdown-1', 'value'))
def update_demographics(age_range, gender_value):
    """Met à jour le graphique et l'insight pour les données démographiques."""
    df = filter_data(df_full, 'age', age_range)
    if gender_value != 'All': df = filter_data(df, 'gender', gender_value)
    insight = "No significant trend identified."; 
    if not df.empty and 'Positive' in df['outcome'].unique():
        pos_cases = df[df['outcome'] == 'Positive'].copy()
        if not pos_cases.empty:
            bins = [14, 30, 45, 60, 75, 90, float('inf')]; labels = ['14-29', '30-44', '45-59', '60-74', '75-89', '90+']
            pos_cases.loc[:, 'age_group'] = pd.cut(pos_cases['age'], bins=bins, labels=labels, right=False)
            peak_group = pos_cases.groupby(['age_group', 'gender'], observed=True).size()
            if not peak_group.empty: insight = f"Insight: Highest number of positive cases in the {peak_group.idxmax()[0]} age group for {peak_group.idxmax()[1]}s."
    return create_age_gender_chart(df), insight

@app.callback(Output('blood-pressure-chart', 'figure'), Output('insight-bp', 'children'), Input('age-slider-2', 'value'), Input('gender-dropdown-2', 'value'))
def update_blood_pressure(age_range, gender_value):
    """Met à jour le graphique et l'insight pour la pression artérielle."""
    df = filter_data(df_full, 'age', age_range)
    if gender_value != 'All': df = filter_data(df, 'gender', gender_value)
    insight = "Not enough data for a trend."
    if not df.empty and 'Positive' in df['outcome'].unique():
        pos_cases = df[df['outcome'] == 'Positive']
        if not pos_cases.empty:
            hyper = pos_cases[(pos_cases['systolic_bp'] >= 140) | (pos_cases['diastolic_bp'] >= 90)]
            if len(pos_cases) > 0: insight = f"Insight: {len(hyper) / len(pos_cases) * 100:.0f}% of positive cases are in the hypertensive range."
    return create_blood_pressure_chart(df), insight

@app.callback(Output('biomarkers-chart', 'figure'), Output('insight-biomarkers', 'children'), Input('age-slider-3', 'value'), Input('gender-dropdown-3', 'value'))
def update_biomarkers(age_range, gender_value):
    """Met à jour le graphique et l'insight pour les biomarqueurs."""
    df = filter_data(df_full, 'age', age_range)
    if gender_value != 'All': df = filter_data(df, 'gender', gender_value)
    insight = "Not enough data to compare."
    if not df.empty and 'Positive' in df['outcome'].unique() and 'Negative' in df['outcome'].unique():
        pos = df[df['outcome'] == 'Positive']; neg = df[df['outcome'] == 'Negative']
        if not pos.empty and not neg.empty and neg['troponin'].median() > 0:
            fold = pos['troponin'].median() / neg['troponin'].median()
            insight = f"Insight: Median Troponin is {fold:.1f}x higher in positive cases."
    return create_biomarkers_chart(df), insight

@app.callback(Output('heatmap-chart', 'figure'), Output('insight-heatmap', 'children'), Input('age-slider-4', 'value'), Input('gender-dropdown-4', 'value'))
def update_heatmap(age_range, gender_value):
    """Met à jour la heatmap."""
    df = filter_data(df_full, 'age', age_range)
    if gender_value != 'All': df = filter_data(df, 'gender', gender_value)
    insight = "Insight: Patient concentration by blood sugar and systolic BP."
    return create_heatmap_chart(df), insight

@app.callback(Output('radar-chart', 'figure'), Output('insight-profiles', 'children'), Input('age-slider-5', 'value'), Input('gender-dropdown-5', 'value'))
def update_radar_chart(age_range, gender_value):
    """Met à jour le graphique radar."""
    df = filter_data(df_full, 'age', age_range)
    if gender_value != 'All': df = filter_data(df, 'gender', gender_value)
    insight = "Insight: Compares the 'average' patient profile for each outcome."
    return create_radar_chart(df), insight

# ==============================================================================
# 6. POINT D'ENTRÉE DE L'APPLICATION
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True)
