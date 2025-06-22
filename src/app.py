import dash
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update, clientside_callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

# Assurez-vous que preprocess.py est dans le même dossier
try:
    from preprocess import filter_data
except ImportError:
    print("AVERTISSEMENT: 'preprocess.py' non trouvé. Utilisation d'une fonction de filtrage de secours.")
    def filter_data(df, col, val):
        if isinstance(val, (tuple, list)):
            return df[(df[col] >= val[0]) & (df[col] <= val[1])]
        else:
            return df[df[col] == val]

# ==============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==============================================================================

try:
    # Chemin robuste vers les données
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'assets', 'data', 'Medicaldataset.csv')
    df_full = pd.read_csv(csv_path)
    
    # Renommage et nettoyage
    df_full = df_full.rename(columns={'Age': 'age', 'Gender': 'gender', 'Heart rate': 'heart_rate', 'Systolic blood pressure': 'systolic_bp', 'Diastolic blood pressure': 'diastolic_bp', 'Blood sugar': 'blood_sugar', 'CK-MB': 'ck_mb', 'Troponin': 'troponin', 'Result': 'outcome'})
    df_full = df_full[df_full['heart_rate'] < 1000]
    if 'gender' in df_full.columns and pd.api.types.is_numeric_dtype(df_full['gender']):
        df_full['gender'] = df_full['gender'].map({0: 'Female', 1: 'Male'})
    df_full['outcome'] = df_full['outcome'].str.capitalize()
except FileNotFoundError:
    print("ERREUR: Fichier CSV non trouvé. L'application fonctionnera avec des données vides.")
    df_full = pd.DataFrame({'age': [], 'gender': [], 'heart_rate': [], 'systolic_bp': [], 'diastolic_bp': [], 'blood_sugar': [], 'ck_mb': [], 'troponin': [], 'outcome': []})

# Valeurs initiales et configuration
initial_age_range = [df_full['age'].min(), df_full['age'].max()] if not df_full.empty else [20, 80]
initial_gender = 'All'
graph_config = {'displaylogo': False, 'toImageButtonOptions': {'format': 'png', 'filename': 'chart_export', 'height': 600, 'width': 800, 'scale': 2}}

# ==============================================================================
# 2. Fonctions de Création de Composants
# ==============================================================================

def tooltip(text):
    """Crée une infobulle d'aide."""
    return html.Span(className="tooltip-container", children=[
        html.I(className="fas fa-info-circle text-slate-400 dark:text-slate-500 ml-2 cursor-pointer"),
        html.Div(className="tooltip-text", children=text)
    ])

def create_visualization_card(card_id, title, tooltip_text, graph_id, insight_id, age_slider_id, gender_dropdown_id):
    """Crée une carte de visualisation complète pour réduire la répétition dans le layout."""
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
    """Crée le graphique de distribution par âge et genre."""
    if df.empty: return go.Figure().update_layout(title='No data to display', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    bins = [14, 30, 45, 60, 75, 90, float('inf')]; labels = ['14-29', '30-44', '45-59', '60-74', '75-89', '90+']
    df_copy = df.copy(); df_copy['age_group'] = pd.cut(df_copy['age'], bins=bins, labels=labels, right=False)
    grouped_df = df_copy.groupby(['age_group', 'gender', 'outcome'], observed=True).size().reset_index(name='count')
    fig = px.bar(grouped_df, x='age_group', y='count', color='outcome', facet_col='gender', labels={'count': 'Number of Patients', 'age_group': 'Age Group'}, color_discrete_map={'Positive': '#f97316', 'Negative': '#0ea5e9'})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20), legend_title_text='Outcome')
    fig.update_xaxes(title_text=None)
    return fig

def create_blood_pressure_chart(df):
    """Crée le graphique de la pression artérielle."""
    if df.empty: return go.Figure().update_layout(title='No data to display', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig = px.scatter(df, x='systolic_bp', y='diastolic_bp', color='outcome', labels={'systolic_bp': 'Systolic BP (mmHg)', 'diastolic_bp': 'Diastolic BP (mmHg)'}, color_discrete_map={'Positive': '#f97316', 'Negative': '#0ea5e9'})
    fig.add_hline(y=90, line_dash="dash", line_color="gray"); fig.add_vline(x=140, line_dash="dash", line_color="gray")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20), legend_title_text='Outcome')
    return fig

def create_biomarkers_chart(df):
    """Crée le graphique des biomarqueurs."""
    if df.empty: return go.Figure().update_layout(title='No data to display', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    biomarkers_median = df.groupby('outcome')[['ck_mb', 'troponin']].median().reset_index()
    biomarkers_melted = biomarkers_median.melt(id_vars='outcome', value_vars=['ck_mb', 'troponin'], var_name='biomarker', value_name='median_value')
    fig = px.bar(biomarkers_melted, x='biomarker', y='median_value', color='outcome', barmode='group', labels={'median_value': 'Median Level (log scale)', 'biomarker': 'Biomarker'}, color_discrete_map={'Positive': '#f97316', 'Negative': '#0ea5e9'}, log_y=True, text_auto='.2s')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20), legend_title_text='Outcome')
    return fig

# ==============================================================================
# 4. INITIALISATION DE L'APPLICATION DASH
# ==============================================================================
app = Dash(__name__, external_scripts=['https://cdn.tailwindcss.com?plugins=forms', 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css'])
app.title = "Cardiac Risk Dashboard"

# ==============================================================================
# 5. MISE EN PAGE DE L'APPLICATION (app.layout)
# ==============================================================================
app.layout = html.Div(id='main-container', className='bg-slate-50 dark:bg-slate-900 text-slate-800 dark:text-slate-200', children=[
    
    # Stores pour la gestion d'état
    dcc.Store(id='theme-store', storage_type='local', data=False),
    dcc.Store(id='filter-store', storage_type='session', data={'age_range': initial_age_range, 'gender': initial_gender}),

    # En-tête et barre de navigation
    html.Header(className='bg-white dark:bg-slate-950/70 backdrop-blur-sm shadow-md sticky top-0 z-50', children=[
        html.Div(className='container mx-auto px-4 py-3', children=[
            html.H1('Heart Attack Risk Factor Dashboard', className='text-xl md:text-2xl font-bold'),
            html.Nav(className='flex justify-between items-center mt-2', children=[
                html.Div(className='flex space-x-4', children=[
                    html.A('Demographics', href='#demographics-card', className='text-sm text-slate-600 dark:text-slate-300 hover:text-sky-500 font-semibold'),
                    html.A('Blood Pressure', href='#bp-card', className='text-sm text-slate-600 dark:text-slate-300 hover:text-sky-500 font-semibold'),
                    html.A('Biomarkers', href='#biomarkers-card', className='text-sm text-slate-600 dark:text-slate-300 hover:text-sky-500 font-semibold'),
                ]),
                # Bloc de contrôles unique et nettoyé
                html.Div(className='flex items-center space-x-4', children=[
                    html.Button("Save Filters", id='save-button', className='text-xs bg-slate-200 dark:bg-slate-700 px-2 py-1 rounded-md hover:bg-slate-300'),
                    html.Button("Load Filters", id='load-button', className='text-xs bg-slate-200 dark:bg-slate-700 px-2 py-1 rounded-md hover:bg-slate-300'),
                    dcc.Checklist(options=[{'label': ' Sync', 'value': 'sync'}], value=[], id='sync-switch', className='text-sm', inputClassName='mr-1'),
                    html.Div(className='flex items-center', children=[
                        html.I(id='theme-icon', className='fas fa-sun text-yellow-500 mr-2'),
                        dcc.Checklist(
                            options=[{'label': 'Dark Mode', 'value': 'dark'}], 
                            value=[], 
                            id='theme-switch',
                            className='text-sm',
                            inputClassName='mr-2'
                        ),
                    ]),
                ])
            ])
        ])
    ]),
    
    # Contenu principal
    html.Main(className='container mx-auto p-4 md:p-8', children=[
        create_visualization_card('demographics-card', 'Outcome Distribution', 'Number of patients by age group and gender.', 'age-gender-chart', 'insight-demographics', 'age-slider-1', 'gender-dropdown-1'),
        create_visualization_card('bp-card', 'Blood Pressure vs. Outcome', 'Systolic vs. Diastolic BP. Lines indicate hypertension thresholds (140/90).', 'blood-pressure-chart', 'insight-bp', 'age-slider-2', 'gender-dropdown-2'),
        create_visualization_card('biomarkers-card', 'Cardiac Biomarker Analysis', 'Median levels of CK-MB and Troponin, key indicators of heart muscle damage.', 'biomarkers-chart', 'insight-biomarkers', 'age-slider-3', 'gender-dropdown-3'),
    ])
])

# ==============================================================================
# 6. CALLBACKS
# ==============================================================================

# --- Callbacks côté client (JavaScript) ---
clientside_callback(
    "function(is_dark){ document.documentElement.classList.toggle('dark', is_dark); return dash_clientside.no_update; }",
    Output('main-container', 'className', allow_duplicate=True), Input('theme-store', 'data'), prevent_initial_call=True
)
clientside_callback(
    "function(n,data){ if(n>0) localStorage.setItem('cardiac_dashboard_filters', JSON.stringify(data)); return ''; }",
    Output('save-button', 'data-saved'), Input('save-button', 'n_clicks'), State('filter-store', 'data'), prevent_initial_call=True
)
clientside_callback(
    "function(n){ if(n>0){ const d=localStorage.getItem('cardiac_dashboard_filters'); if(d) return JSON.parse(d); } return dash_clientside.no_update; }",
    Output('filter-store', 'data', allow_duplicate=True), Input('load-button', 'n_clicks'), prevent_initial_call=True
)

# --- Callbacks côté serveur (Python) ---
@app.callback(Output('theme-store', 'data'), Input('theme-switch', 'value'))
def update_theme_store(switch_value):
    return 'dark' in switch_value

@app.callback(
    Output('theme-icon', 'className'),
    Input('theme-store', 'data')
)
def update_theme_icon(is_dark):
    """Change l'icône du thème en fonction du mode."""
    if is_dark:
        return 'fas fa-moon text-indigo-400 mr-2'
    else:
        return 'fas fa-sun text-yellow-500 mr-2'

@app.callback(
    Output('filter-store', 'data', allow_duplicate=True),
    Output('age-slider-1', 'value'), Output('gender-dropdown-1', 'value'),
    Output('age-slider-2', 'value'), Output('gender-dropdown-2', 'value'),
    Output('age-slider-3', 'value'), Output('gender-dropdown-3', 'value'),
    Input('filter-store', 'data'),
    Input('age-slider-1', 'value'), Input('gender-dropdown-1', 'value'),
    Input('age-slider-2', 'value'), Input('gender-dropdown-2', 'value'),
    Input('age-slider-3', 'value'), Input('gender-dropdown-3', 'value'),
    State('sync-switch', 'value'),
    prevent_initial_call=True
)
def sync_filters(store_data, age1, gen1, age2, gen2, age3, gen3, sync_on):
    """Gère la synchronisation des filtres entre les cartes."""
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'filter-store':
        loaded_age = store_data.get('age_range', initial_age_range)
        loaded_gender = store_data.get('gender', initial_gender)
        return no_update, loaded_age, loaded_gender, loaded_age, loaded_gender, loaded_age, loaded_gender
    if 'sync' in sync_on:
        age_map = {'age-slider-1': age1, 'age-slider-2': age2, 'age-slider-3': age3}
        gender_map = {'gender-dropdown-1': gen1, 'gender-dropdown-2': gen2, 'gender-dropdown-3': gen3}
        new_age_range = age_map.get(trigger_id, initial_age_range)
        new_gender = gender_map.get(trigger_id, initial_gender)
        new_store_data = {'age_range': new_age_range, 'gender': new_gender}
        return new_store_data, new_age_range, new_gender, new_age_range, new_gender, new_age_range, new_gender
    return no_update, no_update, no_update, no_update, no_update, no_update, no_update

@app.callback(
    Output('age-gender-chart', 'figure'), Output('insight-demographics', 'children'),
    Input('age-slider-1', 'value'), Input('gender-dropdown-1', 'value')
)
def update_demographics(age_range, gender_value):
    """Met à jour le graphique et l'insight pour les données démographiques."""
    filtered_df = filter_data(df_full, 'age', age_range)
    if gender_value != 'All': filtered_df = filter_data(filtered_df, 'gender', gender_value)
    insight = "No significant trend identified for this selection."
    if not filtered_df.empty:
        pos_cases = filtered_df[filtered_df['outcome'] == 'Positive']
        if not pos_cases.empty:
            bins = [14, 30, 45, 60, 75, 90, float('inf')]; labels = ['14-29', '30-44', '45-59', '60-74', '75-89', '90+']
            pos_cases = pos_cases.copy(); pos_cases.loc[:, 'age_group'] = pd.cut(pos_cases['age'], bins=bins, labels=labels, right=False)
            if not pos_cases.empty:
                peak_group = pos_cases.groupby(['age_group', 'gender'], observed=True).size()
                if not peak_group.empty:
                    peak_group = peak_group.idxmax(); insight = f"Insight: The highest number of positive cases is in the {peak_group[0]} age group for {peak_group[1]}s."
    return create_age_gender_chart(filtered_df), insight

@app.callback(
    Output('blood-pressure-chart', 'figure'), Output('insight-bp', 'children'),
    Input('age-slider-2', 'value'), Input('gender-dropdown-2', 'value')
)
def update_blood_pressure(age_range, gender_value):
    """Met à jour le graphique et l'insight pour la pression artérielle."""
    filtered_df = filter_data(df_full, 'age', age_range)
    if gender_value != 'All': filtered_df = filter_data(filtered_df, 'gender', gender_value)
    insight = "Not enough data to determine a trend."
    pos_cases = filtered_df[filtered_df['outcome'] == 'Positive']
    if not pos_cases.empty:
        hypertensive = pos_cases[(pos_cases['systolic_bp'] >= 140) | (pos_cases['diastolic_bp'] >= 90)]
        percent_hyper = (len(hypertensive) / len(pos_cases)) * 100
        insight = f"Insight: {percent_hyper:.0f}% of positive cases in this group are in the hypertensive range."
    return create_blood_pressure_chart(filtered_df), insight

@app.callback(
    Output('biomarkers-chart', 'figure'), Output('insight-biomarkers', 'children'),
    Input('age-slider-3', 'value'), Input('gender-dropdown-3', 'value')
)
def update_biomarkers(age_range, gender_value):
    """Met à jour le graphique et l'insight pour les biomarqueurs."""
    filtered_df = filter_data(df_full, 'age', age_range)
    if gender_value != 'All': filtered_df = filter_data(filtered_df, 'gender', gender_value)
    insight = "Not enough data to compare biomarkers."
    df_pos = filtered_df[filtered_df['outcome'] == 'Positive']
    df_neg = filtered_df[filtered_df['outcome'] == 'Negative']
    if not df_pos.empty and not df_neg.empty and df_neg['troponin'].median() > 0:
        fold_change = df_pos['troponin'].median() / df_neg['troponin'].median()
        insight = f"Insight: Median Troponin is {fold_change:.1f} times higher in positive cases for this selection."
    return create_biomarkers_chart(filtered_df), insight

# ==============================================================================
# 7. POINT D'ENTRÉE DE L'APPLICATION
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True)
