from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

# Assuming preprocess.py is in the same directory for the filter_data function
try:
    from preprocess import filter_data
except ImportError:
    print("WARNING: 'preprocess.py' not found. Using a fallback filter function.")
    def filter_data(df, col, val):
        if isinstance(val, tuple):
            return df[(df[col] >= val[0]) & (df[col] <= val[1])]
        else:
            return df[df[col] == val]

# ==============================================================================
# 1. DATA LOADING AND PREPARATION
# ==============================================================================

# Robust path to the data file
try:
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'assets', 'data', 'Medicaldataset.csv')
    df_full = pd.read_csv(csv_path)
    
    # Rename columns for consistency
    df_full = df_full.rename(columns={
        'Age': 'age',
        'Gender': 'gender',
        'Heart rate': 'heart_rate',
        'Systolic blood pressure': 'systolic_bp',
        'Diastolic blood pressure': 'diastolic_bp',
        'Blood sugar': 'blood_sugar',
        'CK-MB': 'ck_mb',
        'Troponin': 'troponin',
        'Result': 'outcome'
    })
    
    # Data cleaning
    df_full = df_full[df_full['heart_rate'] < 1000]

    # Convert gender to string values
    if 'gender' in df_full.columns and pd.api.types.is_numeric_dtype(df_full['gender']):
        df_full['gender'] = df_full['gender'].map({0: 'Female', 1: 'Male'})

except FileNotFoundError:
    print(f"ERROR: CSV file not found.")
    df_full = pd.DataFrame({
        'age': [20, 80], 'gender': ['Male', 'Female'], 'heart_rate': [0,0], 'systolic_bp': [0,0], 'diastolic_bp': [0,0],
        'blood_sugar': [0,0], 'ck_mb': [0,0], 'troponin': [0,0], 'outcome': ['Positive', 'Negative']
    })

# Define colors and initial filter values
colors = {'background': '#f8fafc', 'card': 'white', 'text': '#334155', 'grid': '#e2e8f0'}
initial_age_range = [df_full['age'].min(), df_full['age'].max()]
initial_gender = 'All'

# SVG Icons for card titles
icon_demographics = html.I(className="fas fa-users mr-3 text-slate-500")
icon_blood_pressure = html.I(className="fas fa-heartbeat mr-3 text-slate-500")
icon_biomarkers = html.I(className="fas fa-vial mr-3 text-slate-500")


# ==============================================================================
# 2. Reusable Chart Creation Functions
# ==============================================================================
# (Functions remain the same as the previous version, just ensuring English labels)

def create_age_gender_chart(df):
    if df.empty: return go.Figure().update_layout(title='No data to display for this selection')
    bins = [14, 30, 45, 60, 75, 90, float('inf')]; labels = ['14-29', '30-44', '45-59', '60-74', '75-89', '90+']
    df_copy = df.copy(); df_copy['age_group'] = pd.cut(df_copy['age'], bins=bins, labels=labels, right=False)
    grouped_df = df_copy.groupby(['age_group', 'gender', 'outcome'], observed=True).size().reset_index(name='count')
    fig = px.bar(grouped_df, x='age_group', y='count', color='outcome', facet_col='gender',
        labels={'count': 'Number of Patients', 'age_group': 'Age Group', 'gender': 'Gender', 'outcome': 'Outcome'},
        color_discrete_map={'Positive': '#f97316', 'Negative': '#0ea5e9'})
    fig.update_layout(plot_bgcolor=colors['card'], paper_bgcolor=colors['card'], font_color=colors['text'], margin=dict(t=20))
    return fig

def create_blood_pressure_chart(df):
    if df.empty: return go.Figure().update_layout(title='No data to display')
    fig = px.scatter(df, x='systolic_bp', y='diastolic_bp', color='outcome',
        color_discrete_map={'Positive': '#f97316', 'Negative': '#0ea5e9'},
        labels={'systolic_bp': 'Systolic BP (mmHg)', 'diastolic_bp': 'Diastolic BP (mmHg)', 'outcome': 'Outcome'})
    fig.add_hline(y=90, line_dash="dash", line_color="gray", annotation_text="Diastolic Threshold (90)")
    fig.add_vline(x=140, line_dash="dash", line_color="gray", annotation_text="Systolic Threshold (140)")
    fig.update_layout(plot_bgcolor=colors['card'], paper_bgcolor=colors['card'], font_color=colors['text'], margin=dict(t=20))
    return fig

def create_biomarkers_chart(df, fold_change_troponin):
    if df.empty: return go.Figure().update_layout(title='No data to display')
    biomarkers_median = df.groupby('outcome')[['ck_mb', 'troponin']].median().reset_index()
    biomarkers_melted = biomarkers_median.melt(id_vars='outcome', value_vars=['ck_mb', 'troponin'], var_name='biomarker', value_name='median_value')
    subtitle = f"Median Troponin is ~{fold_change_troponin:.1f}x higher in positive cases" if fold_change_troponin is not None else ""
    fig = px.bar(biomarkers_melted, x='biomarker', y='median_value', color='outcome', barmode='group',
        labels={'median_value': 'Median Level (log scale)', 'biomarker': 'Biomarker', 'outcome': 'Outcome'},
        color_discrete_map={'Positive': '#f97316', 'Negative': '#0ea5e9'}, log_y=True, text_auto='.2s')
    fig.update_layout(plot_bgcolor=colors['card'], paper_bgcolor=colors['card'], font_color=colors['text'], margin=dict(t=40),
        annotations=[go.layout.Annotation(text=subtitle, align='left', showarrow=False, xref='paper', yref='paper', x=0, y=1.1)])
    return fig

# ==============================================================================
# 3. DASH APP LAYOUT
# ==============================================================================
app = Dash(__name__, external_scripts=['https://cdn.tailwindcss.com', 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css'])
server = app.server

# Configuration for export button
graph_config = {'displaylogo': False, 'toImageButtonOptions': {'format': 'png', 'filename': 'chart_export', 'height': 600, 'width': 800, 'scale': 2}}

app.layout = html.Div(id='top', className=f'font-sans antialiased', style={'backgroundColor': colors['background']}, children=[
    
    dcc.Store(id='filter-store', data={'age_range': initial_age_range, 'gender': initial_gender}),

    html.Header(className='bg-slate-900 text-white p-4 text-center shadow-lg', children=[
        html.H1('Heart Attack Risk Factor Dashboard', className='text-3xl font-bold tracking-tight'),
    ]),

    # --- Sticky Navigation Bar ---
    html.Nav(className='sticky top-0 z-50 bg-white shadow-md', children=[
        html.Div(className='container mx-auto px-4 py-2 flex justify-between items-center', children=[
            html.Div(className='flex space-x-4', children=[
                html.A('Demographics', href='#demographics-card', className='text-slate-600 hover:text-sky-500 font-semibold'),
                html.A('Blood Pressure', href='#bp-card', className='text-slate-600 hover:text-sky-500 font-semibold'),
                html.A('Biomarkers', href='#biomarkers-card', className='text-slate-600 hover:text-sky-500 font-semibold'),
            ]),
            html.Div(className='flex items-center', children=[
                dcc.Checklist(options=[{'label': ' Sync Filters', 'value': 'sync'}], value=[], id='sync-switch', className='flex items-center', inputClassName='mr-2'),
            ])
        ])
    ]),
    
    html.Main(className='container mx-auto p-4 md:p-8', children=[
        
        # --- VISUALIZATION CARD 1: Demographics ---
        html.Section(id='demographics-card', className='bg-white p-6 rounded-xl shadow-lg mb-8', children=[
            html.H2(children=[icon_demographics, 'Outcome Distribution by Demographics'], className='text-2xl font-bold text-slate-800 flex items-center'),
            html.P('Analyze how patient outcomes are distributed across different age groups and genders.', className='text-slate-500 mt-2 mb-6'),
            html.Div(className='grid grid-cols-1 md:grid-cols-2 gap-6 border-t pt-6', children=[
                html.Div([html.Label('Age Range', className='font-semibold text-gray-700 block mb-2'), dcc.RangeSlider(id='age-slider-1', min=initial_age_range[0], max=initial_age_range[1], step=1, value=initial_age_range, marks=None, tooltip={"placement": "bottom", "always_visible": True})]),
                html.Div([html.Label('Gender', className='font-semibold text-gray-700 block mb-2'), dcc.Dropdown(id='gender-dropdown-1', options=[{'label': 'All', 'value': 'All'}, {'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}], value=initial_gender, clearable=False)]),
            ]),
            dcc.Loading(children=dcc.Graph(id='age-gender-chart', className='mt-4', config=graph_config))
        ]),
        
        # --- VISUALIZATION CARD 2: Blood Pressure ---
        html.Section(id='bp-card', className='bg-white p-6 rounded-xl shadow-lg mb-8', children=[
            html.H2(children=[icon_blood_pressure, 'Blood Pressure vs. Outcome'], className='text-2xl font-bold text-slate-800 flex items-center'),
            html.P("Explore the relationship between systolic/diastolic blood pressure and patient outcomes.", className='text-slate-500 mt-2 mb-6'),
            html.Div(className='grid grid-cols-1 md:grid-cols-2 gap-6 border-t pt-6', children=[
                html.Div([html.Label('Age Range', className='font-semibold text-gray-700 block mb-2'), dcc.RangeSlider(id='age-slider-2', min=initial_age_range[0], max=initial_age_range[1], step=1, value=initial_age_range, marks=None, tooltip={"placement": "bottom", "always_visible": True})]),
                html.Div([html.Label('Gender', className='font-semibold text-gray-700 block mb-2'), dcc.Dropdown(id='gender-dropdown-2', options=[{'label': 'All', 'value': 'All'}, {'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}], value=initial_gender, clearable=False)]),
            ]),
            dcc.Loading(children=dcc.Graph(id='blood-pressure-chart', className='mt-4', config=graph_config))
        ]),

        # --- VISUALIZATION CARD 3: Biomarkers ---
        html.Section(id='biomarkers-card', className='bg-white p-6 rounded-xl shadow-lg mb-8', children=[
            html.H2(children=[icon_biomarkers, 'Cardiac Biomarker Analysis'], className='text-2xl font-bold text-slate-800 flex items-center'),
            html.P("Compare median levels of key cardiac biomarkers (CK-MB and Troponin).", className='text-slate-500 mt-2 mb-6'),
            html.Div(className='grid grid-cols-1 md:grid-cols-2 gap-6 border-t pt-6', children=[
                html.Div([html.Label('Age Range', className='font-semibold text-gray-700 block mb-2'), dcc.RangeSlider(id='age-slider-3', min=initial_age_range[0], max=initial_age_range[1], step=1, value=initial_age_range, marks=None, tooltip={"placement": "bottom", "always_visible": True})]),
                html.Div([html.Label('Gender', className='font-semibold text-gray-700 block mb-2'), dcc.Dropdown(id='gender-dropdown-3', options=[{'label': 'All', 'value': 'All'}, {'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}], value=initial_gender, clearable=False)]),
            ]),
            dcc.Loading(children=dcc.Graph(id='biomarkers-chart', className='mt-4', config=graph_config))
        ]),
    ]),
    
    # --- Back to Top Button ---
    html.A(html.I(className="fas fa-arrow-up"), href='#top', className='fixed bottom-8 right-8 bg-sky-500 text-white w-12 h-12 rounded-full flex items-center justify-center shadow-lg hover:bg-sky-600 transition-all')
])

# ==============================================================================
# 4. DASH CALLBACKS
# ==============================================================================

# --- Callback for Filter Synchronization ---
@app.callback(
    Output('filter-store', 'data'),
    Output('age-slider-1', 'value'), Output('gender-dropdown-1', 'value'),
    Output('age-slider-2', 'value'), Output('gender-dropdown-2', 'value'),
    Output('age-slider-3', 'value'), Output('gender-dropdown-3', 'value'),
    Input('age-slider-1', 'value'), Input('gender-dropdown-1', 'value'),
    Input('age-slider-2', 'value'), Input('gender-dropdown-2', 'value'),
    Input('age-slider-3', 'value'), Input('gender-dropdown-3', 'value'),
    State('sync-switch', 'value'),
    prevent_initial_call=True
)
def sync_filters(age1, gen1, age2, gen2, age3, gen3, sync_on):
    if not sync_on:
        return no_update

    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    age_map = {'age-slider-1': age1, 'age-slider-2': age2, 'age-slider-3': age3}
    gender_map = {'gender-dropdown-1': gen1, 'gender-dropdown-2': gen2, 'gender-dropdown-3': gen3}

    new_age_range = age_map.get(trigger_id, initial_age_range)
    new_gender = gender_map.get(trigger_id, initial_gender)
    
    new_store_data = {'age_range': new_age_range, 'gender': new_gender}

    return (new_store_data,
            new_age_range, new_gender,
            new_age_range, new_gender,
            new_age_range, new_gender)


# --- Callbacks for updating charts ---
@app.callback(
    Output('age-gender-chart', 'figure'),
    Input('age-slider-1', 'value'), Input('gender-dropdown-1', 'value')
)
def update_age_gender_visual(age_range, gender_value):
    filtered_df = filter_data(df_full, 'age', age_range)
    if gender_value != 'All': filtered_df = filter_data(filtered_df, 'gender', gender_value)
    return create_age_gender_chart(filtered_df)

@app.callback(
    Output('blood-pressure-chart', 'figure'),
    Input('age-slider-2', 'value'), Input('gender-dropdown-2', 'value')
)
def update_blood_pressure_visual(age_range, gender_value):
    filtered_df = filter_data(df_full, 'age', age_range)
    if gender_value != 'All': filtered_df = filter_data(filtered_df, 'gender', gender_value)
    return create_blood_pressure_chart(filtered_df)

@app.callback(
    Output('biomarkers-chart', 'figure'),
    Input('age-slider-3', 'value'), Input('gender-dropdown-3', 'value')
)
def update_biomarkers_visual(age_range, gender_value):
    filtered_df = filter_data(df_full, 'age', age_range)
    if gender_value != 'All': filtered_df = filter_data(filtered_df, 'gender', gender_value)
    df_pos = filtered_df[filtered_df['outcome'] == 'Positive']; df_neg = filtered_df[filtered_df['outcome'] == 'Negative']
    fold_change_trop = None
    if not df_pos.empty and not df_neg.empty and df_neg['troponin'].median() > 0:
        fold_change_trop = df_pos['troponin'].median() / df_neg['troponin'].median()
    return create_biomarkers_chart(filtered_df, fold_change_trop)

# ==============================================================================
# 5. App entrypoint
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True)
