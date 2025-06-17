from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Assuming preprocess.py is in the same directory
try:
    from preprocess import filter_data
except ImportError:
    print("ERROR: 'preprocess.py' not found. Please place it in the same directory as app.py.")
    def filter_data(df, col, val):
        return df

# ==============================================================================
# 1. DATA LOADING AND PREPARATION
# ==============================================================================

try:
    df_full = pd.read_csv('./assets/data/Medicaldataset.csv')
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
    if 'gender' in df_full.columns and pd.api.types.is_numeric_dtype(df_full['gender']):
        df_full['gender'] = df_full['gender'].map({0: 'Female', 1: 'Male'})

except FileNotFoundError:
    print("ERROR: './assets/data/Medicaldataset.csv' not found.")
    df_full = pd.DataFrame()


# Define colors for the graphs
colors = {
    'positive': '#f97316',
    'negative': '#0ea5e9',
    'background': '#f0f4f8',
    'card': 'white',
    'text': '#334155',
    'grid': '#e2e8f0'
}

# ==============================================================================
# 2. Reusable Chart Creation Functions
# ==============================================================================

def create_age_gender_chart(df):
    if df.empty: return go.Figure().update_layout(title='No data to display')
    bins = [14, 30, 45, 60, 75, 90, float('inf')]
    labels = ['14-29', '30-44', '45-59', '60-74', '75-89', '90+']
    df_copy = df.copy()
    df_copy['age_group'] = pd.cut(df_copy['age'], bins=bins, labels=labels, right=False)
    grouped_df = df_copy.groupby(['age_group', 'gender', 'outcome'], observed=True).size().reset_index(name='count')
    
    fig = px.bar(
        grouped_df, x='age_group', y='count', color='outcome', facet_col='gender',
        labels={'count': 'Number of Patients', 'age_group': 'Age Group', 'gender': 'Gender', 'outcome': 'Outcome'},
        color_discrete_map={'Positive': colors['positive'], 'Negative': colors['negative']},
        title='Outcome Distribution by Age and Gender'
    )
    fig.update_layout(plot_bgcolor=colors['card'], paper_bgcolor=colors['card'], font_color=colors['text'], margin=dict(t=50, l=20, r=20, b=20))
    return fig

def create_blood_pressure_chart(df):
    if df.empty: return go.Figure().update_layout(title='No data to display')
    fig = px.scatter(
        df, x='systolic_bp', y='diastolic_bp', color='outcome',
        color_discrete_map={'Positive': colors['positive'], 'Negative': colors['negative']},
        labels={'systolic_bp': 'Systolic BP (mmHg)', 'diastolic_bp': 'Diastolic BP (mmHg)'},
        title='Systolic vs. Diastolic Blood Pressure'
    )
    fig.update_layout(plot_bgcolor=colors['card'], paper_bgcolor=colors['card'], font_color=colors['text'], margin=dict(t=50, l=20, r=20, b=20))
    return fig

def create_biomarkers_chart(df):
    if df.empty: return go.Figure().update_layout(title='No data to display')
    biomarkers_median = df.groupby('outcome')[['ck_mb', 'troponin']].median().reset_index()
    biomarkers_melted = biomarkers_median.melt(id_vars='outcome', value_vars=['ck_mb', 'troponin'], var_name='biomarker', value_name='median_value')
    fig = px.bar(biomarkers_melted, x='biomarker', y='median_value', color='outcome', barmode='group',
        labels={'median_value': 'Median Level (log scale)', 'biomarker': 'Biomarker', 'outcome': 'Outcome'},
        color_discrete_map={'Positive': colors['positive'], 'Negative': colors['negative']}, title='Biomarker Levels by Outcome', log_y=True)
    fig.update_layout(plot_bgcolor=colors['card'], paper_bgcolor=colors['card'], font_color=colors['text'], margin=dict(t=50, l=20, r=20, b=20))
    return fig

def create_combination_heatmap(df):
    if df.empty: return go.Figure().update_layout(title='No data to display')
    bs_bins = pd.cut(df['blood_sugar'], bins=[0, 100, 140, 200, float('inf')], labels=['<100', '100-140', '140-200', '>200'])
    bp_bins = pd.cut(df['systolic_bp'], bins=[0, 120, 140, 160, float('inf')], labels=['<120', '120-140', '140-160', '>160'])
    df_heatmap = df.copy()
    df_heatmap['bs_group'], df_heatmap['bp_group'] = bs_bins, bp_bins
    heatmap_data = df_heatmap.groupby(['bp_group', 'bs_group', 'outcome'], observed=True).size().unstack(fill_value=0)
    heatmap_data['total'] = heatmap_data.get('Positive', 0) + heatmap_data.get('Negative', 0)
    heatmap_data['positive_pct'] = (heatmap_data.get('Positive', 0) / heatmap_data['total']).fillna(0) * 100
    z_data = heatmap_data['positive_pct'].unstack()
    fig = go.Figure(data=go.Heatmap(z=z_data.values, x=z_data.columns, y=z_data.index, colorscale='Oranges',
        hovertemplate='Blood Sugar: %{x}<br>Systolic BP: %{y}<br>Positive Outcome: %{z:.0f}%<extra></extra>'))
    fig.update_layout(title='Interaction of Blood Sugar and Systolic Blood Pressure',
        xaxis_title='Blood Sugar (mg/dL)', yaxis_title='Systolic Blood Pressure (mmHg)',
        plot_bgcolor=colors['card'], paper_bgcolor=colors['card'], font_color=colors['text'])
    return fig

def create_profile_radar_chart(df):
    if df.empty or 'outcome' not in df or df['outcome'].nunique() < 2: return go.Figure().update_layout(title='Not enough data for comparison')
    features = ['age', 'heart_rate', 'systolic_bp', 'blood_sugar', 'ck_mb', 'troponin']
    df_normalized = df.copy()
    for feature in features:
        min_val, max_val = df_full[feature].min(), df_full[feature].max()
        df_normalized[feature] = (df[feature] - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0.5
    profile_means = df_normalized.groupby('outcome')[features].mean().reset_index()
    fig = go.Figure()
    for outcome in ['Positive', 'Negative']:
        if outcome in profile_means['outcome'].values:
            fig.add_trace(go.Scatterpolar(r=profile_means[profile_means['outcome'] == outcome][features].values.flatten(),
                theta=features, fill='toself', name=f'{outcome} Outcome', marker_color=colors[outcome.lower()]))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True,
        title='Average Patient Profile: Positive vs. Negative Outcome',
        paper_bgcolor=colors['card'], plot_bgcolor=colors['card'], font_color=colors['text'])
    return fig

# ==============================================================================
# 3. DASH APP LAYOUT (Section-based structure)
# ==============================================================================
app = Dash(__name__, external_scripts=['https://cdn.tailwindcss.com'])
server = app.server

app.layout = html.Div(className='bg-slate-100 font-sans', children=[
    
    html.Header(className='bg-slate-800 text-white p-6 text-center shadow-lg', children=[
        html.H1('Exploring Heart Attack Risk Factors', className='text-3xl md:text-4xl font-bold mb-2'),
        html.P('An interactive analysis of medical indicators and their correlation with cardiac outcomes.', className='text-slate-300 max-w-3xl mx-auto')
    ]),
    
    # --- Sticky Navigation ---
    html.Nav(className='sticky top-0 z-50 bg-slate-700/80 backdrop-blur-sm text-white shadow-md', children=[
        html.Ul(className='flex items-center justify-center space-x-2 sm:space-x-4 overflow-x-auto p-2 text-sm md:text-base', children=[
            html.Li(html.A('Demographics', href='#demographics', className='font-semibold py-2 px-3 rounded-md hover:bg-slate-600 transition-colors')),
            html.Li(html.A('Biomarkers', href='#biomarkers', className='font-semibold py-2 px-3 rounded-md hover:bg-slate-600 transition-colors')),
            html.Li(html.A('Risk Combinations', href='#combinations', className='font-semibold py-2 px-3 rounded-md hover:bg-slate-600 transition-colors')),
            html.Li(html.A('Patient Profiles', href='#profiles', className='font-semibold py-2 px-3 rounded-md hover:bg-slate-600 transition-colors')),
        ])
    ]),
    
    html.Main(className='container mx-auto p-4 md:p-8', children=[
        
        # --- Filter Controls Section ---
        html.Div(className='bg-white p-6 rounded-lg shadow-md mb-8', children=[
            html.H2('Filter Controls', className='text-xl font-bold mb-4 text-slate-700'),
            html.Div(className='grid grid-cols-1 md:grid-cols-2 gap-6', children=[
                html.Div([
                    html.Label('Age Range', className='font-semibold text-gray-700'),
                    dcc.RangeSlider(id='age-range-slider', min=df_full['age'].min(), max=df_full['age'].max(), step=1,
                        value=[df_full['age'].min(), df_full['age'].max()],
                        marks={i: str(i) for i in range(int(df_full['age'].min()), int(df_full['age'].max()) + 1, 10)},
                        tooltip={"placement": "bottom", "always_visible": True})
                ]),
                html.Div([
                    html.Label('Gender', className='font-semibold text-gray-700'),
                    dcc.Dropdown(id='gender-dropdown',
                        options=[{'label': 'All', 'value': 'All'}, {'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}],
                        value='All', clearable=False)
                ]),
            ])
        ]),

        # --- Graphs Section ---
        html.Section(id='demographics', className='bg-white p-6 rounded-lg shadow-md mb-8 scroll-mt-24', children=[
            html.H2('Section 1: Demographics and Vital Signs', className='text-2xl font-bold mb-4 text-slate-700'),
            dcc.Graph(id='age-gender-chart'),
            html.Hr(className='my-6'),
            dcc.Graph(id='blood-pressure-chart')
        ]),
        
        html.Section(id='biomarkers', className='bg-white p-6 rounded-lg shadow-md mb-8 scroll-mt-24', children=[
            html.H2('Section 2: The Impact of Cardiac Biomarkers', className='text-2xl font-bold mb-4 text-slate-700'),
            dcc.Graph(id='biomarkers-chart'),
        ]),

        html.Section(id='combinations', className='bg-white p-6 rounded-lg shadow-md mb-8 scroll-mt-24', children=[
            html.H2("Section 3: How Risk Factors Combine", className='text-2xl font-bold mb-4 text-slate-700'),
            dcc.Graph(id='combination-heatmap')
        ]),

        html.Section(id='profiles', className='bg-white p-6 rounded-lg shadow-md mb-8 scroll-mt-24', children=[
            html.H2("Section 4: Comparing Patient Profiles", className='text-2xl font-bold mb-4 text-slate-700'),
            dcc.Graph(id='profile-radar-chart')
        ]),
    ]),
    
    html.Footer(className='text-center p-6 bg-slate-800 text-slate-400 text-sm', children=[
        html.P("INF8808E - Data Visualization Project | Heart Attack Analysis")
    ])
])

# ==============================================================================
# 4. CALLBACKS
# ==============================================================================
@app.callback(
    [Output('age-gender-chart', 'figure'),
     Output('blood-pressure-chart', 'figure'),
     Output('biomarkers-chart', 'figure'),
     Output('combination-heatmap', 'figure'),
     Output('profile-radar-chart', 'figure')],
    [Input('age-range-slider', 'value'),
     Input('gender-dropdown', 'value')]
)
def update_graphs(age_range, gender_value):
    # Filter the data based on the global filter controls
    filtered_df = df_full.copy()
    if not filtered_df.empty:
        filtered_df = filter_data(filtered_df, 'age', (age_range[0], age_range[1]))
        if gender_value != 'All':
            filtered_df = filter_data(filtered_df, 'gender', gender_value)
    
    # Generate new figures with the filtered data
    fig_age_gender = create_age_gender_chart(filtered_df)
    fig_bp = create_blood_pressure_chart(filtered_df)
    fig_biomarkers = create_biomarkers_chart(filtered_df)
    fig_heatmap = create_combination_heatmap(filtered_df)
    fig_radar = create_profile_radar_chart(filtered_df)
    
    return fig_age_gender, fig_bp, fig_biomarkers, fig_heatmap, fig_radar

# ==============================================================================
# 5. SCRIPT ENTRYPOINT
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True)

