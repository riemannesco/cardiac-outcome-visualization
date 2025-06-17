from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

# Assuming preprocess.py is in the same directory
try:
    from preprocess import filter_data
except ImportError:
    print("ERROR: 'preprocess.py' not found. Please place it in the same directory as app.py.")
    # Define a fallback function to prevent the app from crashing
    def filter_data(df, col, val):
        return df

# ==============================================================================
# 1. DATA LOADING AND PREPARATION
# ==============================================================================

# Construct a robust path to the data file
try:
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'assets', 'data', 'Medicaldataset.csv')
    df_full = pd.read_csv(csv_path)
    
    # Rename columns to ensure consistency
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
    
    # Convert numerical gender (0/1) to string values ("Female"/"Male")
    if 'gender' in df_full.columns and pd.api.types.is_numeric_dtype(df_full['gender']):
        df_full['gender'] = df_full['gender'].map({0: 'Female', 1: 'Male'})

except FileNotFoundError:
    print(f"ERROR: CSV file not found at expected path: {os.path.abspath(csv_path if 'csv_path' in locals() else './assets/data/Medicaldataset.csv')}")
    # Create a dummy dataframe to allow the app to start without data
    df_full = pd.DataFrame({
        'age': [0], 'gender': [''], 'heart_rate': [0], 'systolic_bp': [0], 'diastolic_bp': [0],
        'blood_sugar': [0], 'ck_mb': [0], 'troponin': [0], 'outcome': ['']
    })


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
    """Creates the outcome distribution bar chart by age and gender."""
    if df.empty:
        return go.Figure().update_layout(title='No data to display')
    
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
    fig.update_layout(plot_bgcolor=colors['card'], paper_bgcolor=colors['card'], font_color=colors['text'])
    return fig

def create_blood_pressure_chart(df):
    """Creates the systolic vs. diastolic blood pressure scatter plot."""
    if df.empty:
        return go.Figure().update_layout(title='No data to display')
    fig = px.scatter(
        df, x='systolic_bp', y='diastolic_bp', color='outcome',
        color_discrete_map={'Positive': colors['positive'], 'Negative': colors['negative']},
        labels={'systolic_bp': 'Systolic BP (mmHg)', 'diastolic_bp': 'Diastolic BP (mmHg)'},
        title='Systolic vs. Diastolic Blood Pressure'
    )
    fig.update_layout(plot_bgcolor=colors['card'], paper_bgcolor=colors['card'], font_color=colors['text'])
    return fig

def create_biomarkers_chart(df):
    if df.empty: return go.Figure().update_layout(title='No data to display')
    biomarkers_median = df.groupby('outcome')[['ck_mb', 'troponin']].median().reset_index()
    biomarkers_melted = biomarkers_median.melt(id_vars='outcome', value_vars=['ck_mb', 'troponin'], var_name='biomarker', value_name='median_value')
    fig = px.bar(biomarkers_melted, x='biomarker', y='median_value', color='outcome', barmode='group',
        labels={'median_value': 'Median Level (log scale)', 'biomarker': 'Biomarker', 'outcome': 'Outcome'},
        color_discrete_map={'Positive': colors['positive'], 'Negative': colors['negative']}, title='Biomarker Levels by Outcome', log_y=True)
    fig.update_layout(plot_bgcolor=colors['card'], paper_bgcolor=colors['card'], font_color=colors['text'])
    return fig

# ==============================================================================
# 3. DASH APP LAYOUT
# ==============================================================================
app = Dash(__name__, external_scripts=['https://cdn.tailwindcss.com'])
server = app.server

app.layout = html.Div(className='bg-slate-100 font-sans', children=[
    
    html.Header(className='bg-slate-800 text-white p-6 text-center shadow-lg', children=[
        html.H1('Exploring Heart Attack Risk Factors', className='text-3xl md:text-4xl font-bold mb-2'),
        html.P('An interactive analysis of medical indicators and their correlation with cardiac outcomes.', className='text-slate-300 max-w-3xl mx-auto')
    ]),
    
    html.Main(className='container mx-auto p-4 md:p-8', children=[
        
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

        html.Section(id='demographics', className='bg-white p-6 rounded-lg shadow-md mb-8', children=[
            html.H2('Section 1: Demographics and Vital Signs', className='text-2xl font-bold mb-4 text-slate-700'),
            dcc.Graph(id='age-gender-chart'),
            html.Hr(className='my-6'),
            dcc.Graph(id='blood-pressure-chart')
        ]),
        
        html.Section(id='biomarkers', className='bg-white p-6 rounded-lg shadow-md mb-8', children=[
            html.H2('Section 2: The Impact of Cardiac Biomarkers', className='text-2xl font-bold mb-4 text-slate-700'),
            dcc.Graph(id='biomarkers-chart'),
        ]),
    ]),
    
    html.Footer(className='text-center p-6 bg-slate-800 text-slate-400 text-sm', children=[
        html.P("INF8808E - Data Visualization Project | Heart Attack Analysis")
    ])
])

# ==============================================================================
# 4. DASH CALLBACK (The heart of the interactivity)
# ==============================================================================
@app.callback(
    [Output('age-gender-chart', 'figure'),
     Output('blood-pressure-chart', 'figure'),
     Output('biomarkers-chart', 'figure')],
    [Input('age-range-slider', 'value'),
     Input('gender-dropdown', 'value')]
)
def update_graphs(age_range, gender_value):
    """
    This master callback updates all graphs based on the filter values.
    It uses the filter_data function from preprocess.py.
    """
    filtered_df = df_full.copy()
    
    filtered_df = filter_data(filtered_df, 'age', (age_range[0], age_range[1]))
    
    if gender_value != 'All':
        filtered_df = filter_data(filtered_df, 'gender', gender_value)
        
    fig_age_gender = create_age_gender_chart(filtered_df)
    fig_bp = create_blood_pressure_chart(filtered_df)
    fig_biomarkers = create_biomarkers_chart(filtered_df)
    
    return fig_age_gender, fig_bp, fig_biomarkers

# ==============================================================================
# 5. App entrypoint
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True)

