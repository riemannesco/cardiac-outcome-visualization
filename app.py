# Import des bibliothèques nécessaires depuis Dash
from dash import Dash, html

# 1. Initialisation de l'application Dash
#    Dash est un framework pour construire des applications web analytiques.
app = Dash(__name__)

# 2. CRUCIAL POUR LE DÉPLOIEMENT SUR RENDER
#    Le serveur de production (Gunicorn) a besoin de trouver cette variable 'server'.
#    On assigne le serveur Flask sous-jacent de l'application Dash à cette variable.
server = app.server

# 3. Définition de la mise en page (layout) de l'application
#    C'est ici que vous décrivez à quoi votre application va ressembler.
#    On utilise des composants HTML (ici, html.H1 et html.P).
app.layout = html.Div(children=[
    html.H1(
        children='Mon application Dash est déployée avec succès !',
        style={'textAlign': 'center', 'color': '#007BFF'}
    ),

    html.P(
        children='Si vous voyez ce message, votre déploiement continu depuis GitHub vers Render fonctionne.',
        style={'textAlign': 'center'}
    )
])

# 4. Point d'entrée pour lancer l'application en local
#    Cette partie n'est exécutée que lorsque vous lancez le script directement
#    avec `python app.py`. Elle n'est pas utilisée par Gunicorn sur Render.
if __name__ == '__main__':
    app.run_server(debug=True)
