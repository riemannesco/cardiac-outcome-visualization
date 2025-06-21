{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  # buildInputs contient les paquets systeme et les outils necessaires.
  buildInputs = [
    # On selectionne ici une version recente et stable de Python 3.
    # Note : Python 3.14 n'est pas encore disponible. Nous utilisons Python 3.12.
    # Le paquet python312 inclut deja `pip`.
    (pkgs.python312.withPackages (ps: [
      # ps (python.pkgs) est l'ensemble des paquets Python fournis par Nix.
      # On declare ici les memes librairies que dans votre requirements.txt.
      ps.dash
      ps.plotly
      ps.pandas
      ps.numpy
      ps.gunicorn
    ]))
  ];

  # Vous pouvez definir des variables d'environnement ici si necessaire.
  shellHook = ''
    # Ce message s'affichera chaque fois que vous entrerez dans l'environnement.
    echo "Bienvenue dans l'environnement de developpement du projet."
    echo "Les librairies Python (pandas, dash, etc.) sont pretes a etre utilisees."
  '';
}
