# projet_M1
Interface web du projet M1

## web_interface
Dossier cotenant le code de l'interface web qui permet de générer des images avec les modèles entraînés (ou réentraînés) grâce aux fichier ipynb dans le dossier "training_model_fine_tuning"
Pour voir l'interface web il faut exécuter le "app2.py" et ouvrir le lien http affiché dans la console dans le navigateur

## training_model_fine_tuning
Le fichier "Copie_de_GAN256.ipynb" permet d'entraîner et de seauvegarder les poids du réseau de neurone en stockant le tout sur Google Drive. Pour cela il est nécessaire de bien spécifier les chemins des dossier de seauvegarde et contenant les images d'entraînement dans le code.
Pour l'exécution du code il est conseillé d'utiliser les unités de calculs payantes de Google Collab pour accélérer l'entraînement.
