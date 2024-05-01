from flask import Flask, send_file, render_template
import os
from keras.layers import Input
from keras.models import load_model
import matplotlib.pyplot as plt
import random
import os
import numpy as np
from models import GAN, Model_GAN


app = Flask(__name__)

#@app.route('/')
#def index():
#    return render_template('index2.html')

@app.route('/')
def generate_image():
    # Insérez ici le code Python pour générer l'image
    
    # Liste de tous les fichiers dans le dossier
    models_folder = "models"
    model_files = os.listdir(models_folder)
 
    # Définition de la fonction noise
        # Définition de la fonction noise
    def noise(n):
        return np.random.uniform(-1.0, 1.0, size=[n, 4096])
    
    # Instancier la classe GAN
    gan_instance = GAN()

    # Générer et enregistrer cinq images avec des modèles choisis aléatoirement
    for i in range(5):
        # Choix aléatoire d'un fichier .h5 parmi les modèles
        selected_model_file = random.choice(model_files)
        selected_model_path = os.path.join(models_folder, selected_model_file)
        
        # Choix aléatoire d'un chemin de poids de modèle
        #selected_model_path = random.choice(model_weights_paths)
        
        # Construire le modèle génératif (sans spécifier la forme des données en entrée)
        generator_model = gan_instance.generator()
        
        # Charger les poids du modèle
        generator_model.load_weights(selected_model_path)
        
        # Générer une image avec le modèle chargé
        generated_image = generator_model.predict(noise(1))  # Génère une seule image
        
        # Définition du chemin du dossier où enregistrer l'image
        output_folder = "static/images"
        
        # Vérifier si le dossier existe, sinon le créer
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Enregistrer l'image générée dans le dossier
        output_path = os.path.join(output_folder, f"generated_image_{i}.png")
        plt.imsave(output_path, generated_image[0])

            # Enregistrez l'image dans un emplacement accessible
    #image_path = "generated_images/generated_image.png"
    
    # Renvoyer le fichier image au navigateur
    #return send_file(output_path, mimetype='image/png')
    #return output_path
    return render_template('index2.html')
        
if __name__ == '__main__':
    app.run(debug=True)