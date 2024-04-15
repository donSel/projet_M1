from flask import Flask, send_file, render_template
import os

app = Flask(__name__)

#@app.route('/')
#def index():
#    return render_template('index2.html')

@app.route('/')
def generate_image():
    # Insérez ici le code Python pour générer l'image
    from keras.layers import Input
    from keras.models import load_model
    import matplotlib.pyplot as plt
    import random
    import numpy as np
    from models import GAN, Model_GAN
    
    # Définition de la fonction noise
    def noise(n):
        return np.random.uniform(-1.0, 1.0, size=[n, 4096])
    
    # Instancier la classe GAN
    gan_instance = GAN()
    
    # Construire le modèle génératif (sans spécifier la forme des données en entrée)
    generator_model = gan_instance.generator()
    
    # Charger les poids du modèle
    generator_model.load_weights("models/gen413.h5")
    
    # Générer une image avec le modèle chargé
    generated_image = generator_model.predict(noise(1))  # Génère une seule image
        
    # Définition du chemin du dossier où enregistrer l'image
    output_folder = "static/images"
    
    # Vérifier si le dossier existe, sinon le créer
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Enregistrer l'image générée dans le dossier
    output_path = os.path.join(output_folder, "generated_image.png")
    plt.imsave(output_path, generated_image[0])
    
    # Enregistrez l'image dans un emplacement accessible
    #image_path = "generated_images/generated_image.png"
    
    # Renvoyer le fichier image au navigateur
    #return send_file(output_path, mimetype='image/png')
    #return output_path
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)



