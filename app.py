from huggingface_hub import hf_hub_download
import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
# Télécharger le fichier du modèle
model_path = hf_hub_download(
    repo_id="ghifariaulia/mobilenetv2-trashnet", filename="mobilenetv2-trashnet.h5")

# Charger le modèle
model = tf.keras.models.load_model(model_path)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def preprocess_image(image):
    # Redimensionner l'image à la taille attendue par le modèle
    img = image.resize((224, 224))
    # Convertir l'image en tableau numpy
    img_array = img_to_array(img)
    # Normaliser les pixels
    img_array = img_array / 255.0
    # Ajouter une dimension pour le batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(image):
    # Prétraiter l'image
    img_array = preprocess_image(image)
    # Faire la prédiction
    predictions = model.predict(img_array)[0]
    # Obtenir les probabilités pour chaque classe
    confidences = {class_names[i]: float(
        predictions[i]) for i in range(len(class_names))}
    # Obtenir la classe prédite
    predicted_class = np.argmax(predictions)
    class_label = class_names[predicted_class]
    # Obtenir le conseil de recyclage
    tip = recycling_tips[class_label]
    # Retourner les informations
    return confidences, f"Catégorie prédite : {class_label}\nConseil : {tip}"


recycling_tips = {
    'cardboard': 'Pliez le carton et évitez de le mouiller avant de le recycler.',
    'glass': 'Retirez les bouchons et rincez les contenants en verre.',
    'metal': 'Rincez les canettes et boîtes métalliques avant de les recycler.',
    'paper': 'Évitez de recycler le papier sale ou gras.',
    'plastic': 'Vérifiez le symbole de recyclage et triez en conséquence.',
    'trash': 'Cet article n\'est pas recyclable, veuillez le jeter correctement.',
}

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Probabilités"),
        gr.Textbox(label="Conseil de recyclage")
    ],
    examples=[

        ["examples/plastic.png"]
    ],
    title="Classification de déchets avec MobileNetV2",
    description="Téléchargez une image de déchet pour connaître sa catégorie et obtenir des conseils de recyclage.",
    flagging_mode="never"
)

# Lancer l'application
iface.launch()
