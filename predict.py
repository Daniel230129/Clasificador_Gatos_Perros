import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Desactivar advertencias innecesarias de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def main():
    parser = argparse.ArgumentParser(description='Clasificador de Perros y Gatos con Deep Learning')
    parser.add_argument('--image', type=str, required=True, help='Ruta de la imagen a predecir (.jpg o .png)')
    parser.add_argument('--model', type=str, required=True, help='Ruta del modelo guardado (.h5)')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: No se encontró la imagen en: {args.image}")
        return
    if not os.path.exists(args.model):
        print(f"Error: No se encontró el modelo en: {args.model}")
        return

    print(f"Cargando el modelo...")
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # Tamaño exacto con el que se entrenó el modelo original
    TARGET_SIZE = (224, 224) 
    
    print(f"Procesando la imagen {args.image}...")
    img = image.load_img(args.image, target_size=TARGET_SIZE)
    img_array = image.img_to_array(img)
    
    # Expandir dimensiones (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    print("Realizando predicción...")
    # verbose=0 oculta la barra de progreso en la terminal para una salida más limpia
    prediccion = model.predict(img_batch, verbose=0)
    
    clases = ["Gato", "Perro"]
    indice_ganador = np.argmax(prediccion[0])
    clase_predicha = clases[indice_ganador]
    confianza = prediccion[0][indice_ganador] * 100

    # Salida requerida por la tarea
    print("\n" + "="*40)
    print(f" CLASE PREDICHA: {clase_predicha}")
    print(f" CONFIANZA:      {confianza:.2f}%")
    print("="*40 + "\n")

    # Visualización requerida por la tarea
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicción: {clase_predicha} (Confianza: {confianza:.2f}%)", fontsize=14, fontweight='bold')
    plt.show()

if __name__ == '__main__':
    main()