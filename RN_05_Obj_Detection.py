import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from random import randint
import zipfile
import tempfile

# Criar um diretório temporário para extrair os arquivos do zip
with tempfile.TemporaryDirectory() as tmp_dir:
    # Extrair o conteúdo do arquivo zip para o diretório temporário
    with zipfile.ZipFile('dados/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.zip', 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

    # Carregar o modelo TensorFlow SavedModel do diretório extraído
    model = tf.saved_model.load(tmp_dir)

#Carregar a imagem
image = Image.open("dados/detect.jpg")
#Converter a imagem para um array numpy
image_np = np.array(image)
#Exibir a imagem usando matplotlib
plt.subplot(121)
plt.imshow(image_np)
plt.title("Imagem Original")
plt.axis('off')  # Desligar os eixos

# Converte a imagem numpy em um tensor e adiciona uma dimensão extra (batch)
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)

detection = model(input_tensor)# Faz a detecção de objetos usando o modelo TensorFlow SavedModel

#Analise os resultados da detecção
boxes = detection['detection_boxes'].numpy()
classes = detection['detection_classes'].numpy().astype(int)
scores = detection['detection_scores'].numpy()

# Define os labels para as classes de objetos detectadas pelo modelo
labels = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
          'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
          'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
          'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
          'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
          'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
          'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
          'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
          'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

for i in range(classes.shape[1]):
    class_id = int(classes[0, i])
    score = scores[0, i]

    if np.any(score > 0.5):  # Filter out low-confidence detections
        h, w, _ = image_np.shape
        ymin, xmin, ymax, xmax = boxes[0, i]

        # Convert normalized coordinates to image coordinates
        xmin = int(xmin * w)
        xmax = int(xmax * w)
        ymin = int(ymin * h)
        ymax = int(ymax * h)

        # Get the class name from the labels list
        class_name = labels[class_id]

        random_color = (randint(0, 256), randint(0, 256), randint(0, 256))

        # Draw bounding box and label on the image
        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), random_color, 2)
        label = f"Class: {class_name}, Score: {score:.2f}"
        cv2.putText(image_np, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, random_color, 2)

# Display the result
plt.subplot(122)
plt.imshow(image_np)
plt.axis('off')
plt.title("Imagem com Object Detection")
plt.tight_layout()  # Ajusta automaticamente a disposição das figuras
plt.show()

#https://www.geeksforgeeks.org/object-detection-using-tensorflow/