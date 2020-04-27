# Heimdall-EYE USO:
# python detect_age.py --image images/antonio.png --face face_detector --age age_detector

# Importamos los paquetes necesarios 
import numpy as np
import argparse
import cv2
import os

# Contruimos los argumentos que vamos a utilizar y analizamos estos argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--face", required=True,
	help="path to face detector model directory")
ap.add_argument("-a", "--age", required=True,
	help="path to age detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Definimos los rangos de edad en los que va a detectar
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
	"(38-43)", "(48-53)", "(60-100)"]

# Carga nuestro modelo detector facial desde la carpeta
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carga nuestro modelo detectar de edad de nuestro disco
print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Cargar la imagen de entrada y contruimos un blob de entrada para la imagen
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# Pasar la gota a traves de la red y obtener las detecciones faciales 
print("[INFO] computing face detections...")
faceNet.setInput(blob)
detections = faceNet.forward()

# Bucle para la deteccion de edad
for i in range(0, detections.shape[2]):
	# Extraer la cofianza ( es decir la probabilidad asociada a la prediccion)
	confidence = detections[0, 0, i, 2]

	# Filtramos las detecciones debiles asegurando que la cofianza es mayor que la confianza minima
	if confidence > args["confidence"]:
		# Calculamos las coordenadas (x,y) del cuadro delimitador para el objeto
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# Extraemos el ROI de la cara y luego contruimos solo el ROI de la Cara
		face = image[startY:endY, startX:endX]
		faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
			(78.4263377603, 87.7689143744, 114.895847746),
			swapRB=False)

		# Hacer predicciones esobre la edad y encontrar un grupo que sea la mayor probalidad al rostros
		ageNet.setInput(faceBlob)
		preds = ageNet.forward()
		i = preds[0].argmax()
		age = AGE_BUCKETS[i]
		ageConfidence = preds[0][i]

		# Mostramos la prediccion de edad por el terminal
		text = "{}: {:.2f}%".format(age, ageConfidence * 100)
		print("[INFO] {}".format(text))

		# Dibujamos el cuadro delimitador de la cara junto con el asociado 
		# Edad prevista
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Mostamos la imagen en el Frame
cv2.imshow("Image", image)
cv2.waitKey(0)
