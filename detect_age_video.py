# Heimdall-EYE USO:
# python detect_age_video.py --face face_detector --age age_detector

# Importamos los paquetes o librerias necesarios
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_age(frame, faceNet, ageNet, minConf=0.5):
	# Definimos un rango de edades para la deteccion 
	AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
		"(38-43)", "(48-53)", "(60-100)"]

	# Inicializamos los valores en una lista
	results = []

	# Cogemos las diminsiones del marco y luego contruimos el blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# Pasamos el blob a traves de la red y obtenemos las detecciones faciales
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# Realizamos un bucle sobre las detecciones
	for i in range(0, detections.shape[2]):
		# Extraemos la confianza es decir la probabilidad asociada a la prediccion
		confidence = detections[0, 0, i, 2]

		# Filtramos las detecciones debiles asegurando la confianza , que sea mayor la confianza que la minima
		if confidence > minConf:
			# Calcular las coordenadas (x,y) del cuadro delimitador para el objeto
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Extraemos el ROI del rostro
			face = frame[startY:endY, startX:endX]

			# Aseguramos de que ROI de la cara sea lo suficiente grande
			if face.shape[0] < 20 or face.shape[1] < 20:
				continue

			# Contruimos el blob solo con el ROI de rostro
			faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
				(78.4263377603, 87.7689143744, 114.895847746),
				swapRB=False)

			# Hacemos la prediccion sobre la edad y encontramos un grupo donde tengamos esas edad , la mayor probabilidad siempre
			ageNet.setInput(faceBlob)
			preds = ageNet.forward()
			i = preds[0].argmax()
			age = AGE_BUCKETS[i]
			ageConfidence = preds[0][i]

			# Contruimos un diccionario que consista tanto en la cara
			# Como en la ubicacion del cuadro delimitador junto con la prediccion de la edad y luego se actualiza nuestra lista de resultados
			d = {
				"loc": (startX, startY, endX, endY),
				"age": (age, ageConfidence)
			}
			results.append(d)

	# Retornamos nuestros resultados a la funcion de lista
	return results

# Contruimos el analizador de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True,
	help="path to face detector model directory")
ap.add_argument("-a", "--age", required=True,
	help="path to age detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Cargamos nuestro modelo de detector facial serial de nuestro disco
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Cargamos nuestro reconocimiento de edad serial del disco
print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Inicializamos el sensor de la camara la la deteccion de edad en directo
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Recorremos los fotogramas de la transmision de video en directo
while True:
	# Tomamosel fotograma de la secuencia del video y cambiamos el tamaño para tener un ancho de pixeles de 400
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# Detectamos las caras en el Frame y predecimos su edad en la lista de edades
	results = detect_and_predict_age(frame, faceNet, ageNet,
		minConf=args["confidence"])

	# Realizamos un bucle para mostrar los resultados de la posible edad
	for r in results:
		# Dibujamos el cuadro delimitador de la cara junto con la edad prevista en el rango de edades
		text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
		(startX, startY, endX, endY) = r["loc"]
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# Mostramos el Frame en la pantalla
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Para poder salir del reconocimiento de edad solo nos basta con pulsar la q
	if key == ord("q"):
		break
		
# Realizamos una limpieza
cv2.destroyAllWindows()
vs.stop()
