import cv2
import os
import imutils


class CapturaRostros:

    def __init__(self, nombre_persona, ruta_base="./Data"):
        self.nombre_persona = nombre_persona
        self.data_path = os.path.abspath(ruta_base)
        self.person_path = os.path.join(self.data_path, self.nombre_persona)
        self.max_fotos = 200
        self.count = 0
        self.capturando = True

        # Cargar clasificado
        self.face_classif = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Crear carpeta si no existe
        if not os.path.exists(self.person_path):
            print(f'Carpeta creada: {self.person_path}')
            os.makedirs(self.person_path)
        else:
            print(f'Usando carpeta existente: {self.person_path}')

    def procesar_frame(self, frame):
        # Procesa un único frame: detecta rostro, guarda imagen y dibuja rectángulo.

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not self.capturando:
            return frame, False

        if self.count >= self.max_fotos:
            self.capturando = False
            return frame, False
        # Copia para recortar el rostro limpio
        aux_frame = frame.copy()

        #cv2.imshow("VIDEO", frame)

        # Preprocesamiento
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #cv2.imshow("GRAY", gray)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detección
        faces = self.face_classif.detectMultiScale(gray, 1.3, 5)

        # Procesar rostros
        for (x, y, w, h) in faces:
            # Dibujar Rectángulos
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extraer rostro de la copia limpia
            rostro = aux_frame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

            # Guardar imagen
            nombre_archivo = os.path.join(
                self.person_path,
                f'rostro_{self.count}.jpg'
            )
            cv2.imwrite(nombre_archivo, rostro)
            self.count += 1
        # Mostrar progreso en el frame
        texto = f'Capturando: {self.count}/{self.max_fotos}'
        cv2.putText(
            frame, texto, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        return frame, self.capturando

    def obtener_estadisticas(self):
        return {
            'nombre_persona': self.nombre_persona,
            'ruta_almacenamiento': self.person_path,
            'total_imagenes': self.count,
            'max_imagenes': self.max_fotos
        }
