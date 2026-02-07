import cv2
import os


class ReconocimientoFacial:

    def __init__(self, data_path="./Data", model_path="modeloLBPHFace.xml"):
        
        self.data_path = os.path.abspath(data_path)
        self.model_path = model_path
        self.face_recognizer = None
        self.face_classif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.image_paths = []
        self.modelo_cargado = False

        self.cargar_modelo()

    def cargar_modelo(self):
        """Intenta cargar el modelo y leer las etiquetas de los usuarios"""
        if not os.path.exists(self.data_path):
            print(f"Advertencia: La ruta de datos {self.data_path} no existe.")
            return
        # Obtener lista de usuarios (nombres de carpetas)
        self.image_paths = os.listdir(self.data_path)
        print('Usuarios encontrados:', self.image_paths)
        # Inicializar el reconocedor
        try:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            print("Error: cv2.face no está disponible. Asegúrate de instalar 'opencv-contrib-python'.")
            return
        # Cargar el modelo si existe
        if os.path.exists(self.model_path):
            try:
                self.face_recognizer.read(self.model_path)
                self.modelo_cargado = True
                print(f"Modelo cargado exitosamente desde {self.model_path}")
            except Exception as e:
                print(f"Error al leer el modelo: {e}")
                self.modelo_cargado = False
        else:
            print(f"Advertencia: No se encontró el archivo de modelo {self.model_path}")
            self.modelo_cargado = False

    def procesar_frame(self, frame):
        """
        Recibe un frame, detecta rostros y trata de reconocerlos.
        Retorna el frame procesado con las anotaciones.
        """
        if not self.modelo_cargado:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aux_frame = gray.copy()
        faces = self.face_classif.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            rostro = aux_frame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

            try:
                result = self.face_recognizer.predict(rostro)

                # Mostrar confianza en pantalla
                cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

                # LBPHFace threshold
                if result[1] < 120:
                    nombre_usuario = self.image_paths[result[0]] if result[0] < len(self.image_paths) else "IndexError"
                    cv2.putText(frame, '{}'.format(nombre_usuario), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            except Exception as e:
                print(f"Error en predicción: {e}")
        return frame
