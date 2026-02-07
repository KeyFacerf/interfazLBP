import tkinter as tk
from tkinter import Label, Button
import cv2 as cv
from PIL import Image, ImageTk
import imutils
from captura_rostros import CapturaRostros
from reconocimiento import ReconocimientoFacial
import recolectar_datos

class InterfazGUI:
    def __init__(self):
        self.vid = None
        self.frame = None
        self.nombre_usuario = "" 
        self.capturador = None 

        self.reconocedor = ReconocimientoFacial(data_path="./Data",
                                                model_path="modeloLBPHFace.xml")  # Instancia de reconocimiento facial

        # Crear ventana principal
        self.pantalla = tk.Tk()
        self.pantalla.title("Sistema de Video")
        self.pantalla.geometry("1024x640")
        self.pantalla.configure(bg="#4A7BA7")
        self.pantalla.resizable(False, False)

        # Frame contenedor principal con borde
        self.frame_principal = tk.Frame(self.pantalla, bg="#E8E8E8", bd=2, relief=tk.RIDGE)
        self.frame_principal.place(x=20, y=20, width=984, height=600)

        # Panel izquierdo para botones
        self.panel_botones = tk.Frame(self.frame_principal, bg="#E8E8E8")
        self.panel_botones.place(x=20, y=20, width=220, height=560)

        # 4 botones
        self.btn_open_camera = Button(
            self.panel_botones,
            text="Open Camera",
            font=("Arial", 14, "bold"),
            bg="#7FA8C9",
            fg="black",
            width=15,
            height=3,
            bd=0,
            cursor="hand2",
            command=self.open_camera
        )
        self.btn_open_camera.pack(pady=(10, 20))

        self.btn_close_camera = Button(
            self.panel_botones,
            text="Close Camera",
            font=("Arial", 14, "bold"),
            bg="#7FA8C9",
            fg="black",
            width=15,
            height=3,
            bd=0,
            cursor="hand2",
            command=self.close_camera
        )
        self.btn_close_camera.pack(pady=(0, 20))

        self.btn_new_user = Button(
            self.panel_botones,
            text="New User",
            font=("Arial", 14, "bold"),
            bg="#7FA8C9",
            fg="black",
            width=15,
            height=3,
            bd=0,
            cursor="hand2",
            command=self.new_user
        )
        self.btn_new_user.pack(pady=(0, 20))

        self.btn_data_process = Button(
            self.panel_botones,
            text="Data Process",
            font=("Arial", 14, "bold"),
            bg="#7FA8C9",
            fg="black",
            width=15,
            height=3,
            bd=0,
            cursor="hand2",
            command=self.data_process
        )
        self.btn_data_process.pack(pady=(0, 20))

        # Panel derecho para video
        self.panel_video = tk.Frame(self.frame_principal, bg="#CCCCCC", bd=3, relief=tk.SOLID)
        self.panel_video.place(x=260, y=50, width=700, height=500)
        self.panel_video.config(highlightbackground="#4A7BA7", highlightthickness=3)

        # Mostrar video
        self.lblVideo = Label(self.panel_video, bg="#CCCCCC")
        self.lblVideo.pack(fill=tk.BOTH, expand=True)

        # Texto "No Video" por defecto
        self.texto_no_video = Label(
            self.lblVideo,
            text="No Video",
            font=("Arial", 48, "bold"),
            bg="#CCCCCC",
            fg="white"
        )
        self.texto_no_video.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def open_camera(self):
        print("Botón 'Open Camera' presionado")

        # Inicializar la cámara si no está abierta
        if self.vid is None or not self.vid.isOpened():
            self.vid = cv.VideoCapture(0)

            if not self.vid.isOpened():
                print("Error: No se pudo abrir la cámara")
                return

            self.texto_no_video.place_forget()
            self.visualizar()
        else:
            print("La cámara ya está abierta")

    def close_camera(self):
        print("Botón 'Close Camera' presionado")

        # Cerrar la cámara si está abierta
        if self.vid is not None and self.vid.isOpened():
            self.vid.release()
            self.vid = None

            # Limpiar el label de video
            self.lblVideo.configure(image='')
            self.lblVideo.image = None

            # Mostrar nuevamente el texto "No Video"
            self.texto_no_video.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

            print("Cámara cerrada correctamente")
        else:
            print("La cámara no estaba abierta")

    def new_user(self):
        print("Botón 'New User' presionado")

        # Crear ventana de diálogo
        dialog = tk.Toplevel(self.pantalla)
        dialog.title("Registrar Nuevo Usuario")
        dialog.geometry("400x200")
        dialog.configure(bg="#E8E8E8")
        dialog.resizable(False, False)

        # Centrar la ventana de diálogo
        dialog.transient(self.pantalla)
        dialog.grab_set()

        # Título
        label_titulo = Label(
            dialog,
            text="Ingrese el nombre del usuario:",
            font=("Arial", 12, "bold"),
            bg="#E8E8E8",
            fg="black"
        )
        label_titulo.pack(pady=(20, 10))

        # Campo de entrada de texto
        entry_nombre = tk.Entry(
            dialog,
            font=("Arial", 14),
            width=25,
            bd=2,
            relief=tk.SOLID
        )
        entry_nombre.pack(pady=10)
        entry_nombre.focus()

        # Variable para guardar el nombre
        nombre_guardado = tk.StringVar()

        def guardar_nombre():
            nombre = entry_nombre.get().strip()

            if nombre:
                # Guardar el nombre en la variable de instancia
                self.nombre_usuario = nombre
                nombre_guardado.set(nombre)

                print(f"Usuario registrado: {self.nombre_usuario}")

                # Cerrar el diálogo
                dialog.destroy()

                # Inicializar captura de rostros integrada
                print(f"Preparando captura de rostros para: {self.nombre_usuario}")

                # Crear instancia de CapturaRostros
                self.capturador = CapturaRostros(
                    nombre_persona=self.nombre_usuario,
                    ruta_base="./Data"
                )

                # Verificar si la cámara está abierta, si no, abrirla
                if self.vid is None or not self.vid.isOpened():
                    print("Abriendo cámara para captura...")
                    self.open_camera()

                print("Captura iniciada. Mira a la cámara.")

            else:
                # Mostrar advertencia si está vacío
                label_error.config(text="Por favor ingrese un nombre")

        def cancelar()
            dialog.destroy()

        # Label para mensajes de error
        label_error = Label(
            dialog,
            text="",
            font=("Arial", 10),
            bg="#E8E8E8",
            fg="red"
        )
        label_error.pack(pady=5)

        # Frame para botones
        frame_botones = tk.Frame(dialog, bg="#E8E8E8")
        frame_botones.pack(pady=10)

        # Botón Guardar
        btn_guardar = Button(
            frame_botones,
            text="Guardar",
            font=("Arial", 11, "bold"),
            bg="#7FA8C9",
            fg="black",
            width=10,
            height=1,
            bd=0,
            cursor="hand2",
            command=guardar_nombre
        )
        btn_guardar.pack(side=tk.LEFT, padx=10)

        # Botón Cancelar
        btn_cancelar = Button(
            frame_botones,
            text="Cancelar",
            font=("Arial", 11, "bold"),
            bg="#CCCCCC",
            fg="black",
            width=10,
            height=1,
            bd=0,
            cursor="hand2",
            command=cancelar
        )
        btn_cancelar.pack(side=tk.LEFT, padx=10)

        # Permitir guardar con Enter
        entry_nombre.bind('<Return>', lambda event: guardar_nombre())

    def data_process(self):
        self.close_camera()
        print("Botón 'Data Process' presionado")
        recolectar_datos.recolectar()
        self.open_camera()

    def visualizar(self):
        if self.vid is not None and self.vid.isOpened():
            ret, self.frame = self.vid.read()
            if ret == True:
                # Convertir de BGR a RGB
                self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
                # Redimensionar frame
                self.frame = imutils.resize(self.frame, width=640)
                # Procesamiento de captura de rostros si está activo
                if self.capturador:
                    self.frame, capturando = self.capturador.procesar_frame(self.frame)

                    if not capturando:
                        print("Captura finalizada exitosamente")
                        stats = self.capturador.obtener_estadisticas()
                        print(f"Total imágenes: {stats['total_imagenes']}")
                        self.capturador = None

                else:
                    self.frame = self.reconocedor.procesar_frame(self.frame)

                # Convertir a formato compatible con tkinter (después de dibujar rectángulos)
                im = Image.fromarray(self.frame)
                img = ImageTk.PhotoImage(image=im)
                # Actualizar el label con la nueva imagen
                self.lblVideo.configure(image=img)
                self.lblVideo.image = img

                # Llamar a este método nuevamente después de 1ms
                self.lblVideo.after(1, self.visualizar)
            else:
                # Si no se pudo leer el frame, cerrar la cámara
                self.vid.release()
                self.vid = None
                print("Error al leer el frame, cámara liberada")

    def actualizar_video(self, img):
        self.lblVideo.configure(image=img)
        self.lblVideo.image = img

    def iniciar(self):
        self.pantalla.mainloop()


#EJECUCIÓN DEL PROGRAMA
if __name__ == "__main__":
    app = InterfazGUI()
    app.iniciar()
