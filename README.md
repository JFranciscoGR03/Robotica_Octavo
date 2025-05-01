<div align="justify">

# Interfaces de Hardware y Software

## 📌 Trabajo 1: HMI for Signal Processing

## 📄 Descripción

El proyecto implementa el procesamiento de señales de audio mediante filtros digitales. En esta tarea, se desarrolló una interfaz para la manipulación de señales, permitiendo la aplicación de filtros pasa bajos, pasa altos y pasa banda. Además, se visualiza la señal tanto en el dominio del tiempo como en de la frecuencia utilizando la Transformada de Fourier. También se permite guardar el audio filtrado y cada una de las gráficas.

## ⚙️ Requisitos

Para ejecutar el proyecto, necesitas tener Python 3.10 instalado. Luego, sigue estos pasos para crear y activar un entorno virtual:
```bash
# 1. Crear el entorno virtual
python -m venv venv

# 2. Activar el entorno virtual (en Windows)
venv\Scripts\activate

# 2. Activar el entorno virtual (en macOS/Linux)
source venv/bin/activate
```

Una vez activado el entorno virtual, instala las dependencias necesarias ejecutando:
```bash
pip install gradio librosa matplotlib numpy scipy soundfile
```

Estas son las bibliotecas necesarias para ejecutar el proyecto:

- **gradio**: Para la interfaz gráfica de usuario.
- **librosa**: Para el procesamiento y análisis de archivos de audio.
- **matplotlib**: Para la visualización de las señales.
- **numpy**: Para el manejo de arrays y cálculos matemáticos.
- **scipy**: Para la aplicación de filtros digitales.
- **soundfile**: Para guardar los archivos de audio filtrados.

## 🧠 Estructura del código

El código principal está contenido en el archivo `HMI_signal_processing.py` y se organiza en las siguientes funciones:

- **`load_audio(file)`**: Carga un archivo de audio y lo convierte a un array numpy mono.

- **`apply_filter(y, sr, filter_type, cutoff, order)`**: Aplica un filtro digital a la señal de audio.

- **`compute_fourier(y, sr)`**: Calcula la Transformada de Fourier de la señal.

- **`plot_signal(y, sr, title)`**: Genera y guarda una gráfica de la señal en el dominio del tiempo.

- **`plot_fourier(y, sr, title)`**: Genera y guarda la Transformada de Fourier de la señal.

- **`process_audio(file, filter_type, cutoff, order, apply_fourier)`**: Procesa el audio aplicando el filtro y generando las gráficas.

- **`create_interface()`**: Crea la interfaz gráfica con Gradio.

## 🧪 Uso

1. **Cargar o grabar un archivo de audio**: Selecciona un archivo de audio para cargar desde tu dispositivo o grabarlo en tiempo real dentro de la interfaz.
2. **Seleccionar el filtro**: Elige entre los filtros `Lowpass`, `Highpass` o `Bandpass` utilizando el dropdown.
3. **Ajustar frecuencias de corte**: Configura las frecuencias de corte según el tipo de filtro seleccionado.
4. **Ajustar el orden del filtro**: Describe el grado de aceptación o rechazo de frecuencias, por arriba o por debajo, de la respectiva frecuencia de corte.
5. **Procesar**: Haz clic en "Procesar" para aplicar el filtro y generar las gráficas de la señal.
6. **Visualización**:
   - Se visualizarán las gráficas en el dominio del tiempo de la señal original y filtrada.
   - Tienes la opción de visualizar la Transformada de Fourier de la señal original y filtrada mediante un checkbox.
7. **Descargar**: El audio filtrado y las gráficas generadas se pueden descargar directamente desde la interfaz.

## 🚀 Ejecución

Para ejecutar el proyecto:

1. Descarga el código del repositorio.
2. Abre el archivo `HMI_signal_processing.py` en un IDE como **Visual Studio Code**.
3. Ejecuta el archivo. Al correrlo, **Gradio generará una URL en la terminal**.
4. Abre esa URL en tu navegador para acceder a la interfaz de usuario.
5. Desde allí podrás:
   - Cargar o grabar un archivo de audio.
   - Aplicar el filtro deseado seleccionando la frecuencia de corte y el orden del filtro.
   - Visualizar las gráficas en el dominio del tiempo y la frecuencia.
   - Descargar el audio filtrado y las gráficas.

## 💻 Interfaz gráfica

![Screenshot interfaz_grafica](https://github.com/user-attachments/assets/7e7c8f09-9140-4ff6-95a9-192009ce5cff)
![Screenshot graficas](https://github.com/user-attachments/assets/f2474795-072d-4436-bb68-d030bbf5ddbc)

## 📌 Trabajo 2: Kalman Filter Application

## 📄 Descripción

Este proyecto realiza **detección y seguimiento de aviones en video** utilizando el modelo **YOLOv8** para detectar objetos y un **Filtro de Kalman** para seguir a múltiples aviones a través de los fotogramas. Se procesan videos cuadro por cuadro para detectar objetos tipo "avión". Cada detección se asocia a un tracker basado en Filtro de Kalman que predice su posición en el siguiente frame, incluso si el objeto se oculta brevemente. Cada avión es identificado con un ID único que trata de permanecer consistente a lo largo del video.

## ⚙️ Requisitos

Este proyecto requiere **Python 3.8 o superior** y las siguientes bibliotecas:

| Paquete        | Versión mínima recomendada | Descripción                                           |
|----------------|----------------------------|-------------------------------------------------------|
| `opencv-python`| `>=4.5.0`                  | Para lectura de video, dibujo de cajas y visualización|
| `numpy`        | `>=1.21.0`                 | Operaciones numéricas y manejo de matrices            |
| `ultralytics`  | `>=8.0.0`                  | Para cargar y ejecutar modelos YOLOv8                 |
| `filterpy`     | `>=1.4.5`                  | Implementación del filtro de Kalman                   |

Primero debes activar el entorno virtual:
```bash
# En Windows
venv\Scripts\activate

# En macOS/Linux
source venv/bin/activate
```

Después puedes instalar todos los requisitos ejecutando:
```bash
pip install opencv-python numpy ultralytics filterpy
```

## 🧠 Estructura del código

La estructura del proyecto es la siguiente:
```plaintext
Tarea2_Kalman_Filter
   ├── Videos_Funcionamiento
      ├── aviones_1.mp4
      ├── aviones_2.mp4
      ├── aviones_3.mp4
      ├── aviones_4.mp4
   ├── Videos_Prueba
      ├── aviones_1.mp4
      ├── aviones_2.mp4
      ├── aviones_3.mp4
      ├── aviones_4.mp4
   ├── yolo_kalman.py
   ├── Reporte_Kalman_Filter_Application.pdf
   ├── yolov8m.pt
```

1. **Tarea2_Kalman_Filter**:
   Es la carpeta principal que contiene todos los archivos relacionados con el proyecto.

2. **Videos_Funcionamiento**:
   Contiene los videos de ejemplo que se usarán para probar el funcionamiento del modelo y el sistema de seguimiento de aviones. Estos videos tienen nombres de `aviones_1.mp4` a `aviones_4.mp4`.

3. **Videos_Prueba**:
   Contiene los videos de prueba, que también van de `aviones_1.mp4` a `aviones_4.mp4`. Pueden ser utilizados para evaluar y comparar el rendimiento del sistema en diferentes condiciones.

4. **yolov8m.pt**:
   Este es el modelo preentrenado de YOLOv8, que es descargado automáticamente al ejecutar el código de `yolo_kalman.py`. Se utiliza para detectar aviones en los videos.

5. **yolo_kalman.py**:
   Este es el archivo principal que contiene el código para la detección de objetos (aviones) y el seguimiento mediante el filtro de Kalman. El flujo principal del código realiza lo siguiente:

   - **Carga del modelo YOLO**: Utiliza el modelo `yolov8m.pt` para detectar aviones en los frames de los videos.

   - **Inicialización de los trackers**: Para cada avión detectado, se crea un objeto `KalmanTracker`, que es un filtro de Kalman que predice y actualiza las posiciones de los aviones.

   - **Detección de aviones**: El modelo YOLO detecta los aviones en cada frame y las cajas delimitadoras (bboxes) se pasan al filtro de Kalman para su seguimiento.

   - **Cálculo de IoU**: Se calcula el índice de intersección sobre la unión (IoU) para asociar las detecciones con los trackers existentes.

   - **Actualización y predicción**: El filtro de Kalman predice la posición futura de los aviones y actualiza los trackers basándose en las detecciones.

   - **Visualización**: Se dibujan las cajas de seguimiento en los frames y se muestra el video procesado con las predicciones y actualizaciones de los aviones detectados.

Cada video de prueba y funcionamiento en las carpetas correspondientes será procesado por el archivo `yolo_kalman.py`, que utilizará el modelo YOLO para detectar los aviones y el filtro de Kalman para hacer el seguimiento y predecir sus trayectorias.

## 🚀 Ejecución

Este proyecto realiza la detección y el seguimiento de aviones en videos utilizando el modelo YOLOv8 y el filtro de Kalman. Para ejecutar el código, sigue los pasos a continuación.

### 1. Clonar el repositorio.

Si aún no tienes una copia local del repositorio, clónalo desde GitHub (si es necesario):

```bash
git clone <URL del repositorio>
cd Tarea2_Kalman_Filter
```

### 2. Instalación de dependencias.

Mediante el comando que se encuentra en el apartado de requisitos.

### 3. Ejecución de código.

Para ejecutar el código y realizar el seguimiento de los aviones en un video, simplemente corre el siguiente comando:
```bash
python yolo_kalman.py
```

### El código realizará lo siguiente:

- Cargará el modelo `yolov8m.pt` de YOLOv8 (si no está presente, se descargará automáticamente).
- Procesará los videos ubicados en la carpeta `Videos_Prueba`, dependiendo de lo que se haya configurado en el código.
- Detectará los aviones en cada frame del video.
- Aplicará el filtro de Kalman para hacer el seguimiento de los aviones detectados.
- Mostrará el video procesado con las cajas de seguimiento y las etiquetas de ID de los aviones en una ventana emergente.

### 4. Salir del video

Para detener la ejecución y cerrar la ventana del video, espera a que finalice el video o simplemente presiona la tecla `q` mientras el video está en ejecución.

## 🔧 Parámetros configurables

Dentro del código, existen parámetros configurables que puedes ajustar según tus necesidades:

- **`confidence_threshold`**:
  El umbral de confianza para las detecciones de YOLO. Solo las detecciones con una confianza mayor a este valor serán procesadas.
  **Valor actual**: `0.5`
  Puedes modificar este valor en el archivo `yolo_kalman.py` según lo necesites.

- **`video_folder`**:
  La carpeta donde se encuentran los videos que deseas procesar. Puedes elegir cualquiera de los videos dentro de la carpeta `Videos_Prueba`. O modifica la ruta de esta variable en el código si deseas trabajar con otro video u otra carpeta de videos.

## ❓ Solución de problemas

Si te encuentras con alguno de los siguientes problemas, aquí tienes algunas posibles soluciones:

- **El modelo YOLO no se descarga automáticamente**:
  Asegúrate de tener una conexión a Internet activa. Si el modelo no se descarga, también puedes descargarlo manualmente desde el repositorio de YOLOv8 y colocarlo en la misma carpeta que el archivo `yolo_kalman.py`.

- **Problemas al leer los videos**:
  Si el video no se puede cargar, asegúrate de que el archivo `.mp4` esté en la carpeta correcta y no esté dañado. Además, verifica que tu instalación de OpenCV sea compatible con el formato de video que estás utilizando.

- **Errores de dependencias**:
  Si tienes problemas con las bibliotecas, asegúrate de que las versiones de las dependencias sean las correctas. Puedes reinstalar las bibliotecas con el siguiente comando:
  ```bash
  pip install --upgrade opencv-python numpy ultralytics filterpy
  ```

## 👨‍💻 Autor

Juan Francisco García Rodríguez.

Integración de robótica y sistemas inteligentes (Gpo 581).

</div>
