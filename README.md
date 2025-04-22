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

## 👨‍💻 Autor

Juan Francisco García Rodríguez.

Integración de robótica y sistemas inteligentes (Gpo 581).

</div>
