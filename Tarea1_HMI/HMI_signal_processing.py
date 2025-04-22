"""
Se implementa el procesamiento de señales de audio mediante filtros digitales.

Incluye la aplicación de filtros pasa bajos, pasa altos y pasa banda, así como
la visualización de la señal en el dominio del tiempo y la frecuencia usando
la Transformada de Fourier. También permite guardar el audio filtrado.
"""

import tempfile

import gradio as gr
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import soundfile as sf


# Cargar y procesar archivos de audio
def load_audio(file):
    """
    Carga un archivo de audio y lo convierte a un array numpy mono.

    Args:
        file (str): Ruta del archivo de audio.

    Returns:
        tuple: Señal de audio (numpy array) y frecuencia de muestreo (int).
    """
    y, sr = librosa.load(file, sr=None, mono=True)
    return y, sr


def apply_filter(y, sr, filter_type="Lowpass", cutoff=[1000], order=4):
    """
    Aplica un filtro digital a la señal de audio.

    Args:
        y (numpy array): Señal de audio.
        sr (int): Frecuencia de muestreo.
        filter_type (str): Tipo de filtro ("Lowpass", "Highpass", "Bandpass").
        cutoff (list): Frecuencia(s) de corte en Hz.
        order (int): Orden del filtro.

    Returns:
        numpy array: Señal filtrada.
    """
    nyquist = 0.5 * sr
    if isinstance(cutoff, int):
        cutoff = [cutoff]
    normalized_cutoff = [f / nyquist for f in cutoff]

    if filter_type == "Lowpass":
        b, a = signal.butter(order, normalized_cutoff, btype="low")
    elif filter_type == "Highpass":
        b, a = signal.butter(order, normalized_cutoff, btype="high")
    elif filter_type == "Bandpass":
        if len(cutoff) != 2:
            raise ValueError(
                "Para un filtro bandpass, 'cutoff' debe tener dos valores."
            )
        b, a = signal.butter(order, normalized_cutoff, btype="band")

    y_filtered = signal.filtfilt(b, a, y)
    return y_filtered


def compute_fourier(y, sr):
    """
    Calcula la Transformada de Fourier de la señal.

    Args:
        y (numpy array): Señal de audio.
        sr (int): Frecuencia de muestreo.

    Returns:
        tuple: Frecuencias y magnitud de la FFT.
    """
    N = len(y)
    T = 1.0 / sr
    xf = np.fft.fftfreq(N, T)[: N // 2]
    yf = np.fft.fft(y)
    return xf, 2.0 / N * np.abs(yf[: N // 2])


def plot_signal(y, sr, title):
    """
    Genera y guarda una gráfica de la señal en el dominio del tiempo.

    Args:
        y (numpy array): Señal de audio.
        sr (int): Frecuencia de muestreo.
        title (str): Título de la gráfica.

    Returns:
        str: Ruta de la imagen guardada.
    """
    plt.figure(figsize=(9, 4))
    plt.plot(np.linspace(0, len(y) / sr, len(y)), y)
    plt.title(title)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid()

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp_file.name, format="png")
    plt.close()
    return tmp_file.name


def plot_fourier(y, sr, title):
    """
    Genera y guarda la Transformada de Fourier de la señal.

    Args:
        y (numpy array): Señal de audio.
        sr (int): Frecuencia de muestreo.
        title (str): Título de la gráfica.

    Returns:
        str: Ruta de la imagen guardada.
    """
    xf, yf = compute_fourier(y, sr)
    mask = xf <= 5000
    xf, yf = xf[mask], yf[mask]

    plt.figure(figsize=(9, 4))
    plt.plot(xf, yf)
    plt.title(title)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud")
    plt.xlim(0, 5000)
    plt.grid()

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp_file.name, format="png")
    plt.close()
    return tmp_file.name


def process_audio(file, filter_type, cutoff, order, apply_fourier):
    """
    Procesa el audio aplicando el filtro y generando las gráficas.

    Args:
        file (str): Ruta del archivo de audio.
        filter_type (str): Tipo de filtro.
        cutoff (list): Frecuencia(s) de corte.
        order (int): Orden del filtro.
        apply_fourier (bool): Si se debe calcular la FFT.

    Returns:
        tuple: Ruta del audio filtrado y rutas de las imágenes generadas.
    """
    y, sr = load_audio(file)
    if y is None:
        return None, None, None, None

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    y_filtered = apply_filter(y, sr, filter_type, cutoff, order)

    original_plot = plot_signal(y, sr, "Señal Original")
    filtered_plot = plot_signal(y_filtered, sr, "Señal Filtrada")

    output_file = "filtered_audio.wav"
    sf.write(output_file, y_filtered, sr)

    fourier_plot_original, fourier_plot_filtered = None, None
    if apply_fourier:
        fourier_plot_original = plot_fourier(
            y, sr, "Transformada de Fourier - Original"
        )
        fourier_plot_filtered = plot_fourier(
            y_filtered, sr, "Transformada de Fourier - Filtrada"
        )

    return (
        output_file,
        original_plot,
        filtered_plot,
        fourier_plot_original,
        fourier_plot_filtered,
    )


def create_interface():
    """Crea la interfaz gráfica con Gradio."""
    theme = gr.themes.Default(primary_hue="red", neutral_hue="gray")
    with gr.Blocks(theme=theme) as demo:
        audio_input = gr.Audio(
            type="filepath",
            label="Cargar/Grabar un archivo de audio",
        )
        filter_type = gr.Dropdown(
            choices=["Lowpass", "Highpass", "Bandpass"],
            label="Tipo de filtro",
            value="Lowpass",
        )
        cutoff_input_1 = gr.Slider(
            100, 5000, value=1000, label="Frecuencia de corte baja (Hz)"
        )
        cutoff_input_2 = gr.Slider(
            100,
            5000,
            value=2000,
            label="Frecuencia de corte alta (Hz)",
            visible=False,
        )
        order_slider = gr.Slider(
            1,
            8,
            value=4,
            label="Orden del filtro",
            step=1,
        )
        fourier_checkbox = gr.Checkbox(
            label="Mostrar transformada de Fourier", value=False
        )

        def update_cutoff_choices(filter_type):
            return (
                (gr.update(visible=True), gr.update(visible=True))
                if filter_type == "Bandpass"
                else (gr.update(visible=True), gr.update(visible=False))
            )

        filter_type.change(
            update_cutoff_choices,
            inputs=filter_type,
            outputs=[cutoff_input_1, cutoff_input_2],
        )
        process_button = gr.Button("Procesar")

        audio_output = gr.Audio(label="Audio filtrado")

        with gr.Row():
            with gr.Column():
                original_plot = gr.Image(label="Señal Original")
                fourier_output_original = gr.Image(
                    label="Transformada de Fourier - Original", visible=False
                )
            with gr.Column():
                filtered_plot = gr.Image(label="Señal Filtrada")
                fourier_output_filtered = gr.Image(
                    label="Transformada de Fourier - Filtrada", visible=False
                )

        # Mostrar/Ocultar gráficas ya generadas
        def toggle_fourier(apply_fourier):
            return (
                gr.update(visible=apply_fourier),  # Fourier Original
                gr.update(visible=apply_fourier),  # Fourier Filtrada
            )

        fourier_checkbox.change(
            toggle_fourier,
            inputs=fourier_checkbox,
            outputs=[fourier_output_original, fourier_output_filtered],
        )

        def process_inputs(
            file,
            filter_type,
            cutoff_1,
            cutoff_2,
            order,
            apply_fourier,
        ):
            cutoff = (
                [round(cutoff_1), round(cutoff_2)]
                if filter_type == "Bandpass"
                else [round(cutoff_1)]
            )
            (
                output_file,
                orig_plot,
                filt_plot,
                fourier_orig,
                fourier_filt,
            ) = process_audio(
                file, filter_type, cutoff, order, True
            )  # <-- Siempre calcula la FFT
            return (
                output_file,
                orig_plot,
                filt_plot,
                gr.update(value=fourier_orig, visible=apply_fourier),
                gr.update(value=fourier_filt, visible=apply_fourier),
            )

        # Resetear todos los outputs al cargar un nuevo archivo
        def reset_outputs(file):
            return (
                None,  # audio_output
                None,  # original_plot
                None,  # filtered_plot
                gr.update(value=None, visible=False),  # fourier_out_original
                gr.update(value=None, visible=False),  # fourier_out_filtered
            )

        audio_input.change(
            reset_outputs,
            inputs=[audio_input],
            outputs=[
                audio_output,
                original_plot,
                filtered_plot,
                fourier_output_original,
                fourier_output_filtered,
            ],
        )

        process_button.click(
            process_inputs,
            inputs=[
                audio_input,
                filter_type,
                cutoff_input_1,
                cutoff_input_2,
                order_slider,
                fourier_checkbox,
            ],
            outputs=[
                audio_output,
                original_plot,
                filtered_plot,
                fourier_output_original,
                fourier_output_filtered,
            ],
        )

    demo.launch(share=True)


create_interface()
