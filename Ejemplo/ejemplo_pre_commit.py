"""Módulo con una calculadora que guarda un historial de operaciones."""

import json


class Calculadora:
    """Realiza operaciones matemáticas y almacena el historial."""

    def __init__(self):
        """Inicializa la calculadora con un historial vacío."""
        self.historial = []

    def sumar(self, a: float, b: float) -> float:
        """Suma dos números y almacena el resultado en el historial."""
        resultado = a + b
        self.historial.append({"operacion": "suma", "resultado": resultado})
        return resultado

    def restar(self, a: float, b: float) -> float:
        """Resta dos números y almacena el resultado en el historial."""
        resultado = a - b
        self.historial.append({"operacion": "resta", "resultado": resultado})
        return resultado

    def imprimir_historial(self):
        """Imprime el historial de operaciones."""
        for entrada in self.historial:
            print(entrada["operacion"], ":", entrada["resultado"])

    def guardar_historial(self, archivo: str):
        """Guarda el historial de operaciones en un archivo JSON."""
        print("Guardando...")
        with open(archivo, "w") as f:
            json.dump(self.historial, f, indent=4)


def ejecutar():
    """Ejecuta la calculadora con operaciones de prueba."""
    print("Iniciando ejecución")
    calc = Calculadora()
    calc.sumar(5, 3)
    calc.restar(10, 2)
    calc.imprimir_historial()
    calc.guardar_historial("historial.json")
