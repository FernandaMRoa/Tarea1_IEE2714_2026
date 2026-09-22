# Tarea 1 — Fundamentos de Procesamiento de Imágenes (IEE2714)

**Pontificia Universidad Católica de Chile**  
**Curso:** IEE2714 — Fundamentos de Procesamiento de Imágenes  
**Semestre:** Segundo Semestre 2026  

---

## Descripción del Proyecto

Este repositorio contiene las soluciones e implementaciones para la Tarea 1 del curso. Incluye el procesamiento de imágenes para saturación de color, ecualización local, reescalado (interpolación por vecino más cercano y bilineal) y el bonus de debayerizado.

---

## Estructura del Repositorio

```text
Tarea1_IEE2714_2026/
│
├── README.md                  # Instrucciones y documentación del proyecto
├── requirements.txt            # Dependencias de Python necesarias
├── .gitignore                 # Archivos e historiales ignorados por Git
│
├── p1_saturacion_color.py     # Pregunta 1: Ajuste de saturación de color
├── p2_ecualizacion_local.py    # Pregunta 2: Ecualización local de histograma
├── p3_reescalado.py           # Pregunta 3: Reescalado (Vecino más cercano vs. Bilineal)
├── bonus_debayer.py           # Bonus: Algoritmo de debayerizado (CFA)
│
├── imagenes/                  # Imágenes de entrada necesarias para la ejecución
│   ├── P1_IMG_2402.tif
│   ├── P2_IMG_2423.tif
│   ├── P3_IMG_2387_crop.tif
│   ├── P4_CRW_4866_CFA.tif
│   └── P4_IMG_2267_CFA.tif
│
└── figuras/                   # Resultados e imágenes generadas automáticamente
    ├── p1/
    ├── p2/
    ├── p3/
    └── bonus/