# Tarea 1 — Fundamentos de Procesamiento de Imágenes (IEE2714)

**Pontificia Universidad Católica de Chile**
**Curso:** IEE2714 — Fundamentos de Procesamiento de Imágenes
**Semestre:** Segundo Semestre 2026

---

## Descripción del Proyecto

Este repositorio contiene las soluciones e implementaciones para la Tarea 1 del curso. Incluye el
procesamiento de imágenes para saturación selectiva de color, ecualización local de histograma,
reescalado (interpolación por vecino más cercano y bilineal) y el bonus de debayerizado (CFA).

Cada pregunta está resuelta en un archivo de Python independiente. Al ejecutar cada archivo se
generan automáticamente todas las figuras de esa pregunta en la subcarpeta correspondiente de
`figuras/`.

---

## Estructura del Repositorio

```text
Tarea1_IEE2714_2026/
│
├── README.md                   # Este archivo
├── requirements.txt            # Dependencias de Python necesarias
├── .gitignore                  # Archivos ignorados por Git
│
├── p1_saturacion_color.py      # Pregunta 1: saturación selectiva de color
├── p2_ecualizacion_local.py    # Pregunta 2: ecualización local de histograma
├── p3_reescalado.py            # Pregunta 3: reescalado (vecino más cercano y bilineal)
├── bonus_debayer.py            # Bonus: debayerizado (CFA)
│
├── imagenes/                   # Imágenes de entrada (deben estar presentes para ejecutar)
│   ├── P1_IMG_2402.tif
│   ├── P2_IMG_2423.tif
│   ├── P3_IMG_2387_crop.tif
│   ├── P4_CRW_4866_CFA.tif
│   └── P4_IMG_2267_CFA.tif
│
└── figuras/                    # Figuras generadas automáticamente al ejecutar los scripts
    ├── p1/
    ├── p2/
    ├── p3/
    └── bonus/
```

---

## Requisitos previos

- **Python 3.10 o superior.**
- Las librerías listadas en `requirements.txt`: `numpy`, `matplotlib`, `scikit-image` y
  `opencv-python`.

Instalación de las dependencias (desde la carpeta raíz del proyecto):

```bash
pip install -r requirements.txt
```

---

## Cómo ejecutar los códigos

> **Importante:** todos los scripts deben ejecutarse **desde la carpeta raíz del proyecto**
> (la carpeta `Tarea1_IEE2714_2026/`), no desde dentro de `imagenes/` ni de `figuras/`. Los scripts
> buscan las imágenes en la ruta relativa `imagenes/<nombre>.tif` y guardan las figuras en
> `figuras/<pregunta>/`, por lo que la ubicación desde la que se ejecuta importa.

Ubicándose primero en la carpeta raíz del proyecto:

```bash
cd Tarea1_IEE2714_2026
```

Luego, cada pregunta se ejecuta con su propio comando:

```bash
python p1_saturacion_color.py     # Pregunta 1  -> genera las figuras en figuras/p1/
python p2_ecualizacion_local.py   # Pregunta 2  -> genera las figuras en figuras/p2/
python p3_reescalado.py           # Pregunta 3  -> genera las figuras en figuras/p3/
python bonus_debayer.py           # Bonus       -> genera las figuras en figuras/bonus/
```

Cada script imprime en la consola la ruta de cada figura que va guardando.

> **Nota sobre el bonus:** `bonus_debayer.py` reutiliza la función `reescalar` de la Pregunta 3, por
> lo que importa `p3_reescalado.py`. Ambos archivos deben estar en la misma carpeta (como ya lo
> están en este repositorio) para que el bonus se ejecute correctamente.

---

## Imágenes de entrada

Las cinco imágenes entregadas por el profesor deben estar en la carpeta `imagenes/` con exactamente
estos nombres:

| Archivo | Se usa en |
|---|---|
| `P1_IMG_2402.tif` | Pregunta 1 |
| `P2_IMG_2423.tif` | Pregunta 2 |
| `P3_IMG_2387_crop.tif` | Pregunta 3 |
| `P4_CRW_4866_CFA.tif` | Bonus |
| `P4_IMG_2267_CFA.tif` | Bonus |

Las imágenes `astronaut`, `coffee`, `text` y `moon` que también se usan como imágenes de prueba
**no** necesitan descargarse: se obtienen automáticamente del módulo `skimage.data`.

---

## Reproducción de las figuras del informe

Para regenerar todas las figuras del informe basta con instalar las dependencias y ejecutar los
cuatro scripts en orden desde la carpeta raíz. Las figuras se sobrescriben en las subcarpetas de
`figuras/`, que es de donde el informe las toma.

---

## Notas de implementación

- Salvo las excepciones permitidas por el enunciado (lectura/escritura de archivos, manejo de
  arreglos y visualización), no se utilizan funciones externas que implementen los algoritmos
  solicitados.
- En la Pregunta 2, `cv2.createCLAHE` se usa **únicamente** como referencia de comparación, tal como
  lo permite el enunciado; el algoritmo propio de control de contraste está implementado desde cero.
- En la Pregunta 3 y en el bonus no se usa ninguna función externa de reescalado, interpolación ni
  demosaicing.
