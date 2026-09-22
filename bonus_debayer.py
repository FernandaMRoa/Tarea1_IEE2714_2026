# =============================================================================
# BONUS: Debayerizado e interpolacion de color (patron RGGB)
#
# Genera todas las figuras del informe en la carpeta figuras/bonus/.
#
# Uso:
#
# Este archivo IMPORTA la funcion reescalar de la Pregunta 3 (p3_reescalado.py).
# Para llevar el Super-Pixel al tamano original se
# reutiliza el reescalado de la Pregunta 3 con factor 2.
#
# Imagenes:
#  - Para la referencia conocida se simula una matriz Bayer a partir de una imagen RGB.
#  - Se usan las dos matrices CFA reales entregadas por el profesor
#    (P4_CRW_4866_CFA y P4_IMG_2267_CFA) como demostracion sobre datos reales.
#
# Las imagenes (P4_CRW_4866_CFA y P4_IMG_2267_CFA) deben estar en la carpeta
# "imagenes", ubicada en la misma carpeta que este script (imagenes/P4_..._CFA.tif).
#
# No se usa ninguna funcion externa de debayerizado o demosaicing.
# =============================================================================


import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, data

# se reutiliza el reescalado de la Pregunta 3
from p3_reescalado import reescalar, psnr

# -----------------------------------------------------------------------------
# Rutas
# -----------------------------------------------------------------------------
RUTA_CFA_1 = os.path.join("imagenes", "P4_CRW_4866_CFA.tif")
RUTA_CFA_2 = os.path.join("imagenes", "P4_IMG_2267_CFA.tif")
DIR_FIG = os.path.join("figuras", "bonus")
os.makedirs(DIR_FIG, exist_ok=True)


def guardar(nombre):
    ruta = os.path.join(DIR_FIG, nombre + ".png")
    plt.savefig(ruta, dpi=130, bbox_inches="tight")
    print("  figura ->", ruta)
    plt.show()
    plt.close()


def mostrar(ax, img, titulo):
    if img.ndim == 2:
        ax.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    else:
        ax.imshow(img, interpolation='nearest')
    ax.set_title(titulo, fontsize=9)
    ax.axis('off')


# =============================================================================
# SIMULACION DE LA MATRIZ BAYER RGGB
# =============================================================================
# Patron RGGB (con (i,j) empezando en 0):
#      columna par   columna impar
#  fila par:   R           G1
#  fila impar: G2          B
#
# La posicion se decide solo por la paridad de fila y columna.

def simular_bayer(rgb):
    "Genera una matriz Bayer RGGB conservando en cada posicion su color."
    N, M = rgb.shape[:2]
    bayer = np.zeros((N, M), dtype=np.float64)
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    bayer[0::2, 0::2] = R[0::2, 0::2]      # R en (par, par)
    bayer[0::2, 1::2] = G[0::2, 1::2]      # G1 en (par, impar)
    bayer[1::2, 0::2] = G[1::2, 0::2]      # G2 en (impar, par)
    bayer[1::2, 1::2] = B[1::2, 1::2]      # B en (impar, impar)
    return bayer


def bayer_a_color_falso(bayer):
    "Pinta cada muestra del mosaico en su canal (para visualizar la matriz Bayer)."
    N, M = bayer.shape
    out = np.zeros((N, M, 3), dtype=np.float64)
    out[0::2, 0::2, 0] = bayer[0::2, 0::2]    # R
    out[0::2, 1::2, 1] = bayer[0::2, 1::2]    # G1
    out[1::2, 0::2, 1] = bayer[1::2, 0::2]    # G2
    out[1::2, 1::2, 2] = bayer[1::2, 1::2]    # B
    return np.clip(out, 0, 255).astype(np.uint8)


# =============================================================================
# METODO 1: SUPER-PIXEL
# =============================================================================
def super_pixel(bayer):
    """
    Agrupa cada bloque 2x2 RGGB en un pixel RGB (R, (G1+G2)/2, B).
    La imagen resultante tiene la mitad del tamano.
    """
    N, M = bayer.shape
    N2, M2 = N - N % 2, M - M % 2
    R = bayer[0:N2:2, 0:M2:2]
    G1 = bayer[0:N2:2, 1:M2:2]
    G2 = bayer[1:N2:2, 0:M2:2]
    B = bayer[1:N2:2, 1:M2:2]
    out = np.stack([R, (G1 + G2) / 2.0, B], axis=-1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# =============================================================================
# METODO 2: DEBAYERIZADO BILINEAL A RESOLUCION ORIGINAL
# =============================================================================
def vecino(b, dy, dx):
    "Desplaza la matriz repitiendo el borde (para promediar vecinos)."
    N, M = b.shape
    iy = np.clip(np.arange(N) + dy, 0, N - 1)
    ix = np.clip(np.arange(M) + dx, 0, M - 1)
    return b[iy[:, None], ix[None, :]]


def debayer_bilineal(bayer):
    """
    Debayerizado bilineal a resolucion original. Para cada posicion se
    interpolan las dos componentes faltantes promediando los vecinos que
    tienen ese color:
      - En R:  G = promedio de los 4 vecinos en cruz; B = promedio de los 4 en diagonal.
      - En B:  igual pero cambiando R y B.
      - En G1 (fila de rojos): R = promedio horizontal; B = promedio vertical.
      - En G2 (fila de azules): R = promedio vertical;  B = promedio horizontal.
    Los bordes se tratan repitiendo el pixel del borde.
    """
    b = bayer.astype(np.float64)
    N, M = b.shape

    # promedios de vecinos (se calculan para toda la imagen de una vez)
    arriba, abajo = vecino(b, -1, 0), vecino(b, 1, 0)
    izq, der = vecino(b, 0, -1), vecino(b, 0, 1)
    d1, d2 = vecino(b, -1, -1), vecino(b, -1, 1)
    d3, d4 = vecino(b, 1, -1), vecino(b, 1, 1)
    cruz = (arriba + abajo + izq + der) / 4.0
    diag = (d1 + d2 + d3 + d4) / 4.0
    horiz = (izq + der) / 2.0
    vert = (arriba + abajo) / 2.0

    # mascaras de cada tipo de posicion (por paridad)
    i = np.arange(N)[:, None]
    j = np.arange(M)[None, :]
    mR = (i % 2 == 0) & (j % 2 == 0)
    mG1 = (i % 2 == 0) & (j % 2 == 1)
    mG2 = (i % 2 == 1) & (j % 2 == 0)
    mB = (i % 2 == 1) & (j % 2 == 1)

    R = np.zeros_like(b)
    G = np.zeros_like(b)
    B = np.zeros_like(b)

    # canal R
    R[mR] = b[mR]              # medido
    R[mG1] = horiz[mG1]        # G en fila de rojos -> R esta a los lados
    R[mG2] = vert[mG2]         # G en fila de azules -> R esta arriba/abajo
    R[mB] = diag[mB]           # en B, los R estan en diagonal
    # canal G
    G[mG1] = b[mG1]            # medido
    G[mG2] = b[mG2]            # medido
    G[mR] = cruz[mR]           # en R, los G estan en cruz
    G[mB] = cruz[mB]           # en B, los G estan en cruz
    # canal B
    B[mB] = b[mB]              # medido
    B[mG1] = vert[mG1]         # G en fila de rojos -> B esta arriba/abajo
    B[mG2] = horiz[mG2]        # G en fila de azules -> B esta a los lados
    B[mR] = diag[mR]           # en R, los B estan en diagonal

    out = np.stack([R, G, B], axis=-1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# =============================================================================
# CARGA DE IMAGENES CFA REALES (16 bits)
# =============================================================================
def cargar_cfa_real(ruta, balance=True):
    """
    Carga una matriz CFA real de 16 bits, la lleva a 8 bits con correccion
    gamma. Si balance=True aplica un balance de blancos simple (iguala el
    promedio de R, G y B).
    """
    raw = io.imread(ruta).astype(np.float64)
    # normaliza a [0,1] con gamma y lleva a 0-255
    x = np.clip((raw / raw.max()) ** (1 / 2.2), 0, 1) * 255.0
    bayer = x
    if balance:
        # promedios por canal (patron RGGB)
        mR = bayer[0::2, 0::2].mean()
        mG = (bayer[0::2, 1::2].mean() + bayer[1::2, 0::2].mean()) / 2
        mB = bayer[1::2, 1::2].mean()
        g = mG
        bayer = bayer.copy()
        bayer[0::2, 0::2] *= g / (mR + 1e-6)
        bayer[1::2, 1::2] *= g / (mB + 1e-6)
    return np.clip(bayer, 0, 255)


# =============================================================================
# EXPERIMENTACION Y ANALISIS
# =============================================================================
def reconstrucciones(rgb):
    "Devuelve las 3 reconstrucciones pedidas a partir de una imagen RGB."
    bayer = simular_bayer(rgb)
    sp = super_pixel(bayer)
    sp_vec = reescalar(sp, 2.0, 'vecino')      # Super-Pixel + vecino (Pregunta 3)
    sp_bil = reescalar(sp, 2.0, 'bilineal')    # Super-Pixel + bilineal (Pregunta 3)
    deb = debayer_bilineal(bayer)              # debayer bilineal a resolucion original
    return bayer, sp, sp_vec, sp_bil, deb


def recortar_par(rgb):
    "Recorta la imagen a dimensiones pares para que el bloque 2x2 calce."
    N, M = rgb.shape[:2]
    return rgb[:N - N % 2, :M - M % 2]


def main():
    # ------- imagen RGB de referencia (para medir PSNR) ----------------------
    # se usa una imagen con detalle de color fino; astronaut 
    ref = recortar_par(data.astronaut())
    print(f"Imagen de referencia RGB: {ref.shape}")

    # ------- [1] Comparacion de los 3 metodos (referencia conocida) ----------
    print("\n[1] Comparacion de metodos con referencia conocida (puntos 1-2)")
    bayer, sp, sp_vec, sp_bil, deb = reconstrucciones(ref)

    def recorta(a, esc=1.0):
        # recorte de la cara para ver detalle
        y0, x0, s = int(20 * esc), int(150 * esc), int(160 * esc)
        return a[y0:y0 + s, x0:x0 + s]

    metodos = [("Original", ref), ("Mosaico Bayer", bayer_a_color_falso(bayer)),
               ("SuperPixel+vecino", sp_vec), ("SuperPixel+bilineal", sp_bil),
               ("Debayer bilineal", deb)]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for j, (t, im_) in enumerate(metodos):
        extra = ""
        if t.startswith(("SuperPixel", "Debayer")):
            p = psnr(ref[:im_.shape[0], :im_.shape[1]], im_[:ref.shape[0], :ref.shape[1]])
            extra = f"\nPSNR={p:.2f} dB"
            print(f"  {t}: PSNR={p:.2f} dB")
        mostrar(axes[0, j], im_, f"{t}{extra}")
        mostrar(axes[1, j], recorta(im_), "recorte (cara)")
    fig.suptitle("BONUS - Reconstrucciones a partir de la matriz Bayer (referencia: astronaut)")
    plt.tight_layout()
    guardar("01_comparacion_metodos")

    # ------- [2] Que se pierde al formar el Super-Pixel (punto 3) -------------
    print("\n[2] Las dos formas de reescalar el Super-Pixel (punto 3)")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    mostrar(axes[0], recorta(ref), "original")
    mostrar(axes[1], recorta(sp, 0.5), "Super-Pixel (mitad de tamano)")
    mostrar(axes[2], recorta(sp_vec), "SP + vecino (x2)")
    mostrar(axes[3], recorta(sp_bil), "SP + bilineal (x2)")
    p_vec = psnr(ref[:sp_vec.shape[0], :sp_vec.shape[1]], sp_vec[:ref.shape[0], :ref.shape[1]])
    p_bil = psnr(ref[:sp_bil.shape[0], :sp_bil.shape[1]], sp_bil[:ref.shape[0], :ref.shape[1]])
    fig.suptitle(f"BONUS.3 - El Super-Pixel descarta la posicion dentro del bloque 2x2 "
                 f"(SP+vecino {p_vec:.1f} dB, SP+bilineal {p_bil:.1f} dB)")
    plt.tight_layout()
    guardar("02_superpixel")

    # ------- [3] Artefacto de color en detalle fino (punto 4) ----------------
    print("\n[3] Artefacto de color en detalle fino (punto 4)")
    # imagen sintetica con lineas finas de color (caso extremo)
    Nn = 128
    synt = np.zeros((Nn, Nn, 3), dtype=np.uint8)
    synt[:, 0::2, 0] = 255       # columnas pares rojas
    synt[:, 1::2, 2] = 255       # columnas impares azules
    bayer_s = simular_bayer(synt)
    deb_s = debayer_bilineal(bayer_s)
    sp_s = reescalar(super_pixel(bayer_s), 2.0, 'bilineal')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    mostrar(axes[0], synt[:40, :40], "original (lineas 1 px rojo/azul)")
    mostrar(axes[1], deb_s[:40, :40], "debayer bilineal")
    mostrar(axes[2], sp_s[:40, :40], "SuperPixel + bilineal")
    fig.suptitle("BONUS.4 - Colores falsos: cuando el detalle es mas fino que el patron de muestreo")
    plt.tight_layout()
    guardar("03_colores_falsos")

    # ------- [4] Segunda imagen RGB con detalle de color (punto 1) -----------
    print("\n[4] Segunda imagen de referencia (coffee)")
    ref2 = recortar_par(data.coffee())
    _, _, sp_vec2, sp_bil2, deb2 = reconstrucciones(ref2)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    mostrar(axes[0], ref2, "original")
    for ax, (t, im_) in zip(axes[1:], [("SP+vecino", sp_vec2), ("SP+bilineal", sp_bil2),
                                       ("Debayer bilineal", deb2)]):
        p = psnr(ref2[:im_.shape[0], :im_.shape[1]], im_[:ref2.shape[0], :ref2.shape[1]])
        mostrar(ax, im_, f"{t}\nPSNR={p:.2f} dB")
        print(f"  {t}: PSNR={p:.2f} dB")
    fig.suptitle("BONUS - Segunda imagen de referencia (coffee)")
    plt.tight_layout()
    guardar("04_segunda_imagen")

    # ------- [5] Matrices CFA ----------------------------------------------
    print("\n[5] Demostracion sobre las matrices CFA reales")
    for ruta, nom in [(RUTA_CFA_1, "P4_CRW_4866"), (RUTA_CFA_2, "P4_IMG_2267")]:
        if not os.path.exists(ruta):
            print(f"  (aviso) no se encontro {ruta}")
            continue
        bayer_r = cargar_cfa_real(ruta, balance=True)
        deb_r = debayer_bilineal(bayer_r)
        sp_r = super_pixel(bayer_r)
        # recorte central para ver detalle
        N, M = deb_r.shape[:2]
        y0, x0, s = N // 2, M // 2, min(300, N // 3, M // 3)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        mostrar(axes[0], bayer_a_color_falso(bayer_r)[y0:y0 + s, x0:x0 + s], "mosaico Bayer (recorte)")
        mostrar(axes[1], sp_r[y0 // 2:y0 // 2 + s // 2, x0 // 2:x0 // 2 + s // 2], "Super-Pixel")
        mostrar(axes[2], deb_r[y0:y0 + s, x0:x0 + s], "Debayer bilineal")
        fig.suptitle(f"BONUS - Matriz CFA real: {nom} (con balance de blancos simple)")
        plt.tight_layout()
        guardar(f"05_cfa_real_{nom}")
        # imagen completa reducida
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        mostrar(ax, reescalar(deb_r.astype(np.uint8), 0.5, 'bilineal'),
                f"{nom} - debayer bilineal (imagen completa)")
        plt.tight_layout()
        guardar(f"05b_cfa_real_completa_{nom}")

    # ------- Preguntas guiadas: traza de un pixel ----------------------------
    print("\n[6] Traza de un pixel (preguntas guiadas)")
    bayer = simular_bayer(ref)
    for (i, j), tipo in [((100, 100), "R (par,par)"), ((100, 101), "G1 (par,impar)"),
                         ((101, 100), "G2 (impar,par)"), ((101, 101), "B (impar,impar)")]:
        deb = debayer_bilineal(bayer)
        print(f"  Pixel ({i},{j}) tipo {tipo}:")
        print(f"    medido en el mosaico = {bayer[i, j]:.0f}")
        print(f"    RGB reconstruido     = {tuple(deb[i, j])}")
        print(f"    RGB original         = {tuple(ref[i, j])}")

    print("\nFin. Todas las figuras estan en", DIR_FIG)


if __name__ == "__main__":
    main()
