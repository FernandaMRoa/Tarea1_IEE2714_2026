# =============================================================================
# Pregunta 3: Reescalado e interpolacion (vecino mas cercano y bilineal)
#
# Genera todas las figuras del informe en la carpeta figuras/p3/.
#
# Uso:
# La imagen P3_IMG_2387_crop.tif debe estar en la misma carpeta que este script
#
# No se usa ninguna funcion externa de interpolacion ni de reescalado.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, data

# -----------------------------------------------------------------------------
# Rutas de entrada / salida
# -----------------------------------------------------------------------------
RUTA_IMG = os.path.join("imagenes", "P3_IMG_2387_crop.tif")    
DIR_FIG = os.path.join("figuras", "p3")
os.makedirs(DIR_FIG, exist_ok=True)


# =============================================================================
# UTILIDADES 
# =============================================================================
def cargar(ruta, gris=False):
    raw = io.imread(ruta)
    if raw.ndim == 3:
        img = raw[:, :, :3]
        if gris:
            g = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            return np.clip(np.round(g), 0, 255).astype(np.uint8)
    else:
        img = raw
    return np.clip(img, 0, 255).astype(np.uint8)


def guardar(nombre):
    "Guarda la figura actual en figuras/p3/ y la muestra."
    ruta = os.path.join(DIR_FIG, nombre + ".png")
    plt.savefig(ruta, dpi=130, bbox_inches="tight")
    print("  figura ->", ruta)
    plt.show()
    plt.close()


def mostrar(ax, img, titulo):
    "Muestra una imagen (gris o RGB) en un eje."
    if img.ndim == 2:
        ax.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    else:
        ax.imshow(img, interpolation='nearest')
    ax.set_title(titulo, fontsize=9)
    ax.axis('off')


# =============================================================================
# REESCALADO (implementacion propia)
# =============================================================================
# CONVENCION DE COORDENADAS:
# se recorre cada pixel de SALIDA (i, j) y se calcula su posicion en la imagen
# de ENTRADA. Cada pixel de indice i cubre el intervalo [i, i+1), y su centro
# esta en i + 0.5. Con factor de escala s, la relacion entre los centros es:
#
#     (i + 0.5) / s = i0 + 0.5      =>      i0 = (i + 0.5) / s - 0.5
#
# Con esta convencion los centros de las dos imagenes estan alineados.

def tamano_salida(N, s):
    "Tamano de salida a partir del factor s: se redondea N*s al entero mas cercano."
    return max(1, int(round(N * s)))


def reescalar(img, s, modo='bilineal'):
    """
    Reescala una imagen (gris o RGB) por un factor real s (0.5 <= s <= 2).
    modo: 'vecino' (vecino mas cercano) o 'bilineal'.

    Se implementa con mapeo inverso: para cada pixel de salida se calcula su
    posicion en la entrada y se interpola. Los bordes se tratan recortando la
    coordenada al rango valido (se repite el pixel del borde).
    """
    plano = (img.ndim == 2)
    A = img.astype(np.float64)
    if plano:
        A = A[:, :, None]           # se trata gris como una imagen de 1 canal
    N, M, C = A.shape

    # tamano de la imagen de salida y factor efectivo (por si N*s no es entero)
    Ns, Ms = tamano_salida(N, s), tamano_salida(M, s)
    sy, sx = Ns / N, Ms / M

    # coordenadas de entrada para cada fila y columna de salida
    i = np.arange(Ns)
    j = np.arange(Ms)
    y = (i + 0.5) / sy - 0.5        # coordenada de entrada de cada fila
    x = (j + 0.5) / sx - 0.5        # coordenada de entrada de cada columna

    salida = np.zeros((Ns, Ms, C), dtype=np.float64)

    if modo == 'vecino':
        # VECINO MAS CERCANO: se redondea la coordenada al pixel mas cercano
        iy = np.clip(np.round(y).astype(int), 0, N - 1)
        ix = np.clip(np.round(x).astype(int), 0, M - 1)
        salida = A[iy[:, None], ix[None, :], :]

    elif modo == 'bilineal':
        # BILINEAL: se toman los 4 vecinos y se combinan con sus pesos
        y0 = np.floor(y).astype(int)      # fila del vecino de arriba
        x0 = np.floor(x).astype(int)      # columna del vecino de la izquierda
        dy = (y - y0)[:, None, None]      # peso vertical (distancia al de arriba)
        dx = (x - x0)[None, :, None]      # peso horizontal

        # se recortan los indices al borde (se repite el pixel del borde)
        y0c = np.clip(y0, 0, N - 1)
        y1c = np.clip(y0 + 1, 0, N - 1)
        x0c = np.clip(x0, 0, M - 1)
        x1c = np.clip(x0 + 1, 0, M - 1)

        # los 4 vecinos
        Ia = A[y0c[:, None], x0c[None, :], :]   # arriba-izquierda
        Ib = A[y0c[:, None], x1c[None, :], :]   # arriba-derecha
        Ic = A[y1c[:, None], x0c[None, :], :]   # abajo-izquierda
        Id = A[y1c[:, None], x1c[None, :], :]   # abajo-derecha

        # pesos bilineales (suman 1)
        salida = (Ia * (1 - dy) * (1 - dx) + Ib * (1 - dy) * dx
                  + Ic * dy * (1 - dx) + Id * dy * dx)
    else:
        raise ValueError("modo debe ser 'vecino' o 'bilineal'")

    if plano:
        salida = salida[:, :, 0]
    return np.clip(np.round(salida), 0, 255).astype(np.uint8)


# =============================================================================
# METRICAS 
# =============================================================================
def psnr(a, b):
    "PSNR en dB entre dos imagenes del mismo tamano (mientras mas grande = mas parecidas)."
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return float('inf')
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def detalle(img):
    "Cantidad de detalle fino: promedio del valor absoluto del gradiente."
    x = img.astype(np.float64)
    if x.ndim == 3:
        x = x.mean(axis=2)
    gx = np.abs(np.diff(x, axis=1)).mean()
    gy = np.abs(np.diff(x, axis=0)).mean()
    return float((gx + gy) / 2)


# =============================================================================
# EXPERIMENTACION Y ANALISIS
# =============================================================================
FACTORES = [0.6, 0.75, 1.3, 1.7]      # 2 reducciones y 2 ampliaciones, no enteros


def main():
    if not os.path.exists(RUTA_IMG):
        raise FileNotFoundError(
            f"No se encontro '{RUTA_IMG}'.")
    img_color = cargar(RUTA_IMG)                 # imagen obligatoria (tigre)
    img_gris = cargar(RUTA_IMG, gris=True)
    img_text = data.text()                       # 2da imagen: lineas finas de texto
    print(f"Imagen '{RUTA_IMG}' cargada. Dimensiones: {img_color.shape}")

    # ---- [1] Comparacion vecino vs bilineal en varios factores --------------
    print("\n[1] Comparacion vecino vs bilineal (puntos 1-4)")
    # recorte con bordes fuertes y bigotes (lineas finas) del tigre
    N, M = img_color.shape[:2]
    ry, rx, rs = 0.55, 0.05, 0.28     # region relativa (zona de bigotes)
    fig, axes = plt.subplots(3, len(FACTORES), figsize=(3.4 * len(FACTORES), 10))
    for j, s in enumerate(FACTORES):
        vec = reescalar(img_color, s, 'vecino')
        bil = reescalar(img_color, s, 'bilineal')
        print(f"  s={s}: salida {vec.shape[:2]}, detalle vecino={detalle(vec):.2f}, "
              f"bilineal={detalle(bil):.2f}")
        for fila, (im_, t_) in enumerate([(img_color, "original"), (vec, "vecino"), (bil, "bilineal")]):
            h_, w_ = im_.shape[:2]
            y0, x0 = int(ry * h_), int(rx * w_)
            dh, dw = int(rs * h_), int(rs * w_)
            mostrar(axes[fila, j], im_[y0:y0 + dh, x0:x0 + dw],
                    f"{t_} s={s}\n{h_}x{w_}")
    fig.suptitle("P3.1-4 - Recortes ampliados (misma region). Fila 1: original, 2: vecino, 3: bilineal")
    plt.tight_layout()
    guardar("01_vecino_vs_bilineal")

    # imagenes completas para una reduccion y una ampliacion
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for fila, s in enumerate([0.6, 1.7]):
        vec = reescalar(img_color, s, 'vecino')
        bil = reescalar(img_color, s, 'bilineal')
        for col, (im_, t_) in enumerate([(img_color, "original"),
                                         (vec, f"vecino s={s}"), (bil, f"bilineal s={s}")]):
            mostrar(axes[fila, col], im_, f"{t_}  {im_.shape[0]}x{im_.shape[1]}")
    fig.suptitle("P3 - Imagenes completas (reduccion s=0.6 y ampliacion s=1.7)")
    plt.tight_layout()
    guardar("01b_completas")

    # ---- [2] Segunda imagen con lineas finas (texto) ------------------------
    print("\n[2] Segunda imagen con lineas finas")
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for fila, s in enumerate([0.6, 1.7]):
        vec = reescalar(img_text, s, 'vecino')
        bil = reescalar(img_text, s, 'bilineal')
        for col, (im_, t_) in enumerate([(img_text, "original"),
                                         (vec, f"vecino s={s}"), (bil, f"bilineal s={s}")]):
            mostrar(axes[fila, col], im_, f"{t_}  {im_.shape[0]}x{im_.shape[1]}")
    fig.suptitle("P3.3 - Segunda imagen (texto): lineas finas")
    plt.tight_layout()
    guardar("02_texto")

    # ---- [3] Reduccion seguida de ampliacion (ida y vuelta) -----------------
    print("\n[3] Reduccion seguida de ampliacion (punto 5)")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for fila, modo in enumerate(['vecino', 'bilineal']):
        chico = reescalar(img_gris, 0.6, modo)
        vuelta = reescalar(chico, img_gris.shape[0] / chico.shape[0], modo)
        # se recortan ambas al tamano comun (el redondeo puede dejar 1 pixel de diferencia)
        n0 = min(img_gris.shape[0], vuelta.shape[0])
        n1 = min(img_gris.shape[1], vuelta.shape[1])
        p = psnr(img_gris[:n0, :n1], vuelta[:n0, :n1])
        vuelta = vuelta[:n0, :n1]
        print(f"  {modo}: s=0.6 -> vuelta, PSNR={p:.2f} dB")
        mostrar(axes[fila, 0], img_gris, "original")
        mostrar(axes[fila, 1], chico, f"reducida s=0.6 ({modo})\n{chico.shape[0]}x{chico.shape[1]}")
        mostrar(axes[fila, 2], vuelta, f"vuelta al tamano original\nPSNR={p:.2f} dB")
    fig.suptitle("P3.5 - Reduccion (s=0.6) seguida de ampliacion. Arriba vecino, abajo bilineal")
    plt.tight_layout()
    guardar("03_ida_vuelta")

    # ---- [4] Reescalados sucesivos vs uno solo equivalente ------------------
    print("\n[4] Reescalados sucesivos vs uno solo (punto 6)")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    # cadena: 1.3 * 1.3 = 1.69 aprox
    suc = reescalar(reescalar(img_gris, 1.3, 'bilineal'), 1.3, 'bilineal')
    s_eq = suc.shape[0] / img_gris.shape[0]
    uno = reescalar(img_gris, s_eq, 'bilineal')
    n0 = min(suc.shape[0], uno.shape[0])
    n1 = min(suc.shape[1], uno.shape[1])
    dif = np.abs(suc[:n0, :n1].astype(float) - uno[:n0, :n1].astype(float))
    p = psnr(suc[:n0, :n1], uno[:n0, :n1])
    print(f"  1.3 x 1.3 = {s_eq:.4f} equivalente, PSNR={p:.2f} dB, dif media={dif.mean():.2f}")
    mostrar(axes[0], suc, f"sucesivos: 1.3 x 1.3\n{suc.shape[0]}x{suc.shape[1]}")
    mostrar(axes[1], uno, f"unico: s={s_eq:.3f}\n{uno.shape[0]}x{uno.shape[1]}")
    im = axes[2].imshow(dif, cmap='inferno')
    axes[2].set_title(f"|diferencia|  (PSNR={p:.1f} dB, media={dif.mean():.2f})", fontsize=9)
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle("P3.6 - Dos reescalados sucesivos vs uno solo equivalente (bilineal)")
    plt.tight_layout()
    guardar("04_sucesivos")

    # ---- [5] Aliasing en reduccion ------------------------------------------
    print("\n[5] Aliasing en reduccion (punto 7)")
    # el texto y los bigotes del tigre tienen estructuras finas que aliasan
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
    for fila, (im_, nom) in enumerate([(img_text, "texto"), (img_gris, "tigre (gris)")]):
        mostrar(axes[fila, 0], im_, f"{nom} original")
        for col, s in enumerate([0.75, 0.6, 0.5], start=1):
            vec = reescalar(im_, s, 'vecino')
            mostrar(axes[fila, col], vec, f"vecino s={s}\ndetalle={detalle(vec):.1f}")
    fig.suptitle("P3.7 - Aliasing en reduccion (vecino mas cercano): las lineas finas se rompen")
    plt.tight_layout()
    guardar("05_aliasing")

    # ---- [6] Exploracion adicional: prefiltrado antes de reducir ------------
    print("\n[6] Exploracion adicional: prefiltrado antes de reducir (punto 8)")

    def suavizar(im, k):
        x = im.astype(np.float64)
        r = k // 2
        xp = np.pad(x, r, mode='edge')
        acc = np.zeros_like(x)
        for ddy in range(-r, r + 1):
            for ddx in range(-r, r + 1):
                acc += xp[r + ddy:r + ddy + x.shape[0], r + ddx:r + ddx + x.shape[1]]
        return np.clip(np.round(acc / (k * k)), 0, 255).astype(np.uint8)

    s = 0.5
    sin_pre = reescalar(img_gris, s, 'vecino')
    con_pre3 = reescalar(suavizar(img_gris, 3), s, 'vecino')
    con_pre5 = reescalar(suavizar(img_gris, 5), s, 'vecino')
    print(f"  detalle original      : {detalle(img_gris):.2f}")
    print(f"  detalle sin prefiltro : {detalle(sin_pre):.2f}")
    print(f"  detalle prefiltro 3x3 : {detalle(con_pre3):.2f}")
    print(f"  detalle prefiltro 5x5 : {detalle(con_pre5):.2f}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (im_, t_) in zip(axes, [(sin_pre, f"sin prefiltro\ndetalle={detalle(sin_pre):.1f}"),
                                    (con_pre3, f"prefiltro 3x3\ndetalle={detalle(con_pre3):.1f}"),
                                    (con_pre5, f"prefiltro 5x5\ndetalle={detalle(con_pre5):.1f}")]):
        mostrar(ax, im_, t_)
    fig.suptitle("P3.8 - Prefiltrado antes de reducir (s=0.5, vecino): atenua el aliasing del punto 7")
    plt.tight_layout()
    guardar("06_prefiltro")

    # ---- Preguntas guiadas: traza de un pixel -------------------------------
    print("\n[7] Traza de un pixel (preguntas guiadas)")
    s = 1.3
    i_out, j_out = 100, 137
    N, M = img_gris.shape
    Ns, Ms = tamano_salida(N, s), tamano_salida(M, s)
    sy, sx = Ns / N, Ms / M
    y = (i_out + 0.5) / sy - 0.5
    x = (j_out + 0.5) / sx - 0.5
    y0, x0 = int(np.floor(y)), int(np.floor(x))
    dy, dx = y - y0, x - x0
    va = img_gris[y0, x0]; vb = img_gris[y0, x0 + 1]
    vc = img_gris[y0 + 1, x0]; vd = img_gris[y0 + 1, x0 + 1]
    valor = (va * (1 - dy) * (1 - dx) + vb * (1 - dy) * dx
             + vc * dy * (1 - dx) + vd * dy * dx)
    print("  " + "=" * 55)
    print(f"  Pixel de salida       : ({i_out}, {j_out}), factor s={s}")
    print(f"  Tamano entrada/salida : {N}x{M} -> {Ns}x{Ms}  (s_ef={sy:.4f}, {sx:.4f})")
    print(f"  Coordenada en entrada : ({y:.4f}, {x:.4f})")
    print(f"  4 vecinos (fila,col)  : ({y0},{x0})={va}  ({y0},{x0+1})={vb}")
    print(f"                          ({y0+1},{x0})={vc}  ({y0+1},{x0+1})={vd}")
    print(f"  Pesos                 : dy={dy:.4f}, dx={dx:.4f}")
    print(f"    w_arr_izq=(1-dy)(1-dx)={(1-dy)*(1-dx):.4f}")
    print(f"    w_arr_der=(1-dy)dx    ={(1-dy)*dx:.4f}")
    print(f"    w_aba_izq=dy(1-dx)    ={dy*(1-dx):.4f}")
    print(f"    w_aba_der=dy*dx       ={dy*dx:.4f}")
    print(f"    suma de pesos         = {(1-dy)*(1-dx)+(1-dy)*dx+dy*(1-dx)+dy*dx:.4f}")
    print(f"  Valor bilineal        : {valor:.2f}")
    print(f"  Valor vecino cercano  : {img_gris[int(round(y)), int(round(x))]}")
    print("  " + "=" * 55)

    print("\nFin. Todas las figuras estan en", DIR_FIG)


if __name__ == "__main__":
    main()
