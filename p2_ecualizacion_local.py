# =============================================================================
# Pregunta 2: Ecualizacion local de histograma y control de contraste
#
# Genera todas las figuras del informe en la carpeta figuras/p2/.
#
# Uso:
# La imagen P2_IMG_2423.tif debe estar en la carpeta "imagenes", ubicada en la
# misma carpeta que este script (imagenes/P2_IMG_2423.tif).
#
# Librerias: cv2 se usa unicamente como referencia de CLAHE para las comparaciones.
# Su codigo no forma parte de la solucion propia. 
# =============================================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, data
import cv2   # solo para CLAHE de referencia

# -----------------------------------------------------------------------------
# Rutas de entrada / salida
# -----------------------------------------------------------------------------
RUTA_IMG = os.path.join("imagenes", "P2_IMG_2423.tif")
DIR_FIG = os.path.join("figuras", "p2")
os.makedirs(DIR_FIG, exist_ok=True)

# -----------------------------------------------------------------------------
# UTILIDADES
# -----------------------------------------------------------------------------

def cargar_gris(ruta):
    "Carga una imagen y la devuelve en escala de grises uint8."
    raw = io.imread(ruta)
    a = raw.astype(np.float64)
    if a.ndim == 3:                      # si es a color, se pasa a gris
        a = a[:, :, :3]
        gris = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    else:
        gris = a
    # se normaliza al rango real de la imagen -> [0, 255]
    mn, mx = gris.min(), gris.max()
    if mx > mn:
        gris = (gris - mn) / (mx - mn) * 255.0
    return np.clip(np.round(gris), 0, 255).astype(np.uint8)

def imhist(X, n_bins=256):
    """
    Histograma de una imagen en escala de grises (basado en la funcion imhist
    de los cuadernos del curso, pero vectorizada con np.bincount para eficiencia).
    Devuelve el vector de cuentas de largo n_bins.
    """
    idx = (X.astype(np.int64) * n_bins) // 256
    return np.bincount(idx.ravel(), minlength=n_bins).astype(np.float64)


def guardar(nombre):
    "Guarda la figura actual en figuras/p2/ y la muestra."
    ruta = os.path.join(DIR_FIG, nombre + ".png")
    plt.savefig(ruta, dpi=130, bbox_inches="tight")
    print("  figura ->", ruta)
    plt.show()
    plt.close()


def entropia(img):
    "Entropia del histograma (bits/pixel): mide cuanta informacion tonal hay."
    h = imhist(img)
    p = h / h.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def contraste_local(img, bloque=8):
    "Desviacion estandar promedio en bloques: mide el contraste local."
    x = img.astype(np.float64)
    N, M = x.shape
    N2, M2 = (N // bloque) * bloque, (M // bloque) * bloque
    b = x[:N2, :M2].reshape(N2 // bloque, bloque, M2 // bloque, bloque)
    return float(b.std(axis=(1, 3)).mean())


# =============================================================================
# ECUALIZACION GLOBAL CLASICA (referencia)
# =============================================================================
def cdf_desde_histograma(hist):
    "CDF normalizada acumulando el histograma."
    return np.cumsum(hist) / hist.sum()


def lut_ecualizacion(hist, n_bins=256):
    """
    Construye la LUT de ecualizacion (256 valores) a partir de un histograma:
        CDF(b) = suma acumulada normalizada
        T(v)   = (L-1) * CDF(bin de v),  con L=256
    """
    total = hist.sum()
    if total <= 0:
        return np.arange(256, dtype=np.float64)
    cdf = np.cumsum(hist) / total
    T_bins = 255.0 * cdf
    v = np.arange(256)
    b = (v * n_bins) // 256
    return T_bins[b]


def ecualizacion_global(img, n_bins=256):
    "Ecualizacion global clasica de histograma. Para referencia."
    lut = lut_ecualizacion(imhist(img, n_bins), n_bins)
    return np.clip(np.round(lut[img]), 0, 255).astype(np.uint8)


# =============================================================================
# CONTROL DE CONTRASTE (PROPIO)
# =============================================================================
# La pendiente de la transformacion de ecualizacion en un nivel v es
#     dT/dv = (L-1) * h(v) / N
# es proporcional a la altura del histograma. Un histograma muy
# concentrado (zona homogenea con ruido) produce pendientes grandes y amplifica
# el ruido. Controlar el contraste = acotar la pendiente = acotar la altura del
# histograma.
#
# El mecanismo (inspirado en CLAHE pero implementado desde cero) tiene un
# parametro principal beta >= 1:
#     h_max = beta * (N / B)   (beta veces la altura media)
# El histograma se recorta en h_max y la masa recortada se redistribuye de
# forma uniforme entre todos los bins. Se itera unas pocas veces porque la
# redistribucion puede volver a superar el limite.
#   beta = 1     -> histograma casi uniforme -> T casi identidad (maxima
#                   restriccion, la imagen no cambia)
#   beta -> inf  -> no se recorta nada -> ecualizacion local sin limite

def limitar_histograma(hist, beta, n_iter=8):
    "Recorta el histograma en beta*(N/B) y redistribuye el exceso (iterando)."
    if beta is None or not np.isfinite(beta):
        return hist
    B = len(hist)
    N = hist.sum()
    if N <= 0:
        return hist
    h_max = max(beta * N / B, 1e-9)
    h = hist.copy()
    for _ in range(n_iter):
        exceso = np.sum(np.maximum(h - h_max, 0.0))
        if exceso <= 1e-9:
            break
        h = np.minimum(h, h_max) + exceso / B
    return h


# =============================================================================
# ECUALIZACION LOCAL SOBRE UNA MALLA DE REGIONES
# =============================================================================
def posiciones(N, tam, paso):
    """
    Origenes de las regiones a lo largo de un eje de largo N.
    - Si tam >= N: una sola region que cubre todo el eje (caso global).
    - Si no: origenes 0, paso, 2*paso, ... mientras la region quepa, mas el
      origen N-tam para cubrir el borde final.
    Devuelve (origenes, tam_efectivo).
    """
    tam = int(min(tam, N))
    paso = max(1, int(paso))
    if tam >= N:
        return np.array([0]), N
    ini = list(range(0, N - tam + 1, paso))
    if ini[-1] != N - tam:
        ini.append(N - tam)
    return np.array(ini, dtype=int), tam


def ecualizacion_local(img, tam_region=(64, 64), paso=(32, 32), n_bins=256,
                       beta=None, combinar='ponderada'):
    """
    Ecualizacion local de histograma sobre una malla de regiones.

    Parametros:
        tam_region : (alto, ancho) de cada region.
        paso       : (dy, dx) distancia entre centros de regiones vecinas.
                     overlap = 1 - paso/tam_region.
        n_bins     : numero de bins de los histogramas locales.
        beta       : parametro de control de contraste (None = sin limite).
        combinar   : como se combinan las transformaciones de regiones que se
                     solapan sobre un mismo pixel:
                        'ponderada' -> promedio con pesos triangulares (suave)
                        'cercana'   -> se usa la region del centro mas cercano

    Con tam_region que cubre toda la imagen, reproduce la ecualizacion global.
    """
    img = np.asarray(img, dtype=np.uint8)
    N, M = img.shape
    iy, th = posiciones(N, tam_region[0], paso[0])
    ix, tw = posiciones(M, tam_region[1], paso[1])

    # 1. Construir la LUT de cada region (con control de contraste opcional)
    luts = np.empty((len(iy), len(ix), 256), dtype=np.float64)
    for a, y0 in enumerate(iy):
        for b, x0 in enumerate(ix):
            region = img[y0:y0 + th, x0:x0 + tw]
            hist = imhist(region, n_bins)              # histograma de la region
            hist = limitar_histograma(hist, beta)      # control de contraste
            luts[a, b] = lut_ecualizacion(hist, n_bins)  # CDF -> transformacion

    cy = iy + (th - 1) / 2.0        # centros de las regiones
    cx = ix + (tw - 1) / 2.0

    # 2. Combinar las transformaciones sobre cada pixel
    if combinar == 'cercana':
        ky = np.argmin(np.abs(np.arange(N)[:, None] - cy[None, :]), axis=1)
        kx = np.argmin(np.abs(np.arange(M)[:, None] - cx[None, :]), axis=1)
        salida = luts[ky[:, None], kx[None, :], img]
    else:
        # pesos triangulares: cada region pesa 1 en su centro y 0 en su borde.
        # El numero de regiones que contribuyen por eje es tam/paso, de modo
        # que el overlap controla el suavizado.
        y = np.arange(N, dtype=np.float64)
        x = np.arange(M, dtype=np.float64)
        Wy = np.maximum(0.0, 1.0 - np.abs(y[None, :] - cy[:, None]) / (th / 2.0))
        Wx = np.maximum(0.0, 1.0 - np.abs(x[None, :] - cx[:, None]) / (tw / 2.0))

        num = np.zeros((N, M), dtype=np.float64)
        den = np.zeros((N, M), dtype=np.float64)
        for a in range(len(iy)):
            fy = np.nonzero(Wy[a])[0]
            if fy.size == 0:
                continue
            y0, y1 = fy[0], fy[-1] + 1
            wy = Wy[a, y0:y1][:, None]
            for b in range(len(ix)):
                fx = np.nonzero(Wx[b])[0]
                if fx.size == 0:
                    continue
                x0, x1 = fx[0], fx[-1] + 1
                w = wy * Wx[b, x0:x1][None, :]
                num[y0:y1, x0:x1] += w * luts[a, b][img[y0:y1, x0:x1]]
                den[y0:y1, x0:x1] += w
        # pixeles sin ninguna region con peso (fronteras con overlap 0): centro mas cercano
        faltan = den < 1e-9
        if np.any(faltan):
            ky = np.argmin(np.abs(np.arange(N)[:, None] - cy[None, :]), axis=1)
            kx = np.argmin(np.abs(np.arange(M)[:, None] - cx[None, :]), axis=1)
            alt = luts[ky[:, None], kx[None, :], img]
            num = np.where(faltan, alt, num)
            den = np.where(faltan, 1.0, den)
        salida = num / den

    return np.clip(np.round(salida), 0, 255).astype(np.uint8)


def clahe_referencia(img, clip=3.0, tiles=(8, 8)):
    "CLAHE de OpenCV, usado unicamente como referencia de comparacion."
    return cv2.createCLAHE(clipLimit=float(clip),
                           tileGridSize=(int(tiles[1]), int(tiles[0]))).apply(img)


# =============================================================================
# EXPERIMENTACION Y ANALISIS
# =============================================================================
def _panel_img(ax, img, titulo):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(titulo, fontsize=9)
    ax.axis('off')


def main():
    if not os.path.exists(RUTA_IMG):
        raise FileNotFoundError(
            f"No se encontro '{RUTA_IMG}'.")
    img = cargar_gris(RUTA_IMG)
    N, M = img.shape
    print(f"Imagen '{RUTA_IMG}' cargada. Dimensiones: {img.shape}")
    print(f"  media={img.mean():.1f}  std={img.std():.1f}  "
          f"entropia={entropia(img):.2f} bits  contraste_local={contraste_local(img):.2f}")

    # ---- [1] Verificacion: una sola region reproduce la global --------------
    print("\n[1] Verificacion del caso limite (una region = ecualizacion global)")
    glob = ecualizacion_global(img)
    local_1region = ecualizacion_local(img, tam_region=(N, M), paso=(N, M))
    dif_max = int(np.max(np.abs(local_1region.astype(int) - glob.astype(int))))
    print(f"  max|local(1 region) - global| = {dif_max}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    _panel_img(axes[0, 0], img, "Original")
    _panel_img(axes[0, 1], glob, "Ecualizacion global clasica")
    _panel_img(axes[0, 2], local_1region, f"Algoritmo propio, 1 region {N}x{M}\nmax|dif|={dif_max}")
    for ax, im_, t_ in zip(axes[1], [img, glob, local_1region], ["", "", ""]):
        ax.bar(np.arange(256), imhist(im_), width=1.0, color='steelblue')
        ax.set_xlim(0, 255); ax.set_yticks([])
    axes[1, 0].set_ylabel("histograma")
    fig.suptitle("P2.1 - Con una sola region que cubre la imagen, se recupera la ecualizacion global")
    plt.tight_layout()
    guardar("01_verificacion_global")

    # ---- Comparacion principal ----------------------------------------------
    print("\n[2] Comparacion principal (original, global, local, propuesta, CLAHE)")
    tam, paso, beta, clip = (64, 64), (32, 32), 15.0, 10.0
    loc = ecualizacion_local(img, tam, paso, beta=None)
    mio = ecualizacion_local(img, tam, paso, beta=beta)
    cla = clahe_referencia(img, clip=clip, tiles=(N // tam[0], M // tam[1]))

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    conjunto = [(img, "Original"), (glob, "Global clasica"),
                (loc, f"Local sin limite\nregion={tam}, paso={paso}"),
                (mio, f"Propuesta (beta={beta})\nregion={tam}, paso={paso}"),
                (cla, f"CLAHE referencia\nclip={clip}, tiles=({N//tam[0]},{M//tam[1]})")]
    for j, (im_, t_) in enumerate(conjunto):
        m_ent, m_cl = entropia(im_), contraste_local(im_)
        _panel_img(axes[0, j], im_, f"{t_}\nH={m_ent:.2f}b, c.local={m_cl:.1f}")
        axes[1, j].bar(np.arange(256), imhist(im_), width=1.0, color='steelblue')
        axes[1, j].set_xlim(0, 255); axes[1, j].set_yticks([])
    fig.suptitle("P2 - Comparacion principal sobre la imagen obligatoria")
    plt.tight_layout()
    guardar("02_comparacion_principal")

    # ---- [2b] Tamano de region y overlap, por separado ----------------------
    print("\n[3] Geometria de la malla: tamano de region y overlap")
    # (a) tamano de region con overlap fijo 0.5
    tams = [16, 32, 64, 128, 256]
    fig, axes = plt.subplots(1, len(tams), figsize=(3.2 * len(tams), 3.6))
    for j, t in enumerate(tams):
        out = ecualizacion_local(img, (t, t), (t // 2, t // 2), beta=None)
        _panel_img(axes[j], out, f"region {t}x{t}, overlap=0.5\nc.local={contraste_local(out):.1f}")
    fig.suptitle("P2.2a - Efecto del tamano de region (overlap fijo = 0.5, sin limite)")
    plt.tight_layout()
    guardar("03a_tamano_region")

    # (b) overlap con tamano de region fijo 64
    T = 64
    pasos = [64, 32, 16, 8]
    fig, axes = plt.subplots(1, len(pasos), figsize=(3.2 * len(pasos), 3.6))
    for j, p in enumerate(pasos):
        t0 = time.perf_counter()
        out = ecualizacion_local(img, (T, T), (p, p), beta=None)
        dt = time.perf_counter() - t0
        _panel_img(axes[j], out, f"paso {p} (overlap={1-p/T:.2f})\n{dt*1000:.0f} ms")
    fig.suptitle("P2.2b - Efecto del overlap (region fija 64x64, sin limite)")
    plt.tight_layout()
    guardar("03b_overlap")

    # ---- [3] Numero de bins -------------------------------------------------
    print("\n[4] Numero de bins")
    bins_list = [8, 16, 32, 64, 128, 256]
    tams_b = [16, 64, 256]
    fig, axes = plt.subplots(len(tams_b), len(bins_list),
                             figsize=(2.6 * len(bins_list), 2.9 * len(tams_b)))
    for i, t in enumerate(tams_b):
        for j, nb in enumerate(bins_list):
            out = ecualizacion_local(img, (t, t), (t // 2, t // 2), n_bins=nb, beta=None)
            _panel_img(axes[i, j], out, f"region {t}, bins {nb}\nH={entropia(out):.2f}")
    fig.suptitle("P2.3 - Numero de bins frente a tamano de region (sin limite)")
    plt.tight_layout()
    guardar("04_bins")

    # ---- [4] Barrido del control de contraste (beta) ------------------------
    print("\n[5] Barrido del parametro de control de contraste (beta)")
    betas = [1.0, 3.0, 6.0, 12.0, 25.0, np.inf]
    fig, axes = plt.subplots(1, len(betas), figsize=(3.2 * len(betas), 3.6))
    ent_b, cl_b = [], []
    for j, b in enumerate(betas):
        out = ecualizacion_local(img, tam, paso, beta=(None if not np.isfinite(b) else b))
        ent_b.append(entropia(out)); cl_b.append(contraste_local(out))
        _panel_img(axes[j], out, f"beta={b}\nc.local={contraste_local(out):.1f}")
    fig.suptitle("P2.4a - Barrido de beta (beta=1 -> identidad, beta=inf -> sin limite)")
    plt.tight_layout()
    guardar("05a_beta")

    # accion matematica de beta sobre el histograma y la LUT de una region
    y0, x0 = 200, 60      # region de fondo (relativamente homogenea)
    region = img[y0:y0 + tam[0], x0:x0 + tam[1]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    h0 = imhist(region, 256)
    axes[0].bar(np.arange(256), h0, width=1.0, color='gray')
    axes[0].set_title(f"Histograma de una region {tam[0]}x{tam[1]}", fontsize=10)
    axes[0].set_xlabel("nivel"); axes[0].set_xlim(0, 120)
    for b in [1.0, 1.5, 3.0, 10.0]:
        axes[1].plot(limitar_histograma(h0, b), lw=1.2, label=f"beta={b}")
        axes[2].plot(lut_ecualizacion(limitar_histograma(h0, b), 256), lw=1.5, label=f"beta={b}")
    axes[1].plot(h0, 'k--', lw=0.8, label='sin limite')
    axes[1].axhline(h0.sum() / 256, color='r', ls=':', label='altura media N/B')
    axes[1].set_title("Histograma recortado y redistribuido", fontsize=10)
    axes[1].set_xlabel("nivel"); axes[1].set_xlim(0, 120); axes[1].legend(fontsize=7)
    axes[2].plot(np.arange(256), 'r:', lw=1, label='identidad')
    axes[2].set_title("LUT resultante T(v)", fontsize=10)
    axes[2].set_xlabel("v"); axes[2].set_ylabel("T(v)"); axes[2].legend(fontsize=7)
    fig.suptitle("P2.4b - Accion de beta: acota la altura del histograma = acota la pendiente de T")
    plt.tight_layout()
    guardar("05b_beta_lut")

    # curva beta -> metricas
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bs = [1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0]
    ent_c = [entropia(ecualizacion_local(img, tam, paso, beta=b)) for b in bs]
    cl_c = [contraste_local(ecualizacion_local(img, tam, paso, beta=b)) for b in bs]
    ax[0].plot(bs, ent_c, 'o-'); ax[0].axhline(entropia(img), color='gray', ls='--', label='original')
    ax[0].set_xscale('log'); ax[0].set_xlabel("beta"); ax[0].set_ylabel("entropia [bits]")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].plot(bs, cl_c, 'o-', color='crimson')
    ax[1].axhline(contraste_local(img), color='gray', ls='--', label='original')
    ax[1].set_xscale('log'); ax[1].set_xlabel("beta"); ax[1].set_ylabel("contraste local")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.suptitle("P2.4c - Transicion continua desde la identidad hasta la ecualizacion sin limite")
    plt.tight_layout()
    guardar("05c_curva_beta")

    # ---- [5] Comparacion con CLAHE en dos imagenes y varias configuraciones --
    print("\n[6] Comparacion con CLAHE en dos imagenes y varias configuraciones")
    configs = [((32, 32), 8.0, 6.0), ((64, 64), 15.0, 10.0), ((128, 128), 22.0, 12.0)]
    imagenes_clahe = [(img, "P2_IMG_2423", "06_vs_clahe"),
                      (data.moon(), "moon", "06b_vs_clahe_moon")]
    for img_c, nombre_c, archivo_c in imagenes_clahe:
        Nc, Mc = img_c.shape
        fig, axes = plt.subplots(len(configs), 4, figsize=(14, 3.4 * len(configs)))
        for i, (tg, bt, cl) in enumerate(configs):
            tiles = (max(1, Nc // tg[0]), max(1, Mc // tg[1]))
            loc_i = ecualizacion_local(img_c, tg, (tg[0] // 2, tg[1] // 2), beta=None)
            mio_i = ecualizacion_local(img_c, tg, (tg[0] // 2, tg[1] // 2), beta=bt)
            cla_i = clahe_referencia(img_c, clip=cl, tiles=tiles)
            for j, (im_, t_) in enumerate([(img_c, "Original"), (loc_i, "Local sin limite"),
                                           (mio_i, f"Propuesta beta={bt}"), (cla_i, f"CLAHE clip={cl}")]):
                _panel_img(axes[i, j], im_,
                           f"{t_}\nregion={tg}, c.local={contraste_local(im_):.1f}")
        fig.suptitle(f"P2.5 - Propuesta frente a CLAHE en distintas configuraciones ({nombre_c})")
        plt.tight_layout()
        guardar(archivo_c)

    # ---- [6] Caso indeseable: ruido en zona homogenea -----------------------
    print("\n[7] Caso indeseable: amplificacion de ruido en zona homogenea")
    tam_r, paso_r = (48, 48), (24, 24)
    loc_r = ecualizacion_local(img, tam_r, paso_r, beta=None)
    mio_r = ecualizacion_local(img, tam_r, paso_r, beta=1.5)
    cla_r = clahe_referencia(img, clip=2.0, tiles=(N // tam_r[0], M // tam_r[1]))

    # zona homogenea = un bloque plano dentro de la imagen.
    # Se busca automaticamente el bloque de menor desviacion estandar para que
    # funcione con cualquier tamano de imagen.
    s = max(48, N // 10)
    mejor = None
    paso_busq = max(20, s // 2)
    for yy in range(0, N - s, paso_busq):
        for xx in range(0, M - s, paso_busq):
            st = img[yy:yy + s, xx:xx + s].std()
            if mejor is None or st < mejor[0]:
                mejor = (st, yy, xx)
    _, yh, xh = mejor
    def crop(a): return a[yh:yh + s, xh:xh + s]
    print(f"  zona homogenea en (fila={yh}, col={xh}), tamano {s}x{s}, std original={crop(img).std():.2f}")
    print(f"  std region homogenea: original={crop(img).std():.1f}, "
          f"local sin limite={crop(loc_r).std():.1f}, propuesta={crop(mio_r).std():.1f}, "
          f"CLAHE={crop(cla_r).std():.1f}")

    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for j, (im_, t_) in enumerate([(img, "Original"), (loc_r, "Local sin limite"),
                                   (mio_r, "Propuesta beta=1.5"), (cla_r, "CLAHE clip=2.0")]):
        _panel_img(axes[0, j], im_, t_)
        axes[0, j].add_patch(Rectangle((xh, yh), s, s, ec='red', fc='none', lw=1.2))
        _panel_img(axes[1, j], crop(im_), f"zona homogenea, std={crop(im_).std():.1f}")
    fig.suptitle("P2.6 - La ecualizacion local sin limite amplifica el ruido de las zonas planas")
    plt.tight_layout()
    guardar("07a_ruido")

    # histograma y CDF de la zona homogenea 
    reg = crop(img)
    h = imhist(reg, 256)
    rmin, rmax = int(reg.min()), int(reg.max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(np.arange(256), h, width=1.0, color='gray')
    axes[0].set_xlim(max(0, rmin - 5), rmax + 5)
    axes[0].set_title(f"Histograma de la zona homogenea ({rmin}..{rmax})", fontsize=10)
    axes[1].plot(cdf_desde_histograma(h), 'k')
    axes[1].set_xlim(max(0, rmin - 5), rmax + 5)
    axes[1].set_title("CDF", fontsize=10)
    lut_sin = lut_ecualizacion(h, 256)
    lut_lim = lut_ecualizacion(limitar_histograma(h, 1.5), 256)
    axes[2].plot(lut_sin, 'k', label='sin limite')
    axes[2].plot(lut_lim, 'b', label='beta=1.5')
    axes[2].plot(np.arange(256), 'r:', label='identidad')
    axes[2].set_xlim(max(0, rmin - 5), rmax + 5)
    pend = np.max(np.diff(lut_sin))
    axes[2].set_title(f"LUT: pendiente maxima={pend:.0f}", fontsize=10)
    axes[2].legend(fontsize=8)
    fig.suptitle("P2.8 - Por que una zona homogenea amplifica el ruido (pendiente grande de T)")
    plt.tight_layout()
    guardar("07b_zona_homogenea")

    # ---- [7] Artefactos de frontera -----------------------------------------
    print("\n[8] Artefactos de frontera entre regiones")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    yb, xb, sb = 150, 400, 250
    casos = [('cercana', (64, 64), (64, 64)),      # sin overlap ni mezcla
             ('cercana', (64, 64), (32, 32)),      # overlap pero sin mezcla
             ('ponderada', (64, 64), (32, 32))]    # overlap con mezcla
    for j, (comb, tg, pg) in enumerate(casos):
        out = ecualizacion_local(img, tg, pg, beta=3.0, combinar=comb)
        _panel_img(axes[0, j], out,
                   f"{comb}, paso={pg[0]} (overlap={1-pg[0]/tg[0]:.2f})")
        _panel_img(axes[1, j], out[yb:yb + sb, xb:xb + sb], "recorte ampliado")
    fig.suptitle("P2.7 - Artefactos de frontera: sin mezcla aparecen bloques, la mezcla ponderada los elimina")
    plt.tight_layout()
    guardar("08_artefactos")

    # ---- [8] Exploracion adicional: iteraciones del recorte del histograma ---
    print("\n[8b] Exploracion adicional: numero de iteraciones del recorte")
    yr, xr = 200, 60
    region_e = img[yr:yr + tam[0], xr:xr + tam[1]]
    h_e = imhist(region_e, 256)
    iters = [1, 2, 4, 8, 16]
    beta_e = 1.5
    B_e = len(h_e)
    N_e = h_e.sum()
    h_max_e = max(beta_e * N_e / B_e, 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    excesos = []
    for ni in iters:
        h_lim = limitar_histograma(h_e, beta_e, n_iter=ni)
        exceso_rest = float(np.sum(np.maximum(h_lim - h_max_e, 0.0)))
        excesos.append(exceso_rest)
        axes[0].plot(lut_ecualizacion(h_lim, 256), lw=1.4, label=f"{ni} iter")
    axes[0].plot(np.arange(256), 'r:', lw=1, label='identidad')
    axes[0].set_xlim(0, 120); axes[0].set_xlabel("v"); axes[0].set_ylabel("T(v)")
    axes[0].set_title(f"LUT segun numero de iteraciones (beta={beta_e})", fontsize=10)
    axes[0].legend(fontsize=7)
    axes[1].plot(iters, excesos, 'o-', color='crimson')
    axes[1].set_xlabel("numero de iteraciones"); axes[1].set_ylabel("masa aun sobre h_max")
    axes[1].set_title("El exceso residual tiende a cero al iterar", fontsize=10)
    axes[1].grid(alpha=0.3)
    fig.suptitle("P2.8b - Exploracion adicional: convergencia del recorte iterativo del histograma")
    plt.tight_layout()
    guardar("09_iteraciones")
    print(f"  exceso residual por iteracion: {[round(e, 1) for e in excesos]}")

    # ---- Preguntas guiadas: traza de una region -----------------------------
    print("\n[9] Traza de una region (preguntas guiadas)")
    ry0, rx0 = 300, 500
    tam_t = (64, 64)
    region_t = img[ry0:ry0 + tam_t[0], rx0:rx0 + tam_t[1]]
    hist_t = imhist(region_t, 256)
    hist_lim = limitar_histograma(hist_t, 2.5)
    lut_t = lut_ecualizacion(hist_lim, 256)
    v_ej = int(region_t[region_t.shape[0] // 2, region_t.shape[1] // 2])
    print("  " + "=" * 55)
    print(f"  Region en (fila={ry0}, col={rx0}), tamano {tam_t}")
    print(f"  Pixeles en la region : {region_t.size}")
    print(f"  Rango de niveles     : {region_t.min()}..{region_t.max()}")
    print(f"  N/B (altura media)   : {hist_t.sum()/256:.1f}")
    print(f"  h_max con beta=2.5   : {2.5*hist_t.sum()/256:.1f}")
    print(f"  Nivel del pixel centro : v={v_ej}")
    print(f"  CDF(v)               : {cdf_desde_histograma(hist_lim)[v_ej]:.4f}")
    print(f"  T(v) = salida        : {lut_t[v_ej]:.1f}")
    print("  " + "=" * 55)

    print("\nFin. Todas las figuras estan en", DIR_FIG)


if __name__ == "__main__":
    main()
