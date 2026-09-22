# =============================================================================
# Pregunta 1: Saturacion selectiva de color (ColorSaturation)
#
# Genera todas las figuras del informe en la carpeta figuras/p1/.
#
# Uso:
# La imagen P1_IMG_2402.tif debe estar en la misma carpeta que este script
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, data

# -----------------------------------------------------------------------------
# Rutas de entrada / salida
# -----------------------------------------------------------------------------
RUTA_IMG = os.path.join("imagenes", "P1_IMG_2402.tif")       
DIR_FIG = os.path.join("figuras", "p1")
os.makedirs(DIR_FIG, exist_ok=True)


# =============================================================================
# IMPLEMENTACION
# =============================================================================

# --- Conversion RGB <-> HSV --------------------------------------------------
def rgb_to_hsv(img_rgb):
    "Convierte imagen RGB a HSV (rango H:[0,360], S:[0,1], V:[0,1])."
    img = img_rgb.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    c_max = np.maximum(np.maximum(R, G), B)
    c_min = np.minimum(np.minimum(R, G), B)
    delta = c_max - c_min

    epsilon = 1e-10
    V = c_max
    S = np.where(c_max > epsilon, delta / (V + epsilon), 0.0)

    H = np.zeros_like(V)
    mask_r = (c_max == R) & (delta > epsilon)
    mask_g = (c_max == G) & (delta > epsilon)
    mask_b = (c_max == B) & (delta > epsilon)

    H[mask_r] = 60.0 * (((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6)
    H[mask_g] = 60.0 * (((B[mask_g] - R[mask_g]) / delta[mask_g]) + 2)
    H[mask_b] = 60.0 * (((R[mask_b] - G[mask_b]) / delta[mask_b]) + 4)

    H = np.where(H < 0, H + 360.0, H)
    return np.stack([H, S, V], axis=-1)


def hsv_to_rgb(img_hsv):
    "Convierte imagen HSV a RGB [0, 1]."
    H = img_hsv[:, :, 0]
    S = img_hsv[:, :, 1]
    V = img_hsv[:, :, 2]

    C = V * S
    X = C * (1.0 - np.abs(((H / 60.0) % 2) - 1.0))
    m = V - C

    R_prime = np.zeros_like(H)
    G_prime = np.zeros_like(H)
    B_prime = np.zeros_like(H)

    mask0 = (H >= 0) & (H < 60)
    mask1 = (H >= 60) & (H < 120)
    mask2 = (H >= 120) & (H < 180)
    mask3 = (H >= 180) & (H < 240)
    mask4 = (H >= 240) & (H < 300)
    mask5 = (H >= 300) & (H < 360)

    R_prime[mask0], G_prime[mask0], B_prime[mask0] = C[mask0], X[mask0], 0
    R_prime[mask1], G_prime[mask1], B_prime[mask1] = X[mask1], C[mask1], 0
    R_prime[mask2], G_prime[mask2], B_prime[mask2] = 0, C[mask2], X[mask2]
    R_prime[mask3], G_prime[mask3], B_prime[mask3] = 0, X[mask3], C[mask3]
    R_prime[mask4], G_prime[mask4], B_prime[mask4] = X[mask4], 0, C[mask4]
    R_prime[mask5], G_prime[mask5], B_prime[mask5] = C[mask5], 0, X[mask5]

    R = R_prime + m
    G = G_prime + m
    B = B_prime + m

    return np.clip(np.stack([R, G, B], axis=-1), 0.0, 1.0)


# --- Conversion RGB <-> CIE L*c*h* -------------------------------------------
def _gamma_to_linear(c: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    "Aplica la correccion gamma no lineal al arreglo sRGB."
    return c ** gamma


def rgb_to_xyz(img_rgb: np.ndarray) -> np.ndarray:
    "Convierte imagen RGB a CIE XYZ usando gamma linealizado y multiplicacion matricial."
    img = img_rgb.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0

    matrix = np.array([
        [0.490, 0.310, 0.200],
        [0.177, 0.813, 0.011],
        [0.000, 0.010, 0.990],
    ])

    rgb_linear = _gamma_to_linear(img, gamma=2.2)
    xyz = rgb_linear @ matrix.T
    return xyz


def xyz_to_lab(img_xyz: np.ndarray) -> np.ndarray:
    "Convierte imagen CIE XYZ a CIELAB (L*a*b*)."
    X = img_xyz[:, :, 0]
    Y = img_xyz[:, :, 1]
    Z = img_xyz[:, :, 2]

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x_norm = X / Xn
    y_norm = Y / Yn
    z_norm = Z / Zn

    delta = 6.0 / 29.0

    def f(t):
        return np.where(t > delta ** 3, np.cbrt(t), (t / (3.0 * delta ** 2)) + (4.0 / 29.0))

    fx, fy, fz = f(x_norm), f(y_norm), f(z_norm)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_lch(img_lab: np.ndarray) -> np.ndarray:
    "Convierte imagen CIELAB a CIE L*C*h*."
    L = img_lab[:, :, 0]
    a = img_lab[:, :, 1]
    b = img_lab[:, :, 2]

    C = np.sqrt(a ** 2 + b ** 2)
    h = np.degrees(np.arctan2(b, a))
    h = np.where(h < 0, h + 360.0, h)
    return np.stack([L, C, h], axis=-1)


def rgb_to_lch(img_rgb: np.ndarray) -> np.ndarray:
    "Funcion principal para obtener CIE L*C*h* a partir de RGB."
    xyz = rgb_to_xyz(img_rgb)
    lab = xyz_to_lab(xyz)
    return lab_to_lch(lab)


def lch_to_lab(lch_img: np.ndarray) -> np.ndarray:
    "Inversa de lab_to_lch: vuelve de coordenadas polares (C*, h*) a cartesianas (a*, b*)."
    L = lch_img[:, :, 0]
    C = lch_img[:, :, 1]
    h_deg = lch_img[:, :, 2]

    h_rad = np.radians(h_deg)
    a = C * np.cos(h_rad)
    b = C * np.sin(h_rad)
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab_img: np.ndarray) -> np.ndarray:
    "Inversa de xyz_to_lab: convierte de CIELAB a CIE XYZ utilizando (Xn, Yn, Zn)."
    L = lab_img[:, :, 0]
    a = lab_img[:, :, 1]
    b = lab_img[:, :, 2]

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    delta = 6.0 / 29.0

    def f_inv(t):
        return np.where(t > delta, t ** 3, 3.0 * (delta ** 2) * (t - (4.0 / 29.0)))

    X = f_inv(fx) * Xn
    Y = f_inv(fy) * Yn
    Z = f_inv(fz) * Zn
    return np.stack([X, Y, Z], axis=-1)


def xyz_to_rgb(xyz_img: np.ndarray) -> np.ndarray:
    "Inversa de rgb_to_xyz: multiplica por la matriz inversa XYZ->RGB y des-linealiza gamma."
    matrix = np.array([
        [0.490, 0.310, 0.200],
        [0.177, 0.813, 0.011],
        [0.000, 0.010, 0.990],
    ])
    inv_matrix = np.linalg.inv(matrix)

    rgb_linear = xyz_img @ inv_matrix.T
    rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
    rgb = rgb_linear ** (1.0 / 2.2)
    return np.clip(rgb, 0.0, 1.0)


def lab_to_rgb(lab_img: np.ndarray) -> np.ndarray:
    "Convierte imagen CIELAB a RGB."
    return xyz_to_rgb(lab_to_xyz(lab_img))


def lch_to_rgb(lch_img: np.ndarray) -> np.ndarray:
    "Funcion principal: convierte de CIE L*C*h* de regreso a RGB."
    return lab_to_rgb(lch_to_lab(lch_img))


def lch_to_rgb_sin_recorte(lch_img: np.ndarray) -> np.ndarray:
    """
    Igual que lch_to_rgb pero sin recortar los valores, para poder detectar que
    pixeles quedan fuera de la gama de sRGB (valores <0 o >1). Solo se usa para
    reportar el tratamiento de fuera de rango, no para producir la imagen final.
    """
    xyz = lab_to_xyz(lch_to_lab(lch_img))
    matrix = np.array([
        [0.490, 0.310, 0.200],
        [0.177, 0.813, 0.011],
        [0.000, 0.010, 0.990],
    ])
    inv_matrix = np.linalg.inv(matrix)
    rgb_linear = xyz @ inv_matrix.T
    # se conserva el signo/rango real: se des-linealiza solo la parte positiva
    rgb = np.sign(rgb_linear) * (np.abs(rgb_linear) ** (1.0 / 2.2))
    return rgb


# -------------------- ColorSaturation ---------------------------------------------
def ColorSaturation(img_rgb: np.ndarray, control_points: list, mode: str = 'HS',
                    m_min: float = 0.0, m_max: float = 4.0, reporte: bool = False):
    """
    Modifica selectivamente la saturacion de una imagen RGB en funcion del tono.

    Parametros:
        img_rgb: Imagen de entrada RGB.
        control_points: Lista de puntos [(h0, m0), (h1, m1), ...] con h_i en [0, 360].
        mode: 'HS' (utiliza HSV) o 'LCH' (utiliza CIE L*C*h*).
        m_min, m_max: rango permitido del factor m. m(h) se recorta a este rango
            antes de aplicarse. El valor NEUTRO es m=1 (deja la componente igual);
            m=m_min es la maxima atenuacion y m=m_max la maxima amplificacion.
        reporte: si es True, retorna (imagen, info) con estadisticas del
            tratamiento de valores fuera de rango.

    Retorna:
        Imagen modificada en formato RGB (o (imagen, info) si reporte=True).
    """
    img = img_rgb.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0

    # 1. Ordenar puntos y garantizar periodicidad circular en tono [0, 360]
    pts = sorted(control_points, key=lambda x: x[0])
    h_pts = [p[0] for p in pts]
    m_pts = [p[1] for p in pts]

    if h_pts[0] != 0:
        h_pts.insert(0, 0)
        m_pts.insert(0, m_pts[-1])
    if h_pts[-1] != 360:
        h_pts.append(360)
        m_pts.append(m_pts[0])

    info = {}

    # 2. Procesamiento segun el modo seleccionado
    if mode.upper() == 'HS':
        hsv = rgb_to_hsv(img)
        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        m_h = np.interp(H, h_pts, m_pts)          # interpolacion lineal circular
        m_h = np.clip(m_h, m_min, m_max)          # se respeta el rango permitido de m
        S_sin = m_h * S                           # g_m(S) = m * S
        S_mod = np.clip(S_sin, 0.0, 1.0)          # tratamiento de fuera de rango: recorte a [0,1]

        # fraccion de pixeles cuya saturacion pedida se salio de [0,1] (clipping)
        info['frac_S_recortada'] = float(np.mean(S_sin > 1.0))
        hsv_mod = np.stack([H, S_mod, V], axis=-1)
        img_res = hsv_to_rgb(hsv_mod)
        # en modo HS, con S en [0,1] y V fijo, el RGB siempre es representable
        info['frac_RGB_fuera_de_gama'] = 0.0

    elif mode.upper() == 'LCH':
        lch = rgb_to_lch(img)
        L = lch[:, :, 0]
        C = lch[:, :, 1]
        h_deg = lch[:, :, 2]

        m_h = np.interp(h_deg, h_pts, m_pts)      # interpolacion lineal circular
        m_h = np.clip(m_h, m_min, m_max)          # se respeta el rango permitido de m
        C_mod = np.maximum(m_h * C, 0.0)          # C*' = m * C*; el croma no puede ser negativo

        lch_mod = np.stack([L, C_mod, h_deg], axis=-1)
        rgb_sin = lch_to_rgb_sin_recorte(lch_mod)  # RGB antes de recortar, para detectar fuera de gama
        # tratamiento de fuera de rango: se recorta por canal a [0,1]
        img_res = np.clip(rgb_sin, 0.0, 1.0)
        fuera = np.any((rgb_sin < 0.0) | (rgb_sin > 1.0), axis=-1)
        info['frac_RGB_fuera_de_gama'] = float(np.mean(fuera))
        info['frac_S_recortada'] = 0.0

    else:
        raise ValueError("El parametro 'mode' debe ser 'HS' o 'LCH'.")

    img_res = np.clip(img_res, 0.0, 1.0)
    return (img_res, info) if reporte else img_res


# =============================================================================
# Familia g_m alternativa (punto 4: efecto del diseno de g_m)
# =============================================================================
# La funcion ColorSaturation usa por defecto g_m lineal: g_m(x) = clip(m*x, 0, 1).
# Para poder analizar como el diseno de g_m cambia el comportamiento, se define
# una familia alternativa "suave" que satura de forma progresiva en vez de
# recortar de golpe. Con ganancia k = m:
#
#     g_suave(u) = k*u / (1 + (k-1)*u)   con u = componente normalizada en [0,1]
#
# Propiedades: g(0)=0, g(1)=1 (nunca se sale de rango, no necesita recorte),
# valor neutro k=1 (identidad), k>1 amplifica y k<1 atenua.

def ColorSaturation_suave(img_rgb, control_points, mode='HS', c_ref=150.0):
    "Igual que ColorSaturation pero con g_m suave en vez de lineal."
    img = img_rgb.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0

    pts = sorted(control_points, key=lambda x: x[0])
    h_pts = [p[0] for p in pts]
    m_pts = [p[1] for p in pts]
    if h_pts[0] != 0:
        h_pts.insert(0, 0); m_pts.insert(0, m_pts[-1])
    if h_pts[-1] != 360:
        h_pts.append(360); m_pts.append(m_pts[0])

    def g_suave(u, k):
        return k * u / (1.0 + (k - 1.0) * u + 1e-12)

    if mode.upper() == 'HS':
        hsv = rgb_to_hsv(img)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        k_h = np.interp(H, h_pts, m_pts)
        S_mod = np.clip(g_suave(np.clip(S, 0, 1), k_h), 0.0, 1.0)
        img_res = hsv_to_rgb(np.stack([H, S_mod, V], axis=-1))
    else:
        lch = rgb_to_lch(img)
        L, C, h_deg = lch[:, :, 0], lch[:, :, 1], lch[:, :, 2]
        k_h = np.interp(h_deg, h_pts, m_pts)
        u = np.clip(C / c_ref, 0.0, 1.0)
        C_mod = c_ref * g_suave(u, k_h)
        img_res = lch_to_rgb(np.stack([L, C_mod, h_deg], axis=-1))
    return np.clip(img_res, 0.0, 1.0)


# =============================================================================
# CODIGO PARA CARGA Y GUARDADO
# =============================================================================
def cargar_imagen(ruta):
    "Carga una imagen RGB y la normaliza a [0, 1]."
    raw = io.imread(ruta)
    if raw.ndim == 3 and raw.shape[2] >= 3:
        img = raw[:, :, :3].astype(np.float64)
    else:
        img = raw.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0 if raw.dtype == np.uint8 else 65535.0
    return img


def guardar(nombre):
    "Guarda la figura actual en figuras/p1/ y la muestra."
    ruta = os.path.join(DIR_FIG, nombre + ".png")
    plt.savefig(ruta, dpi=130, bbox_inches="tight")
    print("  figura ->", ruta)
    plt.show()
    plt.close()


# =============================================================================
# CONFIGURACIONES DE PUNTOS DE CONTROL
# =============================================================================
# Config 1: aumento selectivo de tonos calidos/rojos (alrededor de 0 / 360)
cfg1 = [(0, 2.0), (30, 1.8), (60, 1.0), (300, 1.0), (330, 1.8), (360, 2.0)]
# Config 2: atenuacion selectiva de tonos verdes (alrededor de 120)
cfg2 = [(0, 1.0), (80, 1.0), (120, 0.1), (160, 1.0), (360, 1.0)]
# Config 3: combinacion mixta (aumento azules ~220 y atenuacion calidos ~30)
cfg3 = [(0, 0.2), (30, 0.2), (90, 1.0), (180, 1.0), (220, 2.2), (270, 1.0), (360, 0.2)]

configs = {
    "Config 1 (Aumento Rojos)": cfg1,
    "Config 2 (Atenuacion Verdes)": cfg2,
    "Config 3 (Mixta)": cfg3,
}


# =============================================================================
# EXPERIMENTACION Y ANALISIS
# =============================================================================
def main():
    # ---- Carga de la imagen obligatoria -------------------------------------
    if not os.path.exists(RUTA_IMG):
        raise FileNotFoundError(
            f"No se encontro '{RUTA_IMG}'")
    img_p1 = cargar_imagen(RUTA_IMG)
    print(f"Imagen '{RUTA_IMG}' cargada. Dimensiones: {img_p1.shape}")

    # segunda imagen de contraste cromatico
    img_astro = data.astronaut() / 255.0
    imagenes = {"P1_IMG_2402": img_p1, "Astronauta": img_astro}

    # ---- Histograma de tonos de la imagen original --------------------------
    print("\n[1] Histograma de tonos")
    hsv_orig = rgb_to_hsv(img_p1)
    H_orig = hsv_orig[:, :, 0].ravel()
    S_orig = hsv_orig[:, :, 1].ravel()
    V_orig = hsv_orig[:, :, 2].ravel()

    pesos = S_orig * V_orig
    hist_h, bordes_h = np.histogram(H_orig, bins=36, range=(0, 360), weights=pesos)
    centros_h = 0.5 * (bordes_h[:-1] + bordes_h[1:])
    colores_barra = [plt.cm.hsv(c / 360.0) for c in centros_h]

    plt.figure(figsize=(10, 3.5))
    plt.bar(centros_h, hist_h / hist_h.sum() * 100, width=10,
            color=colores_barra, edgecolor='k', linewidth=0.3)
    plt.xlabel("Tono H (grados)")
    plt.ylabel("Masa relativa (%)")
    plt.title("Histograma de tonos de la imagen original (peso = S*V)")
    plt.xlim(0, 360)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    guardar("01_histograma_tonos")

    top = np.argsort(hist_h)[::-1][:4]
    print("  Sectores dominantes:")
    for i in top:
        print(f"    {int(centros_h[i]-5)}-{int(centros_h[i]+5)} grados: "
              f"{hist_h[i]/hist_h.sum()*100:.1f}%")

    # ---- Punto 1: curvas m(h) con barra de color de tono --------------------
    print("\n[2] Curvas m(h) (Punto 1)")
    h_vals = np.linspace(0, 360, 500)
    fig, ax = plt.subplots(figsize=(10, 4))
    colores_linea = ['tab:red', 'tab:green', 'tab:blue']
    for (name, pts), color in zip(configs.items(), colores_linea):
        pts_s = sorted(pts, key=lambda x: x[0])
        h_p, m_p = [p[0] for p in pts_s], [p[1] for p in pts_s]
        ax.plot(h_vals, np.interp(h_vals, h_p, m_p), label=name, linewidth=2, color=color)
        ax.scatter(h_p, m_p, s=50, color=color, zorder=5)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label="Neutro (m=1)")
    ax.set_xlim(0, 360)
    ax.set_xlabel("Tono H / h* (grados)")
    ax.set_ylabel("Factor m(h)")
    ax.set_title("Curvas m(h) - puntos de control marcados")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    # barra de colores de tono bajo el eje
    grad_h = np.linspace(0, 1, 360)
    ax_cb = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.10,
                          ax.get_position().width, 0.04])
    ax_cb.imshow(plt.cm.hsv(grad_h)[:, :3][None, :, :], aspect='auto', extent=[0, 360, 0, 1])
    ax_cb.set_xticks(np.arange(0, 361, 60))
    ax_cb.set_yticks([])
    ax_cb.set_xlabel("Tono (grados)")
    guardar("02_curvas_mh")

    # ---- Puntos 2 y 3: comparacion visual HS vs LCH -------------------------
    print("\n[3] Comparacion visual HS vs LCH (Puntos 2 y 3)")
    for img_name, img in imagenes.items():
        for cfg_name, pts in configs.items():
            res_hs = ColorSaturation(img, pts, mode='HS')
            res_lch = ColorSaturation(img, pts, mode='LCH')

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].imshow(img); axes[0].set_title(f"Original: {img_name}"); axes[0].axis('off')
            axes[1].imshow(res_hs); axes[1].set_title(f"Modo HS - {cfg_name}"); axes[1].axis('off')
            axes[2].imshow(res_lch); axes[2].set_title(f"Modo LCH - {cfg_name}"); axes[2].axis('off')
            plt.tight_layout()
            slug = cfg_name.split()[1]
            guardar(f"03_hsvslch_{img_name}_{slug}")

    # ---- Punto 3: analisis cuantitativo HS vs LCH ---------------------------
    print("\n[4] Analisis cuantitativo HS vs LCH (Punto 3)")
    print(f"  {'Config':<28} {'RMSE HS':>9} {'RMSE LCH':>9} "
          f"{'dS medio HS':>13} {'dC medio LCH':>14}")
    print("  " + "-" * 76)
    for cfg_name, pts in configs.items():
        res_hs = ColorSaturation(img_p1, pts, mode='HS')
        res_lch = ColorSaturation(img_p1, pts, mode='LCH')
        rmse_hs = np.sqrt(np.mean((res_hs - img_p1) ** 2))
        rmse_lch = np.sqrt(np.mean((res_lch - img_p1) ** 2))
        dS_hs = np.mean(rgb_to_hsv(res_hs)[:, :, 1] - rgb_to_hsv(img_p1)[:, :, 1])
        dC_lch = np.mean(rgb_to_lch(res_lch)[:, :, 1] - rgb_to_lch(img_p1)[:, :, 1])
        print(f"  {cfg_name:<28} {rmse_hs:>9.4f} {rmse_lch:>9.4f} "
              f"{dS_hs:>13.4f} {dC_lch:>14.4f}")

    # mapas de diferencia y de cambio de L* para una config de ejemplo
    cfg_ej = "Config 3 (Mixta)"
    pts_ej = configs[cfg_ej]
    res_hs = ColorSaturation(img_p1, pts_ej, mode='HS')
    res_lch = ColorSaturation(img_p1, pts_ej, mode='LCH')
    diff_hs = np.abs(res_hs - img_p1).mean(axis=-1)
    lch_in = rgb_to_lch(img_p1)
    dL_hs = rgb_to_lch(res_hs)[:, :, 0] - lch_in[:, :, 0]
    dL_lch = rgb_to_lch(res_lch)[:, :, 0] - lch_in[:, :, 0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(img_p1); axes[0, 0].set_title("Original"); axes[0, 0].axis('off')
    axes[0, 1].imshow(res_hs); axes[0, 1].set_title("Modo HS"); axes[0, 1].axis('off')
    axes[0, 2].imshow(res_lch); axes[0, 2].set_title("Modo LCH"); axes[0, 2].axis('off')
    im1 = axes[1, 0].imshow(diff_hs, cmap='magma', vmin=0)
    axes[1, 0].set_title(f"|diferencia| HS (media={diff_hs.mean():.4f})"); axes[1, 0].axis('off')
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    im2 = axes[1, 1].imshow(dL_hs, cmap='coolwarm', vmin=-25, vmax=25)
    axes[1, 1].set_title(f"Cambio dL* modo HS (media={dL_hs.mean():.2f})"); axes[1, 1].axis('off')
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    im3 = axes[1, 2].imshow(dL_lch, cmap='coolwarm', vmin=-25, vmax=25)
    axes[1, 2].set_title(f"Cambio dL* modo LCH (media={dL_lch.mean():.4f})"); axes[1, 2].axis('off')
    fig.colorbar(im3, ax=axes[1, 2], fraction=0.046)
    fig.suptitle(f"Comparacion cuantitativa HS vs LCH - {cfg_ej}")
    plt.tight_layout()
    guardar("04_hsvslch_cuantitativo")

    # ---- Punto 4: exploracion sistematica de parametros ---------------------
    print("\n[5] Exploracion sistematica de parametros (Punto 4)")
    # 4a: amplitud de m
    factores = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
    cfg_base_h = [0, 30, 60, 300, 330, 360]
    fig, axes = plt.subplots(2, len(factores), figsize=(3.2 * len(factores), 6))
    for j, f in enumerate(factores):
        pts_f = [(h, f) if (h < 70 or h > 280) else (h, 1.0) for h in cfg_base_h]
        out = ColorSaturation(img_p1, pts_f, mode='HS')
        S_f = rgb_to_hsv(out)[:, :, 1]
        axes[0, j].imshow(out); axes[0, j].axis('off'); axes[0, j].set_title(f"m={f}", fontsize=10)
        axes[1, j].hist(S_f.ravel(), bins=50, range=(0, 1), color='darkorange')
        axes[1, j].set_xlabel("S'"); axes[1, j].set_yticks([])
        axes[1, j].set_title(f"S'=1: {100*(S_f>=0.999).mean():.1f}%", fontsize=8)
    fig.suptitle("Punto 4a - Amplitud de m (rojos/naranjos)")
    plt.tight_layout()
    guardar("05a_amplitud_m")

    # 4b: separacion / ancho de banda de los puntos de control
    anchos = [20, 60, 120, 200]
    fig, axes = plt.subplots(2, len(anchos), figsize=(3.5 * len(anchos), 6.5))
    centro = 30.0
    for j, w in enumerate(anchos):
        pts_w = [(centro, 2.0), (centro + w / 2, 1.0), (centro + 180, 1.0),
                 (max(0, centro - w / 2), 1.0)]
        pts_s = sorted(pts_w, key=lambda x: x[0])
        h_p, m_p = [p[0] for p in pts_s], [p[1] for p in pts_s]
        out = ColorSaturation(img_p1, pts_w, mode='HS')
        axes[0, j].imshow(out); axes[0, j].axis('off')
        axes[0, j].set_title(f"ancho = {w} grados", fontsize=10)
        axes[1, j].plot(h_vals, np.interp(h_vals, h_p, m_p), color='k')
        axes[1, j].scatter(h_p, m_p, color='red', s=40, zorder=5)
        axes[1, j].set_xlim(0, 360); axes[1, j].set_ylim(0.8, 2.3)
        axes[1, j].set_xlabel("H (grados)"); axes[1, j].grid(alpha=0.3)
    fig.suptitle("Punto 4b - Separacion / posicion de puntos de control")
    plt.tight_layout()
    guardar("05b_separacion")

    # 4c: numero de puntos de control
    n_puntos_list = [2, 4, 8, 16]
    fig, axes = plt.subplots(2, len(n_puntos_list), figsize=(3.5 * len(n_puntos_list), 6.5))

    def m_objetivo(h):
        return 1.0 + 1.2 * np.cos(np.radians(h - 10))

    for j, n in enumerate(n_puntos_list):
        hs = np.linspace(0, 360, n, endpoint=False)
        pts_n = list(zip(hs, m_objetivo(hs)))
        out = ColorSaturation(img_p1, pts_n, mode='HS')
        axes[0, j].imshow(out); axes[0, j].axis('off')
        axes[0, j].set_title(f"n = {n} puntos", fontsize=10)
        pts_s = sorted(pts_n, key=lambda x: x[0])
        h_p, m_p = [p[0] for p in pts_s], [p[1] for p in pts_s]
        axes[1, j].plot(h_vals, m_objetivo(h_vals), '--', color='gray', label='objetivo', lw=1.5)
        axes[1, j].plot(h_vals, np.interp(h_vals, h_p, m_p), color='steelblue',
                        label='lineal/tramos', lw=1.5)
        axes[1, j].scatter(h_p, m_p, color='red', s=30, zorder=5)
        axes[1, j].set_xlim(0, 360); axes[1, j].set_xlabel("H (grados)")
        axes[1, j].legend(fontsize=7); axes[1, j].grid(alpha=0.3)
    fig.suptitle("Punto 4c - Numero de puntos de control")
    plt.tight_layout()
    guardar("05c_num_puntos")

    # ---- Punto 5: limitaciones (clipping) -----------------------------------
    print("\n[6] Limitaciones: clipping (Punto 5)")
    cfg_over = [(0, 4.0), (30, 3.5), (60, 1.0), (300, 1.0), (330, 3.5), (360, 4.0)]
    res_clip_hs, info_hs = ColorSaturation(img_p1, cfg_over, mode='HS', reporte=True)
    res_clip_lch, info_lch = ColorSaturation(img_p1, cfg_over, mode='LCH', reporte=True)

    S_in_c = rgb_to_hsv(img_p1)[:, :, 1].ravel()
    S_out_c = rgb_to_hsv(res_clip_hs)[:, :, 1].ravel()
    C_in_c = rgb_to_lch(img_p1)[:, :, 1].ravel()
    C_out_c = rgb_to_lch(res_clip_lch)[:, :, 1].ravel()
    frac_clip = (S_out_c >= 0.999).mean() * 100
    frac_gama = info_lch['frac_RGB_fuera_de_gama'] * 100

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(img_p1); axes[0, 0].set_title("Original"); axes[0, 0].axis('off')
    axes[0, 1].imshow(res_clip_hs); axes[0, 1].set_title("Modo HS (m_max=4.0)"); axes[0, 1].axis('off')
    axes[0, 2].imshow(res_clip_lch)
    axes[0, 2].set_title(f"Modo LCH (m_max=4.0)\n{frac_gama:.1f}% fuera de gama sRGB"); axes[0, 2].axis('off')
    axes[1, 0].hist(S_in_c, bins=60, range=(0, 1), color='steelblue', alpha=0.7)
    axes[1, 0].set_title("Histograma S (original)"); axes[1, 0].set_xlabel("S")
    axes[1, 1].hist(S_out_c, bins=60, range=(0, 1), color='crimson', alpha=0.8)
    axes[1, 1].axvline(1.0, color='k', lw=1.5, ls='--')
    axes[1, 1].set_title(f"Histograma S' modo HS - {frac_clip:.1f}% en S=1")
    axes[1, 1].set_xlabel("S'")
    axes[1, 2].hist(C_in_c, bins=60, color='steelblue', alpha=0.6, label='C* original')
    axes[1, 2].hist(C_out_c, bins=60, color='orange', alpha=0.6, label="C*'")
    axes[1, 2].set_title("Histograma C* modo LCH"); axes[1, 2].set_xlabel("C*"); axes[1, 2].legend()
    fig.suptitle("Punto 5 - Amplificacion extrema y sus efectos")
    plt.tight_layout()
    guardar("06_clipping")

    S_out_2d = rgb_to_hsv(res_clip_hs)[:, :, 1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].imshow(img_p1); ax[0].set_title("Original"); ax[0].axis('off')
    ax[1].imshow(S_out_2d >= 0.999, cmap='Reds')
    ax[1].set_title(f"Pixeles en S'=1 (recortados): {frac_clip:.1f}%"); ax[1].axis('off')
    plt.tight_layout()
    guardar("06b_mapa_clipping")
    print(f"  Modo HS  - S' en 1: {frac_clip:.2f}% de los pixeles")
    print(f"  Modo LCH - C* medio: {C_in_c.mean():.2f} -> {C_out_c.mean():.2f}")
    print(f"  Modo LCH - RGB fuera de gama sRGB: {frac_gama:.2f}% de los pixeles")

    # ---- Punto 4d: efecto del diseno de g_m ---------------------------------
    print("\n[6b] Efecto del diseno de g_m (Punto 4)")
    # Se compara la g_m lineal (con recorte) contra la g_m suave (racional),
    # usando la misma configuracion de amplificacion fuerte.
    pts_g = [(0, 3.0), (30, 3.0), (60, 1.0), (300, 1.0), (330, 3.0), (360, 3.0)]
    out_lineal = ColorSaturation(img_p1, pts_g, mode='HS')
    out_suave = ColorSaturation_suave(img_p1, pts_g, mode='HS')
    S_lineal = rgb_to_hsv(out_lineal)[:, :, 1].ravel()
    S_suave = rgb_to_hsv(out_suave)[:, :, 1].ravel()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # curvas g_m: como transforman la saturacion para k=3
    u = np.linspace(0, 1, 300)
    k = 3.0
    axes[0, 0].plot(u, np.clip(k * u, 0, 1), color='crimson', lw=2, label='lineal: clip(k·u,0,1)')
    axes[0, 0].plot(u, k * u / (1 + (k - 1) * u), color='seagreen', lw=2, label='suave: k·u/(1+(k-1)u)')
    axes[0, 0].plot(u, u, '--', color='gray', label='neutro (k=1)')
    axes[0, 0].set_xlabel("S"); axes[0, 0].set_ylabel("S'"); axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Familias g_m (k=3)"); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].imshow(out_lineal); axes[0, 1].set_title("g_m lineal (recorta)"); axes[0, 1].axis('off')
    axes[0, 2].imshow(out_suave); axes[0, 2].set_title("g_m suave (satura sin recortar)"); axes[0, 2].axis('off')
    axes[1, 0].axis('off')
    axes[1, 1].hist(S_lineal, bins=60, range=(0, 1), color='crimson', alpha=0.8)
    axes[1, 1].axvline(1.0, color='k', ls='--', lw=1.5)
    axes[1, 1].set_title(f"S' lineal: {100*(S_lineal>=0.999).mean():.1f}% en S=1")
    axes[1, 1].set_xlabel("S'")
    axes[1, 2].hist(S_suave, bins=60, range=(0, 1), color='seagreen', alpha=0.8)
    axes[1, 2].set_title(f"S' suave: {100*(S_suave>=0.999).mean():.1f}% en S=1")
    axes[1, 2].set_xlabel("S'")
    fig.suptitle("Punto 4d - Efecto del diseno de g_m (misma amplificacion k=3)")
    plt.tight_layout()
    guardar("07_familia_gm")
    print(f"  g_m lineal - S' en 1: {100*(S_lineal>=0.999).mean():.2f}%")
    print(f"  g_m suave  - S' en 1: {100*(S_suave>=0.999).mean():.2f}%")

    # ---- Punto 6: exploracion adicional -------------------------------------
    # Objetivo: hacer la imagen mas colorida de forma automatica, equilibrando
    # la paleta. En vez de elegir los tonos a mano, se detectan los picos del
    # histograma de tonos y se realzan todos (m > 1), pero dandole mas ganancia
    # a los tonos secundarios que al dominante. Asi el color principal no se
    # apaga, pero los colores menos presentes suben mas y la imagen gana
    # variedad cromatica (paleta equilibrada).
    print("\n[6c] Exploracion adicional (Punto 6): realce equilibrado de la paleta")
    hist_pesos, bordes = np.histogram(H_orig, bins=36, range=(0, 360), weights=S_orig * V_orig)
    centros = 0.5 * (bordes[:-1] + bordes[1:])
    orden = np.argsort(hist_pesos)[::-1]
 
    # se toman hasta 3 picos separados al menos 40 grados entre si
    picos = []
    for i in orden:
        c = centros[i]
        if hist_pesos[i] <= 0:
            continue
        if all(abs((c - p + 180) % 360 - 180) > 40 for p, _ in picos):
            picos.append((float(c), float(hist_pesos[i])))
        if len(picos) == 3:
            break
 
    # ganancia por pico: inversamente proporcional a su altura (normalizada).
    # El pico mas alto recibe m = m_base; los mas bajos, hasta m_base + realce.
    m_base = 1.3        # todos los tonos suben al menos un poco
    realce = 1.4        # cuanto mas suben los secundarios respecto del dominante
    alturas = np.array([h for _, h in picos])
    alturas_norm = alturas / alturas.max()
    pts_auto = []
    print("  Picos (tono, altura relativa -> ganancia m):")
    for (c, _), hn in zip(picos, alturas_norm):
        m = m_base + realce * (1.0 - hn)     # hn=1 (dominante)->m_base; hn~0->m_base+realce
        pts_auto += [(c % 360, m), ((c + 40) % 360, 1.0), ((c - 40) % 360, 1.0)]
        print(f"    tono={c:5.1f}, altura={hn:.2f} -> m={m:.2f}")
    out_auto = ColorSaturation(img_p1, pts_auto, mode='HS')
 
    # metricas: saturacion media antes y despues (cuantifica "mas colorida")
    S_antes = rgb_to_hsv(img_p1)[:, :, 1].mean()
    S_desp = rgb_to_hsv(out_auto)[:, :, 1].mean()
    print(f"  Saturacion media: {S_antes:.3f} -> {S_desp:.3f}")
 
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    axes[0].bar(centros, hist_pesos / hist_pesos.sum() * 100, width=10,
                color=[plt.cm.hsv(c / 360.0) for c in centros], edgecolor='k', linewidth=0.3)
    for c, _ in picos:
        axes[0].axvline(c, color='k', ls='--')
    axes[0].set_title("Histograma de tonos con picos detectados")
    axes[0].set_xlabel("Tono H (grados)"); axes[0].set_xlim(0, 360)
    h_vals2 = np.linspace(0, 360, 500)
    pts_s = sorted(pts_auto, key=lambda x: x[0])
    hp, mp = [p[0] for p in pts_s], [p[1] for p in pts_s]
    axes[1].plot(h_vals2, np.interp(h_vals2, hp, mp), color='k')
    axes[1].scatter(hp, mp, color='red', s=40, zorder=5)
    axes[1].axhline(1.0, color='gray', ls='--', alpha=0.7, label='neutro (m=1)')
    axes[1].set_title("m(h): todos > 1, secundarios mas altos")
    axes[1].set_xlabel("Tono H (grados)"); axes[1].set_xlim(0, 360)
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    axes[2].imshow(img_p1); axes[2].set_title(f"Original (S media={S_antes:.3f})"); axes[2].axis('off')
    axes[3].imshow(out_auto); axes[3].set_title(f"Resultado (S media={S_desp:.3f})"); axes[3].axis('off')
    fig.suptitle("Punto 6 - Realce equilibrado de la paleta: todos los tonos suben, mas los secundarios")
    plt.tight_layout()
    guardar("08_exploracion_extra")
 

    # ---- Preguntas guiadas: traza de un pixel -------------------------------
    print("\n[7] Traza de un pixel (preguntas guiadas)")
    fila, col = img_p1.shape[0] // 2, img_p1.shape[1] // 2
    rgb_px = img_p1[fila, col]
    hsv_px = rgb_to_hsv(img_p1[fila:fila + 1, col:col + 1])[0, 0]
    H_px, S_px, V_px = hsv_px

    pts_guia = sorted(cfg3, key=lambda x: x[0])
    h_g = [p[0] for p in pts_guia]
    m_g = [p[1] for p in pts_guia]
    m_h_px = float(np.interp(H_px, h_g, m_g))
    S_mod_px = float(np.clip(m_h_px * S_px, 0.0, 1.0))
    rgb_out_px = hsv_to_rgb(np.array([[[H_px, S_mod_px, V_px]]]))[0, 0]

    lch_px = rgb_to_lch(img_p1[fila:fila + 1, col:col + 1])[0, 0]
    L_px, C_px, h_lch_px = lch_px
    m_lch_px = float(np.interp(h_lch_px, h_g, m_g))
    C_mod_px = float(m_lch_px * C_px)

    k = 0
    while k < len(h_g) - 2 and H_px > h_g[k + 1]:
        k += 1
    t = (H_px - h_g[k]) / (h_g[k + 1] - h_g[k] + 1e-12)

    print("  " + "=" * 53)
    print(f"  Pixel: fila={fila}, col={col}")
    print(f"  RGB entrada        : ({rgb_px[0]:.3f}, {rgb_px[1]:.3f}, {rgb_px[2]:.3f})")
    print("  -- Modo HS --")
    print(f"    Tono H           : {H_px:.2f} grados")
    print(f"    Tramo interp.    : [{h_g[k]:.0f}, {h_g[k+1]:.0f}] -> m=({m_g[k]}, {m_g[k+1]})")
    print(f"    Pesos (1-t, t)   : ({1-t:.4f}, {t:.4f})")
    print(f"    m(h)             : {m_h_px:.4f}")
    print(f"    S -> S'          : {S_px:.4f} -> {S_mod_px:.4f}")
    print(f"    RGB salida       : ({rgb_out_px[0]:.3f}, {rgb_out_px[1]:.3f}, {rgb_out_px[2]:.3f})")
    print("  -- Modo LCH --")
    print(f"    Tono h*          : {h_lch_px:.2f} grados")
    print(f"    m(h*)            : {m_lch_px:.4f}")
    print(f"    C* -> C*'        : {C_px:.4f} -> {C_mod_px:.4f}")
    print("  " + "=" * 53)

    print("\nFin. Todas las figuras estan en", DIR_FIG)


if __name__ == "__main__":
    main()
