import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from disparity.methods import InputPair
from aruco_utils import detect_charuco_markers, estimate_camera_pose_with_homography

def rect_maps(calib, alpha=0):
    """
    Calcula los mapas de remapeo para la rectificación estéreo.
    Args:
        calib (dict): Diccionario con los parámetros de calibración estéreo.
        alpha (float): Parámetro de escalado para la rectificación.
    Returns:
        stereo_maps (dict): Diccionario con los mapas de remapeo para ambas cámaras.
    """
    left_K = calib['left_K']
    left_dist = calib['left_dist']
    right_K = calib['right_K']
    right_dist = calib['right_dist']
    R = calib['R']
    T = calib['T']
    img_size = tuple(calib['image_size'])

    if T.ndim == 1:
        T = T.reshape(3, 1)
    elif T.shape == (1, 3):
        T = T.T
    
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        left_K, left_dist,
        right_K, right_dist,
        img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha
    )

    # Calcula los mapas de remapeo para ambas cámaras
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(
        left_K, left_dist, R1, P1, img_size, cv2.CV_32FC1
    )
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(
        right_K, right_dist, R2, P2, img_size, cv2.CV_32FC1
    )
    
    stereo_maps = {
        'left_map_x': left_map_x,
        'left_map_y': left_map_y,
        'right_map_x': right_map_x,
        'right_map_y': right_map_y
    }

    rectification = {
        'R1': R1,
        'R2': R2,
        'P1': P1,
        'P2': P2,
        'Q': Q
    }

    return stereo_maps, rectification

def rectify_stereo_pair(left_img, right_img, stereo_maps):
    """
    Rectifica un par estéreo usando mapas pre-computados.
    Args:
        left_img (str): Ruta a la imagen izquierda.
        right_img (str): Ruta a la imagen derecha.
        stereo_maps (dict): Diccionario con los mapas de remapeo para ambas cámaras.
    Returns:
        left_rect (ndarray): Imagen izquierda rectificada.
        right_rect (ndarray): Imagen derecha rectificada.
    """
    left_img  = cv2.imread(left_img)
    right_img = cv2.imread(right_img)
    left_rect = cv2.remap(left_img, 
                         stereo_maps['left_map_x'], 
                         stereo_maps['left_map_y'], 
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
    
    right_rect = cv2.remap(right_img,
                          stereo_maps['right_map_x'],
                          stereo_maps['right_map_y'],
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    
    return left_rect, right_rect

def compare_rectified_imgs(original, rectified):
    """
    Compara imágenes originales y rectificadas lado a lado
    Args:
        original (ndarray): Imagen original.
        rectified (ndarray): Imagen rectificada.
    """
    original = cv2.imread(original)
    combined = np.hstack([original, rectified])
    plt.figure(figsize=(18,9))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title("Original (izquierda) vs Rectificada (derecha)", fontsize=16)
    plt.axis("off")
    plt.show()

def draw_epipolar_lines(left_rect, right_rect, num_lines=20):
    """Dibuja líneas horizontales para verificar rectificación"""
    combined = np.hstack([left_rect, right_rect])
    h, w = left_rect.shape[:2]
    
    for y in range(0, h, h//num_lines):
        cv2.line(combined, (0, y), (2*w, y), (0, 255, 0), 3)
    
    return combined

def show_pair_any(idx=14, rectificar=True, num_lines=20, stereo_maps=None):
    left_path, right_path = f"data/captures/left_{idx}.jpg", f"data/captures/right_{idx}.jpg"

    if rectificar:
        L, R = rectify_stereo_pair(left_path, right_path, stereo_maps)
        titulo = "Rectificadas"
    else:
        L, R = cv2.imread(left_path), cv2.imread(right_path)
        titulo = "Originales (sin rectificar)"

    combined = draw_epipolar_lines(L, R, num_lines=num_lines)

    plt.figure(figsize=(18,9))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title(f"{titulo} - Par {idx}", fontsize=16)
    plt.axis("off")
    plt.show()

def show_pair_any_budha(idx=14, rectificar=True, num_lines=20, stereo_maps=None):
    left_path, right_path = f"data/stereo_budha_charuco/captures/left_{idx}.jpg", f"data/stereo_budha_charuco/captures/right_{idx}.jpg"

    if rectificar:
        L, R = rectify_stereo_pair(left_path, right_path, stereo_maps)
        titulo = "Rectificadas"
    else:
        L, R = cv2.imread(left_path), cv2.imread(right_path)
        titulo = "Originales (sin rectificar)"

    combined = draw_epipolar_lines(L, R, num_lines=num_lines)

    plt.figure(figsize=(18,9))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title(f"{titulo} - Par {idx}", fontsize=16)
    plt.axis("off")
    plt.show()

def compute_depth(disparity, focal_length, baseline):
    depth_map = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid_mask = disparity > 0
    depth_map[valid_mask] = (focal_length * baseline) / disparity[valid_mask]
    
    return depth_map

def compute_disparity_with_scaling(left_img, right_img, method, calibration):
    # Obtener tamaño original
    h_orig, w_orig = left_img.shape[:2]
    
    # Convertir a escala de grises si es necesario
    if len(left_img.shape) == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_img
        right_gray = right_img
    
    # Crear input pair
    input_pair = InputPair(
        left_image=left_gray,
        right_image=right_gray,
        calibration=calibration
    )
    # Calcular disparidad
    result = method.compute_disparity(input_pair)
    disparity_map = result.disparity_pixels.astype(np.float32)
    
    # Verificar si necesita escalado
    h_disp, w_disp = disparity_map.shape
    
    if (h_disp != h_orig) or (w_disp != w_orig):
        print(f"Escalando disparidad: {w_disp}x{h_disp} ==> {w_orig}x{h_orig}")
        
        # Calcular factor de escala
        scale_x = w_orig / w_disp
        
        # Redimensionar mapa
        disparity_map = cv2.resize(disparity_map, (w_orig, h_orig), 
                                   interpolation=cv2.INTER_LINEAR)
        
        disparity_map *= scale_x
    
    return disparity_map

def disparity_to_point_cloud(disparity_map, color_image, Q, max_depth=10.0):
    """
    Convierte un mapa de disparidad a nube de puntos 3D con color.
    """
    points_3d_mm = cv2.reprojectImageTo3D(disparity_map, Q)
    
    # Convertir a metros
    points_3d = points_3d_mm / 1000.0
    
    # Preparar colores (convertir BGR → RGB y normalizar)
    if len(color_image.shape) == 3:
        colors_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    else:
        colors_rgb = cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)
    
    # Reshape para tener arrays 1D
    points_flat = points_3d.reshape(-1, 3)
    colors_flat = colors_rgb.reshape(-1, 3)
    
    # Filtrar puntos inválidos

    mask_finite = np.all(np.isfinite(points_flat), axis=1)    
    depth = points_flat[:, 2]
    mask_depth = (depth > 0) & (depth < max_depth)
    mask_valid = mask_finite & mask_depth
    
    points_valid = points_flat[mask_valid]
    colors_valid = colors_flat[mask_valid] / 255.0 
    
    return points_valid, colors_valid


def detect_and_estimate_pose(image_path, board, K, dist, min_markers=4):
    """
    Detecta el ChArUco board y estima la pose de la cámara.
    """
    # Cargar imagen
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # Detectar marcadores ArUco del ChArUco
    detection = detect_charuco_markers(img, board)
    if detection is None or len(detection['ids']) < min_markers:
        return None
    
    # --- Estimar pose con homografía ---
    pose_result = estimate_camera_pose_with_homography(
        img,
        board,
        detection,
        (K, dist),
        undistort=False
    )

    if pose_result is None:
        return None

    success, rvec, tvec = pose_result
    if not success:
        return None
    
    # Calcular matrices de transformación
    R_cam_world, _ = cv2.Rodrigues(rvec)
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = R_cam_world
    T_cam_world[:3, 3] = tvec.flatten()
    T_world_cam = np.linalg.inv(T_cam_world)
    
    return {
        'success': True,
        'rvec': rvec,
        'tvec': tvec,
        'num_markers': len(detection['ids']),
        'T_world_cam': T_world_cam,
        'T_cam_world': T_cam_world
    }