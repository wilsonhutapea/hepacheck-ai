import cv2
import numpy as np


def analyze_sclera_color(image, mask):
    """
    Analisis warna sclera dari image dan mask
    
    Args:
        image: numpy array (BGR format)
        mask: binary mask (sclera region)
    
    Returns:
        dict with color information or None
    """
    if mask is None or np.sum(mask) == 0:
        return None
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sclera_pixels = rgb_image[mask > 0]
    
    if len(sclera_pixels) == 0:
        return None
    
    avg_rgb = sclera_pixels.mean(axis=0)
    
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    sclera_hsv = hsv_image[mask > 0]
    avg_hsv = sclera_hsv.mean(axis=0)
    
    return {
        'rgb': avg_rgb,
        'hsv': avg_hsv,
        'pixels': sclera_pixels,
        'pixel_count': len(sclera_pixels)
    }


def detect_jaundice(color_data):
    """
    Deteksi jaundice dari data warna sclera
    
    Args:
        color_data: dict from analyze_sclera_color()
    
    Returns:
        tuple (is_jaundice: bool, confidence: float, yellow_index: float)
    """
    if color_data is None:
        return False, 0, 0
    
    r, g, b = color_data['rgb']
    h, s, v = color_data['hsv']
    
    # Calculate yellow index
    yellow_index = (r + g) / (2 * b + 1)
    
    # Jaundice detection criteria
    is_yellowish = (20 < h < 40) and (s > 15)
    has_yellow_tint = (r > b * 1.1) and (g > b * 1.1)
    high_yellow_index = yellow_index > 1.15
    
    # Calculate confidence score
    confidence = 0
    if is_yellowish:
        confidence += 35
    if has_yellow_tint:
        confidence += 35
    if high_yellow_index:
        confidence += 30
    
    is_jaundice = confidence > 50
    
    return is_jaundice, confidence, yellow_index


def get_risk_level(eye_jaundice_detected, symptom_count, avg_confidence):
    """
    Determine risk level based on eye detection and symptoms
    
    Args:
        eye_jaundice_detected: bool
        symptom_count: int (number of positive symptoms)
        avg_confidence: float (average confidence from both eyes)
    
    Returns:
        tuple (risk_level: str, message: str, recommendation: str)
    """
    if eye_jaundice_detected and symptom_count >= 3:
        return (
            "TINGGI",
            f"⚠️ **RISIKO TINGGI JAUNDICE**\n\n"
            f"- Terdeteksi tanda jaundice pada mata ({avg_confidence:.1f}% confidence)\n"
            f"- {symptom_count} dari 11 gejala terkonfirmasi",
            "**Rekomendasi:** Segera konsultasi dengan dokter untuk pemeriksaan lebih lanjut dan tes fungsi hati."
        )
    elif eye_jaundice_detected or symptom_count >= 5:
        return (
            "SEDANG",
            f"⚠️ **RISIKO SEDANG**\n\n"
            f"- Beberapa indikasi jaundice terdeteksi\n"
            f"- Gejala klinis: {symptom_count}/11",
            "**Rekomendasi:** Disarankan untuk pemeriksaan medis dalam waktu dekat."
        )
    else:
        return (
            "RENDAH",
            "✅ **RISIKO RENDAH**\n\n"
            "Tidak terdeteksi tanda-tanda jaundice yang signifikan.",
            "**Rekomendasi:** Tetap jaga kesehatan hati dengan pola hidup sehat."
        )


def create_visualization_data(detection_result, color_analysis):
    """
    Prepare data for visualization
    
    Args:
        detection_result: dict from eye_detection
        color_analysis: dict with left and right color data
    
    Returns:
        dict with visualization info
    """
    viz_data = {
        'method': detection_result.get('method', 'unknown'),
        'left_eye_image': detection_result.get('left_eye'),
        'right_eye_image': detection_result.get('right_eye'),
        'left_bbox': detection_result.get('left_bbox'),
        'right_bbox': detection_result.get('right_bbox')
    }
    
    # Add masked images
    if detection_result.get('left_eye') is not None and detection_result.get('left_mask') is not None:
        left_masked = detection_result['left_eye'].copy()
        left_masked[detection_result['left_mask'] == 0] = 0
        viz_data['left_masked'] = left_masked
    
    if detection_result.get('right_eye') is not None and detection_result.get('right_mask') is not None:
        right_masked = detection_result['right_eye'].copy()
        right_masked[detection_result['right_mask'] == 0] = 0
        viz_data['right_masked'] = right_masked
    
    return viz_data