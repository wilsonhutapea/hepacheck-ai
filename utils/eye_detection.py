import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Eye Landmark Indices
LEFT_EYE_INDICES = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_INDICES = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]


def detect_with_mediapipe(image):
    """
    Deteksi mata menggunakan MediaPipe Face Mesh
    
    Args:
        image: numpy array (BGR format)
    
    Returns:
        dict with eye regions and masks, or None if detection fails
    """
    mp_face_mesh = mp.solutions.face_mesh
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        h, w = image.shape[:2]
        
        # Extract both eyes
        left_eye, left_mask, left_bbox = extract_eye_region_mediapipe(
            image, landmarks, LEFT_EYE_INDICES, LEFT_IRIS_INDICES
        )
        right_eye, right_mask, right_bbox = extract_eye_region_mediapipe(
            image, landmarks, RIGHT_EYE_INDICES, RIGHT_IRIS_INDICES
        )
        
        return {
            'method': 'mediapipe',
            'left_eye': left_eye,
            'left_mask': left_mask,
            'left_bbox': left_bbox,
            'right_eye': right_eye,
            'right_mask': right_mask,
            'right_bbox': right_bbox,
            'success': True
        }


def extract_eye_region_mediapipe(image, landmarks, eye_indices, iris_indices):
    """Extract eye region using MediaPipe landmarks"""
    h, w = image.shape[:2]
    
    eye_points = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        eye_points.append([x, y])
    
    eye_points = np.array(eye_points, dtype=np.int32)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [eye_points], 255)
    
    # Remove iris/pupil area
    iris_points = []
    for idx in iris_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        iris_points.append([x, y])
    
    if len(iris_points) > 0:
        iris_points = np.array(iris_points, dtype=np.int32)
        center = iris_points.mean(axis=0).astype(int)
        radius = int(np.max(np.linalg.norm(iris_points - center, axis=1)) * 1.5)
        cv2.circle(mask, tuple(center), radius, 0, -1)
    
    x, y, w_eye, h_eye = cv2.boundingRect(eye_points)
    
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w_eye = min(w - x, w_eye + 2 * padding)
    h_eye = min(h - y, h_eye + 2 * padding)
    
    eye_region = image[y:y+h_eye, x:x+w_eye].copy()
    mask_region = mask[y:y+h_eye, x:x+w_eye]
    
    return eye_region, mask_region, (x, y, w_eye, h_eye)


def detect_with_haar_cascade(image):
    """Detect eyes using Haar Cascade (for zoomed faces)"""
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(eyes) == 0:
        return None
    
    # Sort eyes by x-coordinate
    eyes = sorted(eyes, key=lambda e: e[0])
    
    def extract_eye_from_bbox(bbox):
        x, y, w, h = bbox
        eye_region = image[y:y+h, x:x+w].copy()
        mask = create_sclera_mask(eye_region)
        return eye_region, mask, (x, y, w, h)
    
    left_eye, left_mask, left_bbox = extract_eye_from_bbox(eyes[0])
    
    if len(eyes) >= 2:
        right_eye, right_mask, right_bbox = extract_eye_from_bbox(eyes[1])
    else:
        right_eye, right_mask, right_bbox = None, None, None
    
    return {
        'method': 'haar_cascade',
        'left_eye': left_eye,
        'left_mask': left_mask,
        'left_bbox': left_bbox,
        'right_eye': right_eye,
        'right_mask': right_mask,
        'right_bbox': right_bbox,
        'success': True
    }


def detect_with_segmentation(image):
    """Detect eyes using color segmentation (for extreme close-ups)"""
    h, w = image.shape[:2]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    
    if circles is None:
        return None
    
    eye_region = image.copy()
    mask = create_sclera_mask(eye_region)
    
    return {
        'method': 'segmentation',
        'left_eye': eye_region,
        'left_mask': mask,
        'left_bbox': (0, 0, w, h),
        'right_eye': None,
        'right_mask': None,
        'right_bbox': None,
        'success': True
    }


def create_sclera_mask(eye_region):
    """Create mask for sclera (white part) using color thresholding"""
    hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
    
    # White/light colors (sclera)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 60, 255])
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Remove dark regions (iris/pupil)
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    dark_mask = cv2.dilate(dark_mask, kernel, iterations=2)
    
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(dark_mask))
    
    return mask


def multi_stage_detection(image):
    """
    Multi-stage detection: tries different methods until one succeeds
    
    Args:
        image: numpy array (BGR format)
    
    Returns:
        dict with detection results or None
    """
    # Method 1: MediaPipe
    result = detect_with_mediapipe(image)
    if result is not None:
        return result
    
    # Method 2: Haar Cascade
    result = detect_with_haar_cascade(image)
    if result is not None:
        return result
    
    # Method 3: Segmentation
    result = detect_with_segmentation(image)
    return result