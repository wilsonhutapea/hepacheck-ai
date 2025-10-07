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
    """
    Create mask for sclera using adaptive yellow-detection approach.
    
    Strategy:
    1. Find the yellowest/most saturated region in bright areas (likely sclera)
    2. Use that as reference to filter out white skin
    3. Only include pixels with similar hue/saturation to the reference
    """
    hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Step 1: Create initial candidate mask (bright regions only)
    # Focus on reasonably bright pixels that could be sclera
    _, bright_mask = cv2.threshold(v, 120, 255, cv2.THRESH_BINARY)
    
    # Step 2: Remove very dark regions (iris/pupil) early
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_mask = cv2.dilate(dark_mask, kernel, iterations=2)
    candidate_mask = cv2.bitwise_and(bright_mask, cv2.bitwise_not(dark_mask))
    
    # Step 3: Find the yellowest region in candidates
    # Look for yellow hue (15-35) with some saturation
    yellow_hue_mask = cv2.inRange(h, 10, 40)  # Yellow hue range
    yellow_sat_mask = cv2.inRange(s, 15, 255)  # Has some color (not pure white)
    yellow_candidates = cv2.bitwise_and(candidate_mask, yellow_hue_mask)
    yellow_candidates = cv2.bitwise_and(yellow_candidates, yellow_sat_mask)
    
    # Get saturation values from yellow candidate pixels
    yellow_pixels_saturation = s[yellow_candidates > 0]
    
    # Step 4: Determine if this is a jaundiced eye or normal eye
    if len(yellow_pixels_saturation) > 50:  # Need enough pixels for reliable detection
        # Found yellow region - this is likely jaundiced
        # Use the median saturation of yellow pixels as reference
        ref_saturation = np.median(yellow_pixels_saturation)
        ref_hue = np.median(h[yellow_candidates > 0])
        
        # Create mask for pixels similar to the yellow reference
        # Allow some tolerance in hue and saturation
        hue_tolerance = 15
        sat_tolerance = 30
        
        lower_hue = max(0, ref_hue - hue_tolerance)
        upper_hue = min(180, ref_hue + hue_tolerance)
        lower_sat = max(0, ref_saturation - sat_tolerance)
        upper_sat = min(255, ref_saturation + sat_tolerance)
        
        # Main sclera mask: similar hue and saturation to reference
        hue_mask = cv2.inRange(h, int(lower_hue), int(upper_hue))
        sat_mask = cv2.inRange(s, int(lower_sat), int(upper_sat))
        val_mask = cv2.inRange(v, 120, 255)  # Must be bright
        
        mask = cv2.bitwise_and(hue_mask, sat_mask)
        mask = cv2.bitwise_and(mask, val_mask)
        
        # Also include slightly less saturated pixels in the yellow range
        # This catches the full gradient of jaundiced sclera
        lower_sat_extended = max(0, ref_saturation - 50)
        sat_mask_extended = cv2.inRange(s, int(lower_sat_extended), int(upper_sat))
        mask_extended = cv2.bitwise_and(hue_mask, sat_mask_extended)
        mask_extended = cv2.bitwise_and(mask_extended, val_mask)
        
        # Combine both masks
        mask = cv2.bitwise_or(mask, mask_extended)
        
    else:
        # No significant yellow detected - assume normal white sclera
        # Use stricter criteria: very bright, very low saturation (pure white)
        white_sat_mask = cv2.inRange(s, 0, 40)  # Very low saturation
        white_val_mask = cv2.inRange(v, 150, 255)  # Very bright
        mask = cv2.bitwise_and(white_sat_mask, white_val_mask)
    
    # Step 5: Remove dark regions again (iris/pupil)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(dark_mask))
    
    # Step 6: Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Step 7: Remove highly saturated red areas (blood vessels, eyelid edges)
    lower_red1 = np.array([0, 80, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_red))
    
    # Step 8: Spatial filtering - sclera is typically in center/upper regions
    # Remove pixels from the outer edges which are more likely to be skin
    h_img, w_img = eye_region.shape[:2]
    spatial_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    
    # Create an elliptical region focusing on the eye area
    center_x, center_y = w_img // 2, h_img // 2
    axes_x, axes_y = int(w_img * 0.4), int(h_img * 0.4)
    cv2.ellipse(spatial_mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
    
    # Dilate to make it larger and more inclusive
    spatial_mask = cv2.dilate(spatial_mask, kernel, iterations=3)
    
    # Apply spatial filter
    mask = cv2.bitwise_and(mask, spatial_mask)
    
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