"""
Drawing and Annotation Operations

Draw shapes, text, and annotations on images.
"""

import cv2 as cv
import numpy as np
from typing import Tuple, Optional, List, Union


# =============================================================================
# Type Aliases
# =============================================================================
Color = Union[Tuple[int, int, int], Tuple[int, int, int, int], int]
Point = Tuple[int, int]
Size = Tuple[int, int]


# =============================================================================
# Basic Shapes
# =============================================================================

def draw_rectangle(
    image: np.ndarray,
    pt1: Point,
    pt2: Point,
    color: Color = (0, 255, 0),
    thickness: int = 2,
    line_type: int = cv.LINE_8,
    copy: bool = True
) -> np.ndarray:
    """
    Draw a rectangle on an image.
    
    Args:
        image: Input image
        pt1: Top-left corner point (x1, y1)
        pt2: Bottom-right corner point (x2, y2)
        color: Line color (B, G, R) or (B, G, R, A)
        thickness: Line thickness (-1 = filled)
        line_type: Type of line (cv.LINE_8, cv.LINE_4, cv.LINE_AA)
        copy: If True, create a copy before drawing
        
    Returns:
        Image with rectangle drawn
    """
    result = image.copy() if copy else image
    cv.rectangle(result, pt1, pt2, color, thickness, line_type)
    return result


def draw_circle(
    image: np.ndarray,
    center: Point,
    radius: int,
    color: Color = (0, 255, 0),
    thickness: int = 2,
    line_type: int = cv.LINE_8,
    copy: bool = True
) -> np.ndarray:
    """
    Draw a circle on an image.
    
    Args:
        image: Input image
        center: Center point (x, y)
        radius: Radius
        color: Line color
        thickness: Line thickness (-1 = filled)
        line_type: Type of line
        copy: If True, create a copy before drawing
        
    Returns:
        Image with circle drawn
    """
    result = image.copy() if copy else image
    cv.circle(result, center, radius, color, thickness, line_type)
    return result


def draw_line(
    image: np.ndarray,
    pt1: Point,
    pt2: Point,
    color: Color = (0, 255, 0),
    thickness: int = 2,
    line_type: int = cv.LINE_8,
    copy: bool = True
) -> np.ndarray:
    """
    Draw a line on an image.
    
    Args:
        image: Input image
        pt1: Start point (x1, y1)
        pt2: End point (x2, y2)
        color: Line color
        thickness: Line thickness
        line_type: Type of line
        copy: If True, create a copy before drawing
        
    Returns:
        Image with line drawn
    """
    result = image.copy() if copy else image
    cv.line(result, pt1, pt2, color, thickness, line_type)
    return result


def draw_ellipse(
    image: np.ndarray,
    center: Point,
    axes: Size,
    angle: float = 0,
    start_angle: float = 0,
    end_angle: float = 360,
    color: Color = (0, 255, 0),
    thickness: int = 2,
    line_type: int = cv.LINE_8,
    copy: bool = True
) -> np.ndarray:
    """
    Draw an ellipse on an image.
    
    Args:
        image: Input image
        center: Center point (x, y)
        axes: Axis sizes (major, minor)
        angle: Rotation angle
        start_angle: Start angle
        end_angle: End angle
        color: Line color
        thickness: Line thickness (-1 = filled)
        line_type: Type of line
        copy: If True, create a copy before drawing
        
    Returns:
        Image with ellipse drawn
    """
    result = image.copy() if copy else image
    cv.ellipse(result, center, axes, angle, start_angle, end_angle, 
               color, thickness, line_type)
    return result


def draw_polygon(
    image: np.ndarray,
    points: List[Point],
    color: Color = (0, 255, 0),
    thickness: int = 2,
    is_closed: bool = True,
    line_type: int = cv.LINE_8,
    copy: bool = True
) -> np.ndarray:
    """
    วาดรูปหลายเหลี่ยมบนภาพ
    
    Args:
        image: Input image
        points: รายการจุดของรูปหลายเหลี่ยม [(x1,y1), (x2,y2), ...]
        color: สีของเส้น
        thickness: ความหนาของเส้น
        is_closed: ถ้า True จะปิดรูป
        line_type: ชนิดของเส้น
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่มีรูปหลายเหลี่ยม
    """
    result = image.copy() if copy else image
    pts = np.array(points, dtype=np.int32)
    cv.polylines(result, [pts], is_closed, color, thickness, line_type)
    return result


def draw_filled_polygon(
    image: np.ndarray,
    points: List[Point],
    color: Color = (0, 255, 0),
    line_type: int = cv.LINE_8,
    copy: bool = True
) -> np.ndarray:
    """
    วาดรูปหลายเหลี่ยมแบบ filled บนภาพ
    
    Args:
        image: Input image
        points: รายการจุดของรูปหลายเหลี่ยม [(x1,y1), (x2,y2), ...]
        color: สีของรูป
        line_type: ชนิดของเส้น
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่มีรูปหลายเหลี่ยม filled
    """
    result = image.copy() if copy else image
    pts = np.array(points, dtype=np.int32)
    cv.fillPoly(result, [pts], color, line_type)
    return result


def draw_polylines(
    image: np.ndarray,
    points_list: List[List[Point]],
    color: Color = (0, 255, 0),
    thickness: int = 2,
    is_closed: bool = False,
    line_type: int = cv.LINE_8,
    copy: bool = True
) -> np.ndarray:
    """
    วาดเส้นหลายเส้นบนภาพ
    
    Args:
        image: Input image
        points_list: รายการของรายการจุด [[(x1,y1), (x2,y2)], ...]
        color: สีของเส้น
        thickness: ความหนาของเส้น
        is_closed: ถ้า True จะปิดแต่ละเส้น
        line_type: ชนิดของเส้น
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่มีเส้นหลายเส้น
    """
    result = image.copy() if copy else image
    pts_list = [np.array(pts, dtype=np.int32) for pts in points_list]
    cv.polylines(result, pts_list, is_closed, color, thickness, line_type)
    return result


def draw_arrow(
    image: np.ndarray,
    pt1: Point,
    pt2: Point,
    color: Color = (0, 255, 0),
    thickness: int = 2,
    line_type: int = cv.LINE_8,
    tip_length: float = 0.1,
    copy: bool = True
) -> np.ndarray:
    """
    วาดลูกศรบนภาพ
    
    Args:
        image: Input image
        pt1: จุดเริ่มต้น (x1, y1)
        pt2: จุดสิ้นสุด (ปลายลูกศร) (x2, y2)
        color: สีของลูกศร
        thickness: ความหนาของเส้น
        line_type: ชนิดของเส้น
        tip_length: ความยาวหัวลูกศร (เป็นสัดส่วนของความยาวลูกศร)
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่มีลูกศร
    """
    result = image.copy() if copy else image
    cv.arrowedLine(result, pt1, pt2, color, thickness, line_type, 0, tip_length)
    return result


# =============================================================================
# Text Operations
# =============================================================================

def draw_text(
    image: np.ndarray,
    text: str,
    position: Point,
    font_face: int = cv.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
    color: Color = (0, 255, 0),
    thickness: int = 2,
    line_type: int = cv.LINE_AA,
    bottom_left_origin: bool = False,
    copy: bool = True
) -> np.ndarray:
    """
    เขียนข้อความบนภาพ
    
    Args:
        image: Input image
        text: ข้อความที่ต้องการเขียน
        position: ตำแหน่ง (x, y) - มุมซ้ายล่างของข้อความ
        font_face: ฟอนต์ (cv.FONT_HERSHEY_*)
        font_scale: ขนาดฟอนต์
        color: สีของข้อความ
        thickness: ความหนาของตัวอักษร
        line_type: ชนิดของเส้น
        bottom_left_origin: ถ้า True จะใช้มุมซ้ายล่างเป็น origin
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่มีข้อความ
    """
    result = image.copy() if copy else image
    cv.putText(result, text, position, font_face, font_scale, 
               color, thickness, line_type, bottom_left_origin)
    return result


def draw_text_with_background(
    image: np.ndarray,
    text: str,
    position: Point,
    font_face: int = cv.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
    text_color: Color = (255, 255, 255),
    bg_color: Color = (0, 0, 0),
    thickness: int = 2,
    padding: int = 5,
    copy: bool = True
) -> np.ndarray:
    """
    เขียนข้อความพร้อมพื้นหลังบนภาพ
    
    Args:
        image: Input image
        text: ข้อความที่ต้องการเขียน
        position: ตำแหน่ง (x, y) - มุมซ้ายบนของ background
        font_face: ฟอนต์
        font_scale: ขนาดฟอนต์
        text_color: สีของข้อความ
        bg_color: สีพื้นหลัง
        thickness: ความหนาของตัวอักษร
        padding: ระยะห่างจากขอบ
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่มีข้อความพร้อมพื้นหลัง
    """
    result = image.copy() if copy else image
    
    # Calculate text size
    (text_w, text_h), baseline = cv.getTextSize(
        text, font_face, font_scale, thickness
    )
    
    # Draw background
    x, y = position
    cv.rectangle(
        result,
        (x, y),
        (x + text_w + 2 * padding, y + text_h + baseline + 2 * padding),
        bg_color,
        -1
    )
    
    # Draw text
    cv.putText(
        result, text,
        (x + padding, y + text_h + padding),
        font_face, font_scale, text_color, thickness, cv.LINE_AA
    )
    
    return result


def get_text_size(
    text: str,
    font_face: int = cv.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
    thickness: int = 2
) -> Tuple[Size, int]:
    """
    คำนวณขนาดข้อความ
    
    Args:
        text: ข้อความ
        font_face: ฟอนต์
        font_scale: ขนาดฟอนต์
        thickness: ความหนาของตัวอักษร
        
    Returns:
        ((width, height), baseline)
    """
    return cv.getTextSize(text, font_face, font_scale, thickness)


def draw_multiline_text(
    image: np.ndarray,
    text: str,
    position: Point,
    font_face: int = cv.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
    color: Color = (0, 255, 0),
    thickness: int = 2,
    line_spacing: float = 1.5,
    copy: bool = True
) -> np.ndarray:
    """
    เขียนข้อความหลายบรรทัดบนภาพ
    
    Args:
        image: Input image
        text: ข้อความ (ใช้ \\n เพื่อขึ้นบรรทัดใหม่)
        position: ตำแหน่งเริ่มต้น (x, y)
        font_face: ฟอนต์
        font_scale: ขนาดฟอนต์
        color: สีของข้อความ
        thickness: ความหนาของตัวอักษร
        line_spacing: ระยะห่างระหว่างบรรทัด (เท่าของความสูงข้อความ)
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่มีข้อความหลายบรรทัด
    """
    result = image.copy() if copy else image
    
    lines = text.split('\n')
    x, y = position
    
    # Calculate line height
    (_, text_h), _ = cv.getTextSize('Tg', font_face, font_scale, thickness)
    line_height = int(text_h * line_spacing)
    
    for i, line in enumerate(lines):
        y_pos = y + (i + 1) * line_height
        cv.putText(result, line, (x, y_pos), font_face, font_scale, 
                   color, thickness, cv.LINE_AA)
    
    return result


# =============================================================================
# Annotation Utilities
# =============================================================================

def draw_bounding_box(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: Optional[str] = None,
    color: Color = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6,
    copy: bool = True
) -> np.ndarray:
    """
    วาด bounding box พร้อม label บนภาพ
    
    Args:
        image: Input image
        bbox: (x, y, width, height) หรือ (x1, y1, x2, y2)
        label: ข้อความ label (optional)
        color: สีของกรอบ
        thickness: ความหนาของเส้น
        font_scale: ขนาดฟอนต์ของ label
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่มี bounding box
    """
    result = image.copy() if copy else image
    
    x, y, w_or_x2, h_or_y2 = bbox
    
    # Check if format is (x, y, w, h) or (x1, y1, x2, y2)
    if w_or_x2 > x and h_or_y2 > y and w_or_x2 < image.shape[1] * 0.5:
        # Likely (x, y, w, h) format
        x1, y1, x2, y2 = x, y, x + w_or_x2, y + h_or_y2
    else:
        # Likely (x1, y1, x2, y2) format
        x1, y1, x2, y2 = x, y, w_or_x2, h_or_y2
    
    # Draw bounding box
    cv.rectangle(result, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label:
        (text_w, text_h), baseline = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Draw label background
        cv.rectangle(
            result,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w + 5, y1),
            color,
            -1
        )
        
        # Draw label text
        cv.putText(
            result, label,
            (x1 + 2, y1 - baseline - 2),
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness - 1 if thickness > 1 else 1,
            cv.LINE_AA
        )
    
    return result


def draw_bounding_boxes(
    image: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    colors: Optional[List[Color]] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
    copy: bool = True
) -> np.ndarray:
    """
    วาด bounding boxes หลายอันบนภาพ
    
    Args:
        image: Input image
        bboxes: รายการ bounding boxes [(x, y, w, h), ...]
        labels: รายการ labels (optional)
        colors: รายการสี (optional, ถ้าไม่ระบุจะใช้สีสุ่ม)
        thickness: ความหนาของเส้น
        font_scale: ขนาดฟอนต์ของ label
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่มี bounding boxes
    """
    result = image.copy() if copy else image
    
    if colors is None:
        # Generate random colors for each bbox
        np.random.seed(42)
        colors = [
            tuple(int(c) for c in np.random.randint(0, 255, 3))
            for _ in range(len(bboxes))
        ]
    
    if labels is None:
        labels = [None] * len(bboxes)
    
    for bbox, label, color in zip(bboxes, labels, colors):
        result = draw_bounding_box(
            result, bbox, label, color, thickness, font_scale, copy=False
        )
    
    return result


def draw_contours(
    image: np.ndarray,
    contours: List[np.ndarray],
    color: Color = (0, 255, 0),
    thickness: int = 2,
    contour_idx: int = -1,
    copy: bool = True
) -> np.ndarray:
    """
    วาด contours บนภาพ
    
    Args:
        image: Input image
        contours: รายการ contours จาก cv.findContours
        color: สีของ contours
        thickness: ความหนาของเส้น (-1 = filled)
        contour_idx: index ของ contour ที่ต้องการวาด (-1 = ทั้งหมด)
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่มี contours
    """
    result = image.copy() if copy else image
    cv.drawContours(result, contours, contour_idx, color, thickness)
    return result


def draw_keypoints(
    image: np.ndarray,
    keypoints: List[Point],
    color: Color = (0, 255, 0),
    radius: int = 5,
    thickness: int = -1,
    copy: bool = True
) -> np.ndarray:
    """
    วาด keypoints บนภาพ
    
    Args:
        image: Input image
        keypoints: รายการจุด [(x, y), ...]
        color: สีของจุด
        radius: รัศมีของจุด
        thickness: ความหนาของวงกลม (-1 = filled)
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่มี keypoints
    """
    result = image.copy() if copy else image
    
    for pt in keypoints:
        cv.circle(result, pt, radius, color, thickness)
    
    return result


def draw_grid(
    image: np.ndarray,
    grid_size: Size = (10, 10),
    color: Color = (128, 128, 128),
    thickness: int = 1,
    copy: bool = True
) -> np.ndarray:
    """
    วาดตารางบนภาพ
    
    Args:
        image: Input image
        grid_size: จำนวนช่อง (rows, cols)
        color: สีของเส้น
        thickness: ความหนาของเส้น
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่มีตาราง
    """
    result = image.copy() if copy else image
    h, w = result.shape[:2]
    rows, cols = grid_size
    
    # Draw vertical lines
    for i in range(1, cols):
        x = int(w * i / cols)
        cv.line(result, (x, 0), (x, h), color, thickness)
    
    # Draw horizontal lines
    for i in range(1, rows):
        y = int(h * i / rows)
        cv.line(result, (0, y), (w, y), color, thickness)
    
    return result


def draw_crosshair(
    image: np.ndarray,
    center: Optional[Point] = None,
    size: int = 20,
    color: Color = (0, 255, 0),
    thickness: int = 2,
    copy: bool = True
) -> np.ndarray:
    """
    วาด crosshair บนภาพ
    
    Args:
        image: Input image
        center: จุดศูนย์กลาง (x, y), ถ้า None จะใช้กลางภาพ
        size: ความยาวของ crosshair
        color: สีของ crosshair
        thickness: ความหนาของเส้น
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่มี crosshair
    """
    result = image.copy() if copy else image
    h, w = result.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    cx, cy = center
    half = size // 2
    
    # Horizontal line
    cv.line(result, (cx - half, cy), (cx + half, cy), color, thickness)
    # Vertical line
    cv.line(result, (cx, cy - half), (cx, cy + half), color, thickness)
    
    return result


def draw_marker(
    image: np.ndarray,
    position: Point,
    marker_type: int = cv.MARKER_CROSS,
    color: Color = (0, 255, 0),
    marker_size: int = 20,
    thickness: int = 2,
    copy: bool = True
) -> np.ndarray:
    """
    วาด marker บนภาพ
    
    Args:
        image: Input image
        position: ตำแหน่ง (x, y)
        marker_type: ชนิดของ marker (cv.MARKER_*)
        color: สี
        marker_size: ขนาด
        thickness: ความหนาของเส้น
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่มี marker
        
    Marker Types:
        - cv.MARKER_CROSS: กากบาท
        - cv.MARKER_TILTED_CROSS: กากบาทเอียง
        - cv.MARKER_STAR: ดาว
        - cv.MARKER_DIAMOND: เพชร
        - cv.MARKER_SQUARE: สี่เหลี่ยม
        - cv.MARKER_TRIANGLE_UP: สามเหลี่ยมขึ้น
        - cv.MARKER_TRIANGLE_DOWN: สามเหลี่ยมลง
    """
    result = image.copy() if copy else image
    cv.drawMarker(result, position, color, marker_type, marker_size, thickness)
    return result


# =============================================================================
# Overlay Operations
# =============================================================================

def overlay_image(
    background: np.ndarray,
    overlay: np.ndarray,
    position: Point = (0, 0),
    alpha: float = 1.0,
    copy: bool = True
) -> np.ndarray:
    """
    วาง overlay image บน background
    
    Args:
        background: Background image
        overlay: Overlay image (ต้องเล็กกว่าหรือเท่ากับ background)
        position: ตำแหน่งมุมซ้ายบนของ overlay (x, y)
        alpha: ความโปร่งใสของ overlay (0-1)
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่รวมกัน
    """
    result = background.copy() if copy else background
    
    x, y = position
    h, w = overlay.shape[:2]
    bg_h, bg_w = result.shape[:2]
    
    # Resize if overlay is too large
    if x + w > bg_w:
        w = bg_w - x
    if y + h > bg_h:
        h = bg_h - y
    
    if w <= 0 or h <= 0:
        return result
    
    overlay_crop = overlay[:h, :w]
    roi = result[y:y+h, x:x+w]
    
    if alpha < 1.0:
        cv.addWeighted(overlay_crop, alpha, roi, 1 - alpha, 0, roi)
    else:
        result[y:y+h, x:x+w] = overlay_crop
    
    return result


def add_alpha_channel(
    image: np.ndarray,
    alpha: int = 255
) -> np.ndarray:
    """
    เพิ่ม alpha channel ให้กับภาพ
    
    Args:
        image: Input image (BGR หรือ grayscale)
        alpha: ค่า alpha เริ่มต้น (0-255)
        
    Returns:
        ภาพที่มี alpha channel (BGRA)
    """
    if len(image.shape) == 2:
        # Grayscale -> BGRA
        bgr = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        return cv.cvtColor(bgr, cv.COLOR_BGR2BGRA)
    elif image.shape[2] == 3:
        # BGR -> BGRA
        bgra = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha
        return bgra
    else:
        # Already has alpha
        return image.copy()


def create_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Color = (0, 255, 0),
    alpha: float = 0.5,
    copy: bool = True
) -> np.ndarray:
    """
    สร้าง overlay จาก mask บนภาพ
    
    Args:
        image: Input image (BGR)
        mask: Binary mask
        color: สีของ overlay
        alpha: ความโปร่งใส (0-1)
        copy: ถ้า True จะสร้างสำเนาก่อนวาด
        
    Returns:
        ภาพที่มี mask overlay
    """
    result = image.copy() if copy else image
    
    # Create color overlay
    overlay = np.zeros_like(result)
    overlay[mask > 0] = color
    
    # Merge overlay with original image
    cv.addWeighted(overlay, alpha, result, 1.0, 0, result)
    
    return result
