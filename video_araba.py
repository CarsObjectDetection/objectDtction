from ultralytics import YOLO
import cv2
import time
import numpy as np

# HSV renk aralıkları 
COLOR_RANGES = {
    'red': [(0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255)],  # Kırmızı
    'green': [(36, 25, 25), (86, 255, 255)],  # Yeşil
    'blue': [(90, 50, 50), (130, 255, 255)],  # Mavi
    'white': [(0, 0, 200), (180, 40, 255)],  # Beyaz
    'black': [(0, 0, 0), (180, 255, 50)],  # Siyah
    'gray': [(0, 0, 50), (180, 20, 200)],  # Gri (düşük doygunluk, orta parlaklık)
}

def get_dominant_color_hsv(image):
    # HSV'ye çevir
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Renk tespiti
    max_area = 0
    dominant_color = 'unknown'

    for color_name, hsv_range in COLOR_RANGES.items():
        if len(hsv_range) == 2:  # Tek renk
            lower, upper = hsv_range
            mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        else:  # Kırmızı için ayrı iki aralık var
            lower1, upper1, lower2, upper2 = hsv_range
            mask1 = cv2.inRange(hsv_image, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv_image, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask1, mask2)

        # Maskeyi kullanarak alan büyüklüğünü hesapla
        color_area = cv2.countNonZero(mask)
        if color_area > max_area:
            max_area = color_area
            dominant_color = color_name

    return dominant_color

def adjust_brightness_contrast(image, alpha=1.05, beta=10):
    """
    Görüntüde parlaklık ve kontrast ayarlarını yapar (daha az agresif)
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# YOLO modeli
model = YOLO("best.pt")
cap = cv2.VideoCapture("Drone_Video.mp4")

fps = int(cap.get(cv2.CAP_PROP_FPS))
duration_to_process = 60
total_frames = int(duration_to_process * fps)

frame_nmr = 0

while True:
    ret, frame = cap.read()
    if not ret or frame_nmr >= total_frames:
        break

    # Görüntüde sadece küçük ayarlamalar yapalım
    frame = adjust_brightness_contrast(frame)

    results = model(frame)[0]

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue

        # Kutu yüksekliğini göz önünde bulundur, sadece alt kısmı dikkate alacağız
        body_roi = frame[y2 - (y2 - y1) // 3 : y2, x1:x2]  # Alt kısmı daha iyi seçmek için

        if body_roi.size == 0:
            continue

        color_name = get_dominant_color_hsv(body_roi)

        # Kutu çiz ve renk ismini yaz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, color_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Vehicle Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
