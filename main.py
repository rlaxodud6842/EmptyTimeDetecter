import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def extract_timetable_box(image_path, padding=-1):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 이미지 못 불러옴: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_box = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > max_area and w > 200 and h > 200:
            max_area = area
            best_box = (x, y, w, h)

    if best_box is None:
        print(f"❌ 시간표 박스를 못 찾았어: {image_path}")
        return None

    x, y, w, h = best_box
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = w + padding * 2
    h = h + padding * 2

    timetable_roi = img[y:y+h, x:x+w]
    resized = cv2.resize(timetable_roi,(1200,1200))
    return resized

def draw_day_of_week(image,countours):
    image_height, image_width = image.shape[:2]

    day_of_week_ratio = {
        'Mon': (0.055, 0.0, 0.18, 0.05),
        'Tue': (0.245, 0.0, 0.18, 0.05),
        'Wed': (0.435, 0.0, 0.18, 0.05),
        'Thu': (0.625, 0.0, 0.18, 0.05),
        'Fri': (0.815, 0.0, 0.18, 0.05),
    }
    y_rate = 0.001

    time_of_day_ratio = {
    '9:00': (0.052, y_rate),    # 9시 시작 y 비율, 높이 비율
    '10:00': (0.17, y_rate),   # 10시 시작 y 비율, 높이 비율
    '11:00': (0.29, y_rate),   # 11시 시작 y 비율, 높이 비율
    '12:00': (0.41, y_rate),   # 12시 시작 y 비율, 높이 비율
    '13:00': (0.534, y_rate),   # 13시 시작 y 비율, 높이 비율
    '14:00': (0.55, y_rate),   # 14시 시작 y 비율, 높이 비율
    '15:00': (0.65, y_rate),   # 15시 시작 y 비율, 높이 비율
    '16:00': (0.75, y_rate),   # 16시 시작 y 비율, 높이 비율
    '17:00': (0.85, y_rate),   # 17시 시작 y 비율, 높이 비율
    }

    for key, (x_r, y_r, w_r, h_r) in day_of_week_ratio.items():
        x = int(x_r * image_width)
        y = int(y_r * image_height)
        w = int(w_r * image_width)
        h = int(h_r * image_height)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, key, (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
    for cnt in countours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 20:
            # 🗓️ 요일 판단
            day = "Unknown"
            for d, (x_ratio, _, w_ratio, _) in day_of_week_ratio.items():
                start_x = int(x_ratio * image_width)
                end_x = int((x_ratio + w_ratio) * image_width)
                if start_x <= x <= end_x or start_x <= x + w // 2 <= end_x:
                    day = d
                    break

            # 박스 그리기
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, day, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)
            
        # 각 시간대별로 사각형 그리기
    for time, (y_r, h_r) in time_of_day_ratio.items():
        # 비율을 실제 이미지 크기에 맞춰 계산
        y = int(y_r * image_height)  # 해당 시간대의 시작 y 좌표
        h = int(h_r * image_height)  # 각 시간대의 높이 (예: 1시간 = 0.08 비율)

        # 사각형 그리기 (시간 영역)
        cv2.rectangle(image, (0, y), (image_width, y + h), (255, 0, 0), 2)  # 파란색 사각형, 두께 2
        
        # 시간 텍스트 추가 (사각형 왼쪽 위에 표시)
        cv2.putText(image, time, (10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("ROI for Days of Week")
    plt.show()

def get_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    is_dark = mean_val < 127
    if is_dark:
        gray = cv2.bitwise_not(gray)
        kernel_sharpen = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        gray = cv2.filter2D(gray, -1, kernel_sharpen)

    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    if is_dark:
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=3)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 20:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return contours

def process_folder(folder_path):
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in valid_exts:
            roi_image = extract_timetable_box(file_path)
            if roi_image is not None:
                resized = cv2.resize(roi_image, (1000, 1250))  # 📏 기준 리사이즈
                contours = get_contours(resized)
                draw_day_of_week(resized,contours)

process_folder("./imgs")  # 사용 시 경로 수정