import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image






def extract_and_resize(image_path, padding=10, save_path=None):
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
    cv2.imwrite(save_path, timetable_roi) if save_path else None
    return timetable_roi

def ratio(image):
    # 이미지 열기
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 사이즈 확인 (width, height)
    width, height = image.size
    print(f"이미지 크기: {width}x{height}")

    # 비율 계산
    ratio = width / height
    print(f"비율 (가로/세로): {ratio:.2f}")

def process_folder(folder_path):
    # 허용된 확장자 정의
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    for filename in os.listdir(folder_path):
        # 전체 경로
        file_path = os.path.join(folder_path, filename)

        # 확장자 검사 및 파일인지 확인
        if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in valid_exts:
            ratio(extract_and_resize(file_path,save_path=os.path.join(folder_path, "processed_" + filename)))
        

# 예시 경로
process_folder("./imgs")