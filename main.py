import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def extract_and_resize(image_path,target_size=(800, 1100), padding=10, save_path=None):
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

    # cv2.imwrite(save_path, timetable_roi) if save_path else None [저장기능]
    return resized

def get_contours(img,path):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 블랙 모드 감지 및 반전
    mean_val = np.mean(gray)
    is_dark = mean_val < 127
    if is_dark:
        gray = cv2.bitwise_not(gray)
        
        kernel_sharpen = np.array([[0, -0.5, 0],
                                   [-0.5, 3,-0.5],
                                   [0, -0.5, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
        gray = sharpened
        
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    if is_dark:
        # === 🔥 침식으로 덩어리 분리 ===
        kernel = np.ones((3, 3), np.uint8)  # 커널 사이즈 조절 가능
        binary = cv2.erode(binary, kernel, iterations=3)
        
    # 컨투어 추출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 20:  # 너무 작은 건 무시
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
    cv2.imshow("Detected timetable cells", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    save_image(img, save_path=path)  # 저장기능
    

    return contours

def save_image(image, save_path):
    cv2.imwrite(save_path, image) if save_path else None

        
def show_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('Detected Timetable Cells')
    plt.show()

def get_ratio(image):
    # 이미지 열기
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 사이즈 확인 (width, height)
    width, height = image.size
    print(f"이미지 크기: {width}x{height}")

    # 비율 계산
    ratio = width / height
    #print(f"비율 (가로/세로): {ratio:.2f}")
    return ratio

def process_folder(folder_path):
    # 허용된 확장자 정의
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    for filename in os.listdir(folder_path):
        # 전체 경로
        file_path = os.path.join(folder_path, filename)

        # 확장자 검사 및 파일인지 확인
        if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in valid_exts:
            roi_image = extract_and_resize(file_path,save_path=os.path.join(folder_path, "processed_" + filename))
            ratio = get_ratio(roi_image) #이미지 비율 보기.
            # show_image(roi_image)
            get_contours(roi_image,file_path)
            
            
        
# 예시 경로
process_folder("./imgs")

