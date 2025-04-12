import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
class imageProcessor:
    def handle_mode(self,img):
        # 라이트 모드 대응 - 색 반전, THRESHOLD 조정
        THEME = 'DARK'
        
        if img[0][0][0] > 100:
            img = 255 - img
            THEME = 'LIGHT'

        img_conv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return THEME, img_conv

    def get_roi(self,img):
        # CHECK DARK, LIGHT MODE and convert gray scale
        THEME, img_gray = self.handle_mode(img)
        
        if THEME == 'DARK':
            THRESHOLD_THEME = 30
        else:
            THRESHOLD_THEME = 10
        
        ret, otsu = cv2.threshold(img_gray, THRESHOLD_THEME, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
        # size별로 sort.
        cnts = sorted(contours, key = cv2.contourArea, reverse=True)
        
        # 가장 큰 box = ROI
        x, y, width, height = cv2.boundingRect(cnts[0])
        return img[y:y+height, x:x+width], THEME



    def time_exception(self,custom_time):
        minute = custom_time - int(custom_time)
        minute_output = [0, 0.25, 0.5, 0.75]
        
        target = [abs(x - minute) for x in minute_output]
        output = int(custom_time) + minute_output[target.index(min(target))]
        
        return output

    def get_time(self,img, box, box_height, box_width):

        start = end = box[0][0]
        for temp in box:
            if sum(temp[0]) > sum(end):
                end = temp[0]


        time_line = int(img.shape[0]/box_height) + 9
        class_start = (img.shape[0] - start[1])/box_height
        class_end = (end[1] - start[1])/box_height
        
        class_daytime = self.calculate_daytime(img.shape[1], box_width, start[0])
        class_time = self.calculate_time(time_line - class_start, class_end)

        return class_daytime, class_time

    def get_timebox(self,THEME, ROI, box_height, box_width,img):
        gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        is_dark = mean_val < 127
        if is_dark:
            gray = cv2.bitwise_not(gray)
            kernel_sharpen = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
            gray = cv2.filter2D(gray, -1, kernel_sharpen)

        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        if is_dark:
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.erode(binary, kernel, iterations=2)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        # box 크기로 필요 없는 것 제거
        results = [x for x in contours if cv2.contourArea(x) > 5000]
        

        for cnt in results:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 30 and h > 20:
                cv2.rectangle(ROI, (x, y), (x + w, y + h), (0, 0, 255), 2)

        plt.imshow(cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("ROI for Days of Week")
        plt.show()

        export_data = {}
        for box in results:
            class_daytime, class_time = self.get_time(ROI, box, box_height, box_width)
            # print(class_daytime, class_time)

            if class_daytime in export_data:
                export_data[class_daytime].append(class_time)
            else:
                export_data[class_daytime] = [class_time]

        return export_data

    def get_standard_box_size(self,ROI):
        # CHECK DARK, LIGHT MODE and convert gray scale
        THEME, img_gray = self.handle_mode(ROI)
        
        if THEME == 'DARK':
            THRESHOLD_THEME = 30
        else:
            THRESHOLD_THEME = 10
        
        ret, otsu = cv2.threshold(img_gray, THRESHOLD_THEME, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # size별로 sort.
        cnts = sorted(contours, key = cv2.contourArea, reverse=True)
        
        # 두 번째 sizse = 기본 box
        x, y, width, height = cv2.boundingRect(cnts[1])
        
        return height, width

    def calculate_daytime(self,roi_width, box_width, startpoint):
        daytime_output = ['금','목','수','화','월']

        for daytime in daytime_output:
            roi_width = roi_width - box_width
            if roi_width - startpoint < 5:
                print(daytime)
                return daytime

    def calculate_time(self,start, end):
        starttime_list = [9, 10, 10.5, 12, 13.5, 14,14.5, 15, 16.5, 18, 19.5]
        endtime_list = [10.25, 11.75, 13.25, 13.75, 14.75, 16.25, 17.75, 19.25, 20.75]
        
        class_start = [abs(x - start) for x in starttime_list]
        class_end = [abs(x - (start + end)) for x in endtime_list]
        
        output_start = starttime_list[class_start.index(min(class_start))]
        output_end = endtime_list[class_end.index(min(class_end))]


        # 커스텀으로 한 시간의 경우 예외 처리 - 30분 단위로 가장 가까운 곳으로 지정
        if min(class_start) > 0.35:
            output_start = self.time_exception(start)
            output_end = self.time_exception(start + end)

        return (f'{output_start}-{output_end}')

    def export_img(self,img):

        # get roi if needed
        ROI, THEME = self.get_roi(img)
        
        # 안에 아무것도 없으면서 가장 큰 box - 기준이 되는 box size.
        box_height, box_width = self.get_standard_box_size(ROI)
        
        # get timetable box
        output = self.get_timebox(THEME, ROI, box_height, box_width,img)
        print(output)
        return output
    
    def process_folder(self,folder_path):
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in valid_exts:
                img = cv2.imread(file_path)
                return self.export_img(img)
    
# IP =imageProcessor()
# IP.process_folder('./uploads/')