time_blocks = {}
start_hour = 9
interval = 15
total_blocks = ((19 - start_hour) * 60) // interval  # 36
block_height_ratio = 1 / total_blocks  # 각 블록의 높이 비율

for i in range(total_blocks):
    hour = start_hour + (i * interval) // 60
    minute = (i * interval) % 60
    end_minute = (minute + interval) % 60
    end_hour = hour + ((minute + interval) // 60)
    label = f"{hour:02d}:{minute:02d}~{end_hour:02d}:{end_minute:02d}"
    
    y_ratio = i * block_height_ratio
    time_blocks[label] = (0.0, y_ratio, 1.0, block_height_ratio)  # (x, y, w, h) 비율

print(time_blocks['09:00~09:15'])