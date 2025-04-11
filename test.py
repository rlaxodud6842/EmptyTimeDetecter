import matplotlib.pyplot as plt
import numpy as np

# 시간대 (시간단위: 24시 기준)
hours = np.array([i for i in range(24)])

# 공강 시간대
break_times = {
    '화': [['16', '16.75'], ['14', '15.75'], ['12', '13.25'], ['10', '11.75']],
    '월': [['15', '16.25'], ['12', '13.25'], ['10.5', '11.75']],
    '수': [['13.5', '14.75'], ['12', '13.25'], ['9', '10.25']],
    '목': [['11', '12.75']],
    '금': [['10', '10.75']]
}

# 요일을 위한 색상 설정
colors = {'화': 'red', '월': 'blue', '수': 'green', '목': 'purple', '금': 'orange'}

# 그래프 설정
fig, ax = plt.subplots(figsize=(10, 6))

# 각 요일의 공강 시간대 시각적으로 표시
for day, times in break_times.items():
    for start, end in times:
        start_hour = float(start)
        end_hour = float(end)
        
        # 색상으로 공강 시간을 표시
        ax.barh(day, end_hour - start_hour, left=start_hour, color=colors[day], edgecolor='black', height=0.8)

# x축과 y축 설정
ax.set_xlim(9, 18)  # 오전 9시부터 오후 6시까지
ax.set_xticks(np.arange(9, 19, 1))  # 1시간 단위로 x축을 표시
ax.set_xlabel('시간 (24시 기준)')
ax.set_ylabel('요일')

# 제목 추가
ax.set_title('공강 시간대')

# 요일에 대한 레전드 추가
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors.values()]
labels = [day for day in colors.keys()]
ax.legend(handles, labels, title='요일')

plt.tight_layout()
plt.show()
