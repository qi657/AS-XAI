import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

font1 = {'family': 'Arial',
         'weight': 'normal',
         'size': 13,
         # 'weight': 'bold',
         }
font2 = {'family': 'Calibri',
         'weight': 'normal',
         'size': 15,
         'weight': 'bold',
         }

name = 'primrose'

data = np.genfromtxt('../test-flower.csv', delimiter=',', dtype=str, encoding='utf-8')
eye_indices = np.where(data[:, 3] == 'eye')[0]
nose_indices = np.where(data[:, 3] == 'nose')[0]
ear_indices = np.where(data[:, 3] == 'ear')[0]
leg_indices = np.where(data[:, 3] == 'leg')[0]

petal_indices = np.where(data[:, 3] == 'petal')[0]
pistil_indices = np.where(data[:, 3] == 'pistil')[0]
other_indices = np.where(data[:, 3] == 'other')[0]


eye_data = data[eye_indices, 2].astype(float)
nose_data = data[nose_indices, 2].astype(float)
ear_data = data[ear_indices, 2].astype(float)
leg_data = data[leg_indices, 2].astype(float)

pistil_data = data[pistil_indices, 2].astype(float)
petal_data = data[petal_indices, 2].astype(float)
other_data = data[other_indices, 2].astype(float)

# 计算每个类别的平均值
if name == 'cat' or name == 'dog':
    eye_values = data[eye_indices, 2].astype(float)
    eye_values = np.sum(eye_values)
    nose_values = data[nose_indices, 2].astype(float)
    nose_values = np.sum(nose_values)
    ear_values = data[ear_indices, 2].astype(float)
    ear_values = np.sum(ear_values)
    leg_values = data[leg_indices, 2].astype(float)
    leg_values = np.sum(leg_values)

    column_3 = data[:, 2].astype(float)
    total_sum = np.sum(column_3)
    eye_mean = np.round(eye_values / total_sum * 10, 1)
    nose_mean = np.round(nose_values / total_sum * 10, 1)
    ear_mean = np.round(ear_values / total_sum * 10, 1)
    leg_mean = np.round(leg_values / total_sum * 10, 1)
    # print(eye_mean, ear_mean, nose_mean, leg_mean)

elif name == 'primrose' or name == 'black eye susan':
    pistil_values = data[pistil_indices, 2].astype(float)
    pistil_values = np.sum(pistil_values)
    petal_values = data[petal_indices, 2].astype(float)
    petal_values = np.sum(petal_values)
    other_values = data[other_indices, 2].astype(float)
    other_values = np.sum(other_values)

    column_3 = data[:, 2].astype(float)
    total_sum = np.sum(column_3)
    pistil_mean = np.round(pistil_values / total_sum * 10, 1)
    petal_mean = np.round(petal_values / total_sum * 10, 1)
    other_mean = np.round(other_values / total_sum * 10, 1)

# 创建画布和子图
# fig, ax = plt.subplots()
fig,ax = plt.subplots(figsize=(7, 3), dpi=300)

# 定义自定义颜色，以及相应的透明度
# 使用LinearSegmentedColormap创建自定义渐变颜色映射
if name == 'cat' or name == 'dog':
    colors_eye = [(0, '#F9DBDB'), (0.5, '#C14C81'), (1, '#C14C81')]  # 红色渐变
    cmap_eye = LinearSegmentedColormap.from_list('custom_red', colors_eye, N=256)

    colors_ear = [(0, '#FBE7D2'), (0.5, '#C38530'), (1, '#C38530')]  # 黄色渐变
    cmap_ear = LinearSegmentedColormap.from_list('custom_yellow', colors_ear, N=256)

    colors_nose = [(0, '#D9DBF9'), (0.5, '#162D7D'), (1, '#162D7D')]  # 蓝色渐变
    cmap_nose = LinearSegmentedColormap.from_list('custom_blue', colors_nose, N=256)

    colors_leg = [(0, '#D2F0EE'), (0.5, '#135E58'), (1, '#135E58')]  # 绿色渐变
    cmap_leg = LinearSegmentedColormap.from_list('custom_green', colors_leg, N=256)

elif name == 'primrose' or name == 'black eye susan':
    colors_other = [(0, '#D9DBF9'), (0.5, '#162D7D'), (1, '#162D7D')]  # 蓝色渐变
    cmap_other = LinearSegmentedColormap.from_list('custom_blue', colors_other, N=256)

    colors_pistil = [(0, '#F9DBDB'), (0.5, '#C14C81'), (1, '#C14C81')]  # 红色渐变
    cmap_pistil = LinearSegmentedColormap.from_list('custom_red', colors_pistil, N=256)

    colors_petal = [(0, '#FBE7D2'), (0.5, '#C38530'), (1, '#C38530')]  # 黄色渐变
    cmap_petal = LinearSegmentedColormap.from_list('custom_yellow', colors_petal, N=256)


# 创建画布和子图
# fig, ax = plt.subplots()

if name =='cat':
    # 绘制散点图，并使用c参数指定颜色，根据值的大小渐变
    ax.scatter(leg_data, ['Leg'] * len(leg_data), marker='o', c=leg_data, cmap=cmap_leg, alpha=0.5,s=50)
    ax.scatter(nose_data, ['Nose'] * len(nose_data), marker='o', c=nose_data, cmap=cmap_nose, alpha=0.5,s=50)
    ax.scatter(ear_data, ['Ear'] * len(ear_data), marker='o', c=ear_data, cmap=cmap_ear, alpha=0.5,s=50)
    ax.scatter(eye_data, ['Eye'] * len(eye_data), marker='o', c=eye_data, cmap=cmap_eye, alpha=0.5,s=50)
elif name == 'dog':
    # 绘制散点图，并使用c参数指定颜色，根据值的大小渐变
    ax.scatter(leg_data, ['Leg'] * len(leg_data), marker='o', c=leg_data, cmap=cmap_leg, alpha=0.5,s=50)
    ax.scatter(ear_data, ['Ear'] * len(ear_data), marker='o', c=ear_data, cmap=cmap_ear, alpha=0.5,s=50)
    ax.scatter(nose_data, ['Nose'] * len(nose_data), marker='o', c=nose_data, cmap=cmap_nose, alpha=0.5,s=50)
    ax.scatter(eye_data, ['Eye'] * len(eye_data), marker='o', c=eye_data, cmap=cmap_eye, alpha=0.5,s=50)
elif name == 'primrose' or name == 'black eye susan':
    ax.scatter(other_data, ['other'] * len(other_data), marker='o', c=other_data, cmap=cmap_other, alpha=0.5, s=50)
    ax.scatter(pistil_data, ['pistil'] * len(pistil_data), marker='o', c=pistil_data, cmap=cmap_pistil, alpha=0.5, s=50)
    ax.scatter(petal_data, ['petal'] * len(petal_data), marker='o', c=petal_data, cmap=cmap_petal, alpha=0.5, s=50)

# ax.axvline(x=eye_mean, color='#C14C81', linestyle='--', linewidth=1.2,label='Eye rank mean')
# ax.axvline(x=ear_mean, color='#C38530', linestyle='--', linewidth=1.2,label='Ear rank mean')
# ax.axvline(x=nose_mean, color='#162D7D', linestyle='--',  linewidth=1.2,label='Nose rank mean')
# ax.axvline(x=leg_mean, color='#135E58', linestyle='--',  linewidth=1.2,label='Leg rank mean')

ax.axvline(x=pistil_mean, color='#C14C81', linestyle='--', linewidth=1.2,label='pistil rank mean')
ax.axvline(x=petal_mean, color='#C38530', linestyle='--',  linewidth=1.2,label='petal rank mean')
ax.axvline(x=other_mean.all(), color='#162D7D', linestyle='--',  linewidth=1.2,label='other rank mean')

# ax.scatter(eye_mean, 'Eye', marker='^', color='#AB4372', s=100)
# ax.scatter(ear_mean, 'Ear', marker='^', color='#AB712B', s=100)
# ax.scatter(nose_mean, 'Nose', marker='^', color='#122661', s=100)
# ax.scatter(leg_mean, 'Leg', marker='^', color='#0E4A46', s=100)

ax.set_title(f'Filter rank of {name} data',font2)
# plt.legend(prop={'weight': 'bold'})
# plt.legend(loc='upper left',fontsize=8)
# ax.legend(loc='center right', bbox_to_anchor=(1, 1.07))

# 设置纵轴标签
ax.set_ylabel('Categories', font2)

# 设置横轴标签
ax.set_xlabel('Rank', font2)

# 设置横轴刻度间隔
xticks_interval = 1
xticks_positions = np.arange(0, 8, xticks_interval)
plt.xticks(xticks_positions,fontname='Arial', fontsize=13)

# plt.xlim(0, 12)
# plt.ylim(-0.5, 3.5)  # cat & dog
plt.ylim(-0.5, 2.5)  # flowers
# plt.ylim(-0.5, 1.5)

# 设置纵轴刻度标签，并调整字体
plt.yticks(fontname='Arial', fontsize=13)

plt.savefig(f'../lime_save_2kinds/filter_rank_{name}.tiff',bbox_inches='tight', dpi=300)

# 显示图表
plt.show()
