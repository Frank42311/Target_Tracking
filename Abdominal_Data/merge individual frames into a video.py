import os
import cv2
import pydicom
import numpy as np

# DICOM 文件的根目录
root_dir = 'Moving'  # 或者是 'G:/Moving'，取决于你的实际情况

# 视频输出设置
fps = 24  # 帧率
video_size = (256, 256)  # 根据你的DICOM图像尺寸进行修改
video_output = 'output_video.avi'

# 初始化视频编写器
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_output, fourcc, fps, video_size)

# 遍历 DICOM 文件
for subfolder in sorted(os.listdir(root_dir)):
    dicom_folder = os.path.join(root_dir, subfolder)
    if not os.path.isdir(dicom_folder):
        continue
    for dicom_file in sorted(os.listdir(dicom_folder)):
        # 只处理 .dcm 文件
        if not dicom_file.endswith('.dcm'):
            continue

        dicom_path = os.path.join(dicom_folder, dicom_file)
        dataset = pydicom.dcmread(dicom_path)

        # 将 DICOM 图像转换成 8 位深度的灰度图像
        image = dataset.pixel_array
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = np.uint8(image)

        # 转换为3通道灰度图像
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 调整图像大小以适应视频尺寸
        image_resized = cv2.resize(image, video_size)

        # 将图像写入视频
        video_writer.write(image_resized)

# 释放视频编写器
video_writer.release()

print("视频创建完成!")
