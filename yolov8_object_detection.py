import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 后续代码（加载模型、摄像头读取等）保持不变
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
#...（其余代码）

from ultralytics import YOLO
import cv2

# 加载预训练的 YOLOv8 模型
model = YOLO('yolov8n.pt')

# 打开默认摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 读取一帧图像
    success, frame = cap.read()

    if success:
        # 使用模型进行目标识别
        results = model(frame)

        # 处理识别结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                # 获取类别索引
                cls = int(box.cls[0])
                # 获取类别名称
                class_name = model.names[cls]
                # 获取置信度
                confidence = float(box.conf[0])

                # 在图像上绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 在图像上添加类别名称和置信度信息
                cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 显示处理后的帧
        cv2.imshow('YOLOv8 Object Detection', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # 如果无法读取帧，退出循环
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()