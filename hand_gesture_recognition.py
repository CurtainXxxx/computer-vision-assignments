import cv2
import mediapipe as mp

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def get_distance(pt1, pt2):
    """计算两点之间的欧氏距离"""
    return ((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)**0.5

def is_fist(hand_landmarks):
    """判断握拳手势"""
    tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    return all(get_distance(hand_landmarks.landmark[tip], wrist) < 0.1 for tip in tips)

def is_victory(hand_landmarks):
    """判断剪刀手（胜利手势）"""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # 食指和中指伸直，其他手指弯曲
    return (
        get_distance(index_tip, wrist) > 0.15 and
        get_distance(middle_tip, wrist) > 0.15 and
        get_distance(ring_tip, wrist) < 0.1 and
        get_distance(pinky_tip, wrist) < 0.1
    )

def is_ok(hand_landmarks):
    """判断OK手势"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return get_distance(thumb_tip, index_tip) < 0.05

def get_gesture(hand_landmarks):
    """综合判断手势类型"""
    if is_fist(hand_landmarks):
        return "Fist"
    if is_victory(hand_landmarks):
        return "Victory"
    if is_ok(hand_landmarks):
        return "OK"
    return "Open Hand"

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("无法读取摄像头画面。")
        continue

    # 图像预处理
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 获取手势类型
            gesture = get_gesture(hand_landmarks)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            position = (
                int(wrist.x * image.shape[1]),
                int(wrist.y * image.shape[0])
            )
            
            # 设置不同颜色
            color = (0, 255, 0) if gesture == "Fist" else (
                    0, 0, 255) if gesture == "Open Hand" else (
                    255, 0, 0) if gesture == "Victory" else (255, 255, 0)
            
            cv2.putText(image, gesture, position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 显示窗口并添加关闭检测
    cv2.imshow('费枭健202408784110', image)
    
    # 关闭指令（支持两种方式）
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('费枭健202408784110', cv2.WND_PROP_VISIBLE) < 1:
        break

# 释放资源
cap.release() 
cv2.destroyAllWindows()