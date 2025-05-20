import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# EAR 계산 함수
def calculate_ear(landmarks, eye_indices):
    A = np.linalg.norm(np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) -
                       np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]))
    B = np.linalg.norm(np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y]) -
                       np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]))
    C = np.linalg.norm(np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) -
                       np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y]))
    ear = (A + B) / (2.0 * C)
    return ear

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)

glabella_indices = [107, 9, 336, 108, 151, 337]
zygomatic_indices = [50, 117, 116, 118, 280, 347, 346, 340, 352, 123, 111]
target_indices = glabella_indices + zygomatic_indices
anchor_idx = 1  # 코끝 기준점

history = {idx: [] for idx in target_indices}
glabella_movement_avg = []
zygomatic_movement_avg = []
ear_history = []

# 눈 랜드마크 인덱스 (MediaPipe 기준)
LEFT_EYE_IDX = [33, 159, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 380, 374, 263, 386, 385]
EAR_THRESHOLD = 0.3

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle('근육 움직임, 눈 깜빡임(EAR) 실시간 그래프')

# 근육 이동량 plot
line1, = ax1.plot([], [], label='미간근육 평균 이동량', color='orange')
line2, = ax1.plot([], [], label='광대뼈주변근육 평균 이동량', color='blue')
ax1.set_ylabel('평균 이동량 (픽셀)')
ax1.legend()
ax1.grid(True)

# EAR plot
ear_line, = ax2.plot([], [], color='lime', label='EAR')
ax2.axhline(EAR_THRESHOLD, color='blue', linestyle='--', label='EAR 임계값')
ax2.set_ylabel('EAR')
ax2.set_xlabel('Frame')
ax2.set_ylim(0.15, 0.45)
ax2.legend()
ax2.grid(True)

window_size = 300
frame_num = 0

EAR_THRESHOLD = 0.3  # EAR 임계값
blink_counter = 0
blink_timestamps = []
fps = 30
window_sec = 5

blink_state = False  # False: 눈 뜸(초록), True: 눈 감김(빨강)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape[:]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    ear = None
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            anchor_lm = face_landmarks.landmark[anchor_idx]
            anchor_x, anchor_y = int(anchor_lm.x * w), int(anchor_lm.y * h)

            glabella_movements = []
            zygomatic_movements = []
            frame_num += 1

            # EAR 계산
            left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE_IDX)
            right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE_IDX)
            ear = (left_ear + right_ear) / 2.0
            ear_history.append(ear)

            for idx in target_indices:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                rel_x, rel_y = x - anchor_x, y - anchor_y
                history[idx].append((rel_x, rel_y))

                # 변화량 계산
                if len(history[idx]) > 1:
                    prev = np.array(history[idx][-2])
                    curr = np.array(history[idx][-1])
                    movement = np.linalg.norm(curr - prev)
                else:
                    movement = 0

                if idx in glabella_indices:
                    glabella_movements.append(movement)
                else:
                    zygomatic_movements.append(movement)

                # 시각화
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                
            eye_color = (0, 0, 255) if ear is not None and ear < EAR_THRESHOLD else (0, 255, 0)

            for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX: 
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, eye_color, -1)

            # 부위별 평균 이동량 저장
            if glabella_movements:
                glabella_movement_avg.append(np.mean(glabella_movements))
            if zygomatic_movements:
                zygomatic_movement_avg.append(np.mean(zygomatic_movements))
            if ear is not None:
                eye_color = (0, 0, 255) if ear < EAR_THRESHOLD else (0, 255, 0)
                # 눈 깜빡임 카운트 (빨강→초록 전환 시 카운트)
                if ear < EAR_THRESHOLD:
                    blink_state = True
                else:
                    if blink_state:
                        blink_counter += 1
                        blink_timestamps.append(frame_num)
                    blink_state = False

                for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX: 
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, eye_color, -1)

    # --- 최근 N초간 깜빡임 빈도 계산 및 표시 ---
    window_frames = window_sec * fps
    recent_blinks = [t for t in blink_timestamps if frame_num - t < window_frames]
    recent_blink_rate = len(recent_blinks) / window_sec
    total_blink_rate = blink_counter / (frame_num / fps) if frame_num > 0 else 0

    if recent_blink_rate >= 2.5:
        cv2.putText(frame, "lie", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    else:
        cv2.putText(frame, f"Blink Frequency: {recent_blink_rate:.2f}/s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    cv2.putText(frame, f"Blinks: {blink_counter}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
    # 캠화면 표시
    cv2.imshow('FaceMesh', frame)

    # 실시간 그래프 업데이트 (슬라이딩 윈도우)
    frames = np.arange(len(glabella_movement_avg))
    if len(frames) > window_size:
        frames_to_plot = frames[-window_size:]
        glabella_to_plot = glabella_movement_avg[-window_size:]
        zygomatic_to_plot = zygomatic_movement_avg[-window_size:]
        ear_to_plot = ear_history[-window_size:]
    else:
        frames_to_plot = frames
        glabella_to_plot = glabella_movement_avg
        zygomatic_to_plot = zygomatic_movement_avg
        ear_to_plot = ear_history

    # 근육 이동량 plot
    line1.set_data(frames_to_plot, glabella_to_plot)
    line2.set_data(frames_to_plot, zygomatic_to_plot)
    ax1.set_xlim(frames_to_plot[0] if len(frames_to_plot) > 0 else 0, 
                 frames_to_plot[-1] if len(frames_to_plot) > 0 else window_size)
    ax1.relim()
    ax1.autoscale_view(scaley=True)

    # EAR plot
    ear_line.set_data(frames_to_plot, ear_to_plot)
    ax2.set_xlim(frames_to_plot[0] if len(frames_to_plot) > 0 else 0, 
                 frames_to_plot[-1] if len(frames_to_plot) > 0 else window_size)
    ax2.relim()
    ax2.autoscale_view(scaley=True)

    plt.draw()
    plt.pause(0.001)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
