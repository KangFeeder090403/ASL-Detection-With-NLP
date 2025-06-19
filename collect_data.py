# collect_data.py
import cv2
import numpy as np
import os
import string
from cvzone.HandTrackingModule import HandDetector

# Konfigurasi
DATASET_PATH = "dataset_isyarat"
IMG_WIDTH, IMG_HEIGHT = 400, 400 # Ukuran gambar skeleton yang akan disimpan
OFFSET = 29 # Offset untuk cropping ROI

# Inisialisasi
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat membuka webcam.")
    exit()

detector_main = HandDetector(maxHands=1)
detector_roi = HandDetector(maxHands=1) # Untuk deteksi di ROI

# Daftar kelas (alfabet A-Z, dan mungkin gestur lain)
# Anda bisa memperluas ini dengan 'Space', 'Next', 'Backspace'
# classes = list(string.ascii_uppercase)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'Space', 'Next', 'Backspace'] # Sesuaikan dengan kebutuhan Anda
current_class_idx = 0
current_class = classes[current_class_idx]
img_counter = 0
max_imgs_per_class = 1000 # Jumlah gambar per kelas yang ingin dikumpulkan
collecting = False

# Buat folder dataset jika belum ada
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
for cls_name in classes:
    class_path = os.path.join(DATASET_PATH, cls_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    else:
        # Hitung gambar yang sudah ada untuk kelas ini agar tidak menimpa
        # atau untuk melanjutkan pengumpulan data
        img_counter = len(os.listdir(class_path)) if cls_name == current_class else 0


print(f"Mulai mengumpulkan data untuk kelas: {current_class}")
print(f"Tekan 's' untuk mulai/berhenti menyimpan gambar untuk kelas saat ini.")
print(f"Tekan 'n' untuk pindah ke kelas berikutnya.")
print(f"Tekan 'q' untuk keluar.")

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Tidak bisa membaca frame.")
        break

    frame_flipped = cv2.flip(frame, 1)
    display_frame = frame_flipped.copy() # Frame untuk menampilkan info

    hands_data, _ = detector_main.findHands(frame_flipped.copy(), draw=False)
    img_skeleton_display = np.ones((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8) * 255

    if hands_data:
        hand = hands_data[0]
        bbox = hand['bbox']
        x, y, w, h = bbox

        y_start = max(0, y - OFFSET)
        y_end = min(frame_flipped.shape[0], y + h + OFFSET)
        x_start = max(0, x - OFFSET)
        x_end = min(frame_flipped.shape[1], x + w + OFFSET)

        img_roi = frame_flipped[y_start:y_end, x_start:x_end]

        if img_roi.size > 0:
            hands_in_roi, _ = detector_roi.findHands(img_roi.copy(), draw=False)

            if hands_in_roi:
                hand_in_roi = hands_in_roi[0]
                landmarks = hand_in_roi['lmList']

                # Gambar skeleton
                img_skeleton_save = np.ones((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8) * 255
                original_w_roi, original_h_roi = w, h
                offset_x_skeleton = (IMG_WIDTH - original_w_roi) // 2
                offset_y_skeleton = (IMG_HEIGHT - original_h_roi) // 2
                offset_x_skeleton = max(0, offset_x_skeleton - 15)
                offset_y_skeleton = max(0, offset_y_skeleton - 15)

                processed_landmarks = []
                for lm_idx, lm in enumerate(landmarks):
                    x_lm, y_lm = int(lm[0] + offset_x_skeleton), int(lm[1] + offset_y_skeleton)
                    x_lm = max(0, min(x_lm, IMG_WIDTH - 1))
                    y_lm = max(0, min(y_lm, IMG_HEIGHT - 1))
                    processed_landmarks.append((x_lm, y_lm, lm[2]))
                    cv2.circle(img_skeleton_save, (x_lm, y_lm), 3, (0, 0, 255), -1)
                    cv2.circle(img_skeleton_display, (x_lm, y_lm), 3, (0, 0, 255), -1)


                hand_connections = [
                    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20), (5,9),(9,13),(13,17)
                ]
                for conn in hand_connections:
                    p1 = processed_landmarks[conn[0]]
                    p2 = processed_landmarks[conn[1]]
                    cv2.line(img_skeleton_save, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 0), 2)
                    cv2.line(img_skeleton_display, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 0), 2)


                if collecting and img_counter < max_imgs_per_class:
                    img_name = f"{current_class}_{img_counter}.png"
                    save_path = os.path.join(DATASET_PATH, current_class, img_name)
                    cv2.imwrite(save_path, img_skeleton_save)
                    print(f"Menyimpan: {save_path}")
                    img_counter += 1
                    if img_counter >= max_imgs_per_class:
                        collecting = False # Otomatis berhenti jika sudah maks
                        print(f"Pengumpulan untuk kelas {current_class} selesai ({max_imgs_per_class} gambar). Tekan 'n' untuk kelas berikutnya.")


    # Tampilkan informasi di frame
    cv2.putText(display_frame, f"Kelas: {current_class} ({img_counter}/{max_imgs_per_class})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Collecting: {'YA' if collecting else 'TIDAK'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if collecting else (255,0,0), 2)
    cv2.putText(display_frame, "S: Start/Stop, N: Next Class, Q: Quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Pengumpulan Data - Webcam", display_frame)
    cv2.imshow("Pengumpulan Data - Skeleton", img_skeleton_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        collecting = not collecting
        if collecting:
            # Reset counter jika memulai lagi untuk kelas yang sama dan belum penuh
            class_dir_path = os.path.join(DATASET_PATH, current_class)
            img_counter = len(os.listdir(class_dir_path))
            print(f"Mulai menyimpan untuk kelas {current_class}. Gambar tersimpan: {img_counter}")
        else:
            print(f"Berhenti menyimpan untuk kelas {current_class}.")
    elif key == ord('n'):
        collecting = False
        current_class_idx = (current_class_idx + 1) % len(classes)
        current_class = classes[current_class_idx]
        class_dir_path = os.path.join(DATASET_PATH, current_class)
        img_counter = len(os.listdir(class_dir_path)) # Update counter untuk kelas baru
        print(f"Pindah ke kelas: {current_class}. Gambar tersimpan: {img_counter}")


cap.release()
cv2.destroyAllWindows()
