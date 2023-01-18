import cv2
import mediapipe as mp
import time
import os

print('|| Exam Hall Monitoring System ||\n')
student_name = input('Enter Student Name : ')
student_id = input('Enter Student Id : ')

directory_name = student_name
directory_path = 'datasets/{}'.format(directory_name)


def create_directory(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print('Directory is created successfully...')
        else:
            print('Directory is already exist...')
    except OSError:
        print('Error: Creating directory. ' + dir_path)


create_directory(directory_path)


def fancy_rectangle(frame, rectangle_box, length=25, thickness=4, rect_width=1):
    x, y, w, h = rectangle_box
    x1, y1 = x + w, y + h

    # rectangles are top, right, bottom, left
    cv2.rectangle(frame, rectangle_box, (222, 196, 176), rect_width)
    # top left
    cv2.line(frame, (x, y), (x + length, y), (222, 196, 176), thickness)
    cv2.line(frame, (x, y), (x, y + length), (222, 196, 176), thickness)
    # top right
    cv2.line(frame, (x1, y), (x1 - length, y), (222, 196, 176), thickness)
    cv2.line(frame, (x1, y), (x1, y + length), (222, 196, 176), thickness)
    # bottom left
    cv2.line(frame, (x, y1), (x + length, y1), (222, 196, 176), thickness)
    cv2.line(frame, (x, y1), (x, y1 - length), (222, 196, 176), thickness)
    # bottom right
    cv2.line(frame, (x1, y1), (x1 - length, y1), (222, 196, 176), thickness)
    cv2.line(frame, (x1, y1), (x1, y1 - length), (222, 196, 176), thickness)

    return frame


cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDrawn = mp.solutions.drawing_utils

face_detection = mpFaceDetection.FaceDetection(0.75)
img_count = 0

while True:
    # Capture frame-by-frame
    success, img = cap.read()
    if not success:
        print('Failed to capture frame..')
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(imgRGB)

    if results.detections:
        for _, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape

            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            img = fancy_rectangle(img, bbox)

            img_id = student_name + '.{}.{}'.format(student_id, img_count)
            cv2.imwrite(directory_path + os.sep + img_id + ".jpg",
                        img[int(bboxC.ymin * ih):int(bboxC.ymin * ih) + int(bboxC.height * ih),
                        int(bboxC.xmin * iw):int(bboxC.xmin * iw) + int(bboxC.width * iw)])
            print('Image capture successfully.....')
            img_count = img_count + 1

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 87, 51), 2)

    resize = cv2.resize(img, (640, 480))

    cv2.imshow("Exam hall monitering system", resize)

    key = cv2.waitKey(1)

    if key == ord('q'):
        print('"Esc" & "q" button is pressed....')
        break
    # elif img_count >= 100:
    #     print('Image is saved successfully...')
    #     break
    # elif key == ord('s'):
    #     img_id = student_name + '.{}.{}'.format(student_id, img_count)
    #     cv2.imwrite(directory_path + os.sep + img_id + ".jpg", img)
    #     print('Image is saved...')
    #     img_count = img_count + 1

cap.release()
cv2.destroyAllWindows()
