import csv
import cv2
from datetime import datetime
import os
import serial

port = "/dev/ttyACM0"
baud_rate = 9600
ser = serial.Serial(port, baud_rate)

video_stream = cv2.VideoCapture(0)

if not os.path.exists('../train_img'):
    os.makedirs('../train_img/img')

csv_path = '../train_img/labels.csv'
path_img_save = '../train_img/img/'

key_pressed = None

video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
video_stream.set(cv2.CAP_PROP_FPS, 30)

print('let the training begin')

with open(csv_path, 'a', newline='\n') as file:
    writer = csv.writer(file)
    while True:
        start = datetime.now()
        retrieved_frame, rgb_imb = video_stream.read()
        cv2.imshow("label", rgb_imb)
        end = datetime.now()

        if key_pressed in [None, 'x']:
            time_elapsed = (end - start).total_seconds()
            print('it took {} s between each frame'.format(time_elapsed))

        key = cv2.waitKey(1) & 0xFF
        key = chr(key)

        if key is 'q':
            ser.write('x'.encode())
            break

        if key is 'x':
            ser.write('x'.encode())
            key_pressed = 'x'
            print('image collection paused')

        if key in ['w', 'a', 'd']:
            key_pressed = key

        if key_pressed not in [None, 'x']:
            img_path = path_img_save + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + ".jpg"
            cv2.imwrite(img_path, rgb_imb)
            writer.writerow([img_path, key_pressed])
            ser.write(key_pressed.encode())

            print("Saved image {} with key pressed {}".format(img_path, key_pressed))

video_stream.release()
cv2.destroyAllWindows()
