import cv2
import numpy as np
import tflite_runtime.interpreter as lite
import serial

ser = serial.Serial('/dev/ttyACM0', 9600)

video_stream = cv2.VideoCapture(0)
print('video stream opened: {}'.format(video_stream.isOpened()))

model = lite.Interpreter('simple_model.tflite')
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()
print('tf lite model loaded')

video_stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
video_stream.set(cv2.CAP_PROP_FPS, 30)

directions = ['a', 'd', 'w']

start_driving = False

while True:
    retrieved_frame, rgb_img = video_stream.read()
    cv2.imshow('car video feed', rgb_img)

    rgb_img = cv2.resize(rgb_img, (320, 180), interpolation=cv2.INTER_NEAREST)
    rgb_img = np.expand_dims(rgb_img, axis=0)
    
    model.set_tensor(input_details[0]['index'], rgb_img)
    model.invoke()
    predictions = model.get_tensor(output_details[0]['index'])

    index_direction = np.argmax(predictions)
    print('the direction chosen is {}'.format(directions[index_direction]))
    print('probabilities: {}'.format(predictions))

    key = cv2.waitKey(1) & 0xFF
    key = chr(key)

    if key == 'w':
        start_driving = True
        print('autonomous driving engaged')
    elif key == 'q':
        ser.write('x'.encode())
        start_driving = False
        print('autonomous driving stopped')
    elif key == 'x':
        ser.write('x'.encode())
        print('autonomous driving stopped... will now proceed to release resources')
        break

    if start_driving:
        print('driving {}'.format(directions[index_direction]))
        ser.write(directions[index_direction].encode())

video_stream.release()
cv2.destroyAllWindows()

print('see you next time')
