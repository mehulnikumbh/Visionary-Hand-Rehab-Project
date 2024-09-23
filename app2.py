# Checking the global setting of the detected object.
# import base64

import cv2
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from flask_socketio import SocketIO, send
from ultralytics import YOLO
import mediapipe as mp
import time
import numpy as np
import math

class CamShift():

    # Function to initialize the variables and reading video frames:
    def __init__(cls):
        cls.FLAG_track = 0
        cls.region_of_interest = None
        # cls.rect_coord = None
        cls.cap = cv2.VideoCapture(0)
        cls.tracker_center = None
        # ret, cls.video_frame = cls.cap.read()

    def set_yolo_coordinates(self, yolo_coords):
        x1, y1, x2, y2 = yolo_coords  # Unpack coordinates
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        self.region_of_interest = (x1, y1, x2, y2)
        self.FLAG_track = 1  # Start tracking immediately

    def hue(cls):
        b = 24
        bins = cls.histogram.shape[0]
        img = np.zeros((256, bins * b, 3), np.uint8)
        for j in range(bins):
            s1 = int(cls.histogram[j])
            cv2.rectangle(img, (j * (b + 2), 255), ((j + 1) * (b - 2), 255 - s1), (int(180 * (j / bins)), 255, 255), -1)
        # Coverting image from HSV color space to RGB color space in order to show the histogram image in RGB format:
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def set_tracker_coords(self, center):
        x, y = center
        x = int(x)
        y = int(y)
        self.tracker_center = (x, y)

    def main(cls, frame):  ### can provide frame
        # while True:  # Loop used for processing all the frames
        #     ret, cls.video_frame = cls.cap.read()
        #     new_w = cls.video_frame.copy()
        new_w = frame.copy()
        # region_of_interest_hsv = cv2.cvtColor(cls.video_frame, cv2.COLOR_BGR2HSV)
        region_of_interest_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(region_of_interest_hsv, np.array((0., 30., 32.)), np.array((180., 255., 255.)))

        if cls.region_of_interest:
            a1, b1, a2, b2 = cls.region_of_interest
            cls.window = (a1, b1, (a2 - a1), (b2 - b1))
            image_hsv = region_of_interest_hsv[b1:b2, a1:a2]
            image_mask = mask[b1:b2, a1:a2]
            histogram = cv2.calcHist([image_hsv], [0], image_mask, [16], [0, 180])
            cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
            cls.histogram = histogram.reshape(-1)

            # Calling the 'hue' function to choose the region shape to be tracked based on histogram bins:
            cls.hue()
            region_new_w = new_w[b1:b2, a1:a2]
            cv2.bitwise_not(region_new_w, region_new_w)

        if cls.FLAG_track == 1:
            cls.region_of_interest = None
            probability = cv2.calcBackProject([region_of_interest_hsv], [0], cls.histogram, [0, 180], 1)
            probability &= mask
            terminate = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)
            track, cls.window = cv2.CamShift(probability, cls.window, terminate)
            # Assuming 'track' contains the result of cv2.CamShift
            center_x, center_y = track[0]
            combined_center = (int(center_x), int(center_y))
            cls.set_tracker_coords(combined_center)
            cv2.circle(frame, combined_center, 5, (0, 0, 255), -1)
            cv2.ellipse(frame, track, (255, 0, 255), 3)  ##Tracking print.
            # print(type(track))
            # cv2.imshow('Tracking', new_w)
        # return new_w


# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app)
selected_classes = []
# 'ball', 'credit card', 'cup', 'hand', 'marker'
my_dict = {'ball': 0, 'credit card': 1, 'cup': 2, 'marker': 4}
yolo_list = []
# detected_objects = []
detected_item = None
Btn1 = "LOW"
Buzz = "off"
count_variable = 0
ble_connected = False
Video_page = False

# Load YOLOv8 model (adjust the path as needed)
model = YOLO("epoch5.pt")
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

tracker = CamShift()

# Frame skipping factor (process every Nth frame for non-critical operations)
frame_skip_factor = 10  # Adjust as needed
frame_counter = 0
grip_timer = 0
grip_duration_threshold = 1  # 3 seconds threshold for grip

start_time = time.time()
start_time_yo = time.time()
frame_count = 0
fps = 0
grip = {
    0: 1,
    1: 2,
    2: 3,
    4: 4,
}
distanceTI = 0

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Function to generate video frames with object detection
def gen_frames(Video_page):
    global Btn1, count_variable, ble_connected, grip_timer, Buzz, distanceTI

    if Video_page:
        cap = cv2.VideoCapture(1)
        count_variable = 0

    while Video_page:

        detected_objects = []
        detected_objects.clear()


        # Read frame from camera if BLE device is connected
        if ble_connected:
            success, img = cap.read()
            h, w, c = img.shape

        else:
            # If BLE device is not connected, yield an empty frame
            img = None
            success = False

        # Perform object detection using YOLOv8 if frame is available
        if img is not None:
            Btn1 = "LOW"
            socketio.send('object_not_detected')
            socketio.send('no_grip')
            print(Buzz)
            if Buzz == 'on':
                Btn1 = "HIGH"
                socketio.send('grip')
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results_mp = hands.process(imgRGB)
            # Mediapipe
            center_x = 0
            center_y = 0
            center_mp = (0, 0)
            if results_mp.multi_hand_landmarks:
                for handLms in results_mp.multi_hand_landmarks:
                    x_min, y_min, x_max, y_max = float('inf'), float('inf'), -float('inf'), -float('inf')
                    handedness = h in handLms.landmark
                    x_coords = [lm.x for lm in handLms.landmark]
                    y_coords = [lm.y for lm in handLms.landmark]

                    thumb_landmark = handLms.landmark[mpHands.HandLandmark.THUMB_TIP]
                    index_finger_landmark = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]

                    # Get coordinates of thumb and index finger landmarks
                    thumb_x, thumb_y = int(thumb_landmark.x * img.shape[1]), int(thumb_landmark.y * img.shape[0])
                    index_x, index_y = int(index_finger_landmark.x * img.shape[1]), int(
                        index_finger_landmark.y * img.shape[0])

                    # Calculate distance between thumb and index finger
                    distanceTI = calculate_distance(thumb_x, thumb_y, index_x, index_y)

                    # Calculate simple arithmetic mean for x and y
                    centroid_x = sum(x_coords) / len(x_coords)
                    centroid_y = sum(y_coords) / len(y_coords)
                    cv2.circle(img, (int(centroid_x * w), int(centroid_y * h)), 5, (0, 234, 250), -1)
                    # print("X: ",centroid_x,"Y: ",centroid_y)
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                    center_x = int(centroid_x * w)
                    center_y = int(centroid_y * h)
                    center_mp = (center_x, center_y)

            # Perform object detection using YOLOv8
            results = model(img, classes=yolo_list, conf=0.4)
            # results = model(frame)

            # Extract bounding boxes, classes, names, and confidences
            boxes = results[0].boxes.xyxy.tolist()
            classes = results[0].boxes.cls.tolist()
            names = results[0].names
            confidences = results[0].boxes.conf.tolist()
            count_variable += 1
            # Iterate through the results
            cxmp, cymp = center_mp

            # Iterate through the results
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box
                confidence = conf
                detected_class = cls
                name = names[int(cls)]
                detected_objects.append(name)
                # print(detected_objects)


                centeryolo_x = int(x1 + x2) // 2
                centeryolo_y = int(y1 + y2) // 2
                centeryolo = (centeryolo_x, centeryolo_y)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(img, centeryolo, 5, (0, 0, 255), -1)
                cv2.putText(img, f"{name} {confidence:.2f} class id {int(cls)}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if center_mp > (0, 0):
                    cv2.line(img, centeryolo, center_mp, (0, 0, 255), 5)
                    distance = int(math.sqrt(((center_x - centeryolo_x) ** 2) + ((center_y - centeryolo_y) ** 2)))
                    # cv2.putText(img, str(distance),
                    #             (((centeryolo_x + center_x) // 2), ((centeryolo_y + center_y) // 2)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    if distance <= 200 and \
                            (name == "ball" or name == "marker" or name == "cup" or name == "credit card") \
                            and tracker.FLAG_track == 0:
                        # cv2.putText(img, f"Gripping {name}!", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        # Check for grip
                        if distance <= 150 and tracker.FLAG_track == 0:
                            tracker.set_yolo_coordinates(box)
                            tracker.FLAG_track = 1
                            break
                        else:
                            grip_timer = 0
                            # cv2.putText(img, 'NO GRIP :(', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Track object using CamShift
            print(detected_objects)
            for name in detected_objects:
                if name == 'ball':
                    Btn1 = "HIGH"
                    socketio.send('object_detected')
                if name == 'credit card':
                    Btn1 = "HIGH"
                    socketio.send('object_detected2')
                if name == 'cup':
                    Btn1 = "HIGH"
                    socketio.send('object_detected3')
                if name == 'marker':
                    Btn1 = "HIGH"
                    socketio.send('object_detected4')
                # if Buzz == 'off':
                #     Btn1 = "HIGH"
                #     socketio.send('grip')
                else:
                    Btn1 = "LOW"
                    socketio.send('object_not_detected')
                    socketio.send('no_grip')
            if tracker.FLAG_track == 1:
                tracker.main(img)
                tracker_center = tracker.tracker_center
                tcx, tcy = tracker_center

                distance_mt = int(math.sqrt(((cxmp - tcx) ** 2) + ((cymp - tcy) ** 2)))
                cv2.line(img, tracker_center, center_mp, (0, 0, 255), 5)
                cv2.putText(img, str(distance_mt),
                            (((tcx + cxmp) // 2), ((tcy + cymp) // 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(img, str(int(distanceTI)),
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if distance_mt <= 150 and distanceTI <= 100:
                    grip_timer += 1
                    if grip_timer >= grip_duration_threshold * 10:  # 30 frames per second
                        cv2.putText(img, 'GRIP!!!', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        Buzz = "on"
                        socketio.send('grip')

                        # cv2.putText(img, f'Elapsed time: {grip_timer}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # else:
                        # cv2.putText(img, 'GETTING CLOSER!', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    grip_timer = 0
                if distance_mt >= 200:
                    tracker.FLAG_track = 0
                    Buzz = "off"
                    socketio.send('no_grip')
                    continue

            # Encode frame as JPEG and yield it
            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            frame_byte = b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n'
            # yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            # Convert detected_objects to string
            # detected_objects_str = ', '.join(detected_objects)

            # Yield frame and detected_objects as bytes and string respectively
            # yield frame_byte, str(count_variable) + ',' + Btn1
            yield frame_byte


# Routes for the Flask app
@app.route('/')
def index():
    global Video_page
    Video_page = False
    return render_template('indexFE.html')


@app.route('/submit', methods=['POST'])
def submit():
    global selected_classes
    selected_classes = request.form.getlist('class')
    global yolo_list
    yolo_list = [my_dict[key] for key in selected_classes]
    return redirect(url_for('results'))


@app.route('/results')
def results():
    return render_template('index2FE.html', selected_classes=selected_classes, detected_item=detected_item, Btn1=Btn1)


@app.route('/video_feed')
def video_feed():
    global Video_page
    Video_page = True
    def generate():
        for frames in gen_frames(Video_page):
            yield frames

    # return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/count_variable')
def count_variable_route():
    global count_variable
    return str(count_variable)


@app.route('/button_state')
def button_state_route():
    global Btn1
    return Btn1


@app.route('/stream_list')
def stream_list():
    return Response(Btn1, mimetype='text/plain')


# SocketIO connection event
@socketio.on('connect')
def handle_connect():
    global ble_connected
    print('BLE device connected')
    ble_connected = True


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)

