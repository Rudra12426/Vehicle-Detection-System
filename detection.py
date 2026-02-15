# if you facing error run it in jupter notebook 
# use any traffic simulation video
# =====================================================
# ADVANCED VEHICLE COUNTING + SPEED + LANES + SOUND
# =====================================================

import cv2
from ultralytics import YOLO
import csv
import winsound
import numpy as np

print("✅ Libraries loaded")

# -------------------------------
# STEP 1: Load YOLO Model
# -------------------------------
model = YOLO("yolov8n.pt")  # YOLOv8 Nano
print("✅ YOLO model loaded")

# -------------------------------
# STEP 2: Video Input & FPS
# -------------------------------
video_path = "Car Video.mp4"  # Replace with your video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))
print(f"✅ Video loaded | FPS: {fps} | Resolution: {width}x{height}")

# Output video writer
out = cv2.VideoWriter(
    "output_annotated.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# -------------------------------
# STEP 3: Vehicle Classes
# -------------------------------
class_map = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}
vehicle_count = {v: set() for v in class_map.values()}  # unique IDs
vehicle_positions = {}  # previous y-center
vehicle_speeds = {}     # speed per vehicle
prev_max_vehicle = None  # for smart beep

# -------------------------------
# STEP 4: Lane Setup
# -------------------------------
lane_lines = [width // 3, 2 * width // 3]  # Example: 3 vertical lanes
lane_count = {
    "Lane 1": {v: set() for v in class_map.values()},
    "Lane 2": {v: set() for v in class_map.values()},
    "Lane 3": {v: set() for v in class_map.values()}
}

print("✅ Vehicle counters and lane setup ready")

# -------------------------------
# STEP 5: Main Loop
# -------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=list(class_map.keys())
    )

    # If there are tracked objects
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()
        classes_ids = results[0].boxes.cls.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for track_id, cls_id, box in zip(ids, classes_ids, boxes):
            track_id = int(track_id)
            cls_id = int(cls_id)
            vehicle_name = class_map[cls_id]

            # -------------------------------
            # Vehicle Counting
            # -------------------------------
            vehicle_count[vehicle_name].add(track_id)

            # -------------------------------
            # Speed Estimation
            # -------------------------------
            y_center = (int(box[1]) + int(box[3])) // 2
            prev_y = vehicle_positions.get(track_id, y_center)
            distance_pixels = abs(y_center - prev_y)
            vehicle_positions[track_id] = y_center

            meters_per_pixel = 0.05  # adjust for your camera
            distance_meters = distance_pixels * meters_per_pixel
            speed_m_s = distance_meters * fps
            speed_kmh = speed_m_s * 3.6
            vehicle_speeds[track_id] = speed_kmh

            # -------------------------------
            # Lane Assignment
            # -------------------------------
            x_center = (int(box[0]) + int(box[2])) // 2
            if x_center < lane_lines[0]:
                lane = "Lane 1"
            elif x_center < lane_lines[1]:
                lane = "Lane 2"
            else:
                lane = "Lane 3"

            lane_count[lane][vehicle_name].add(track_id)

            # -------------------------------
            # Annotate Lane + Speed
            # -------------------------------
            cv2.putText(frame, f"{vehicle_name}", (int(box[0]), int(box[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"{int(speed_kmh)} km/h", (int(box[0]), int(box[1]) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, lane, (int(box[0]), int(box[1]) - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # -------------------------------
    # Annotate Bounding Boxes
    # -------------------------------
    annotated_frame = results[0].plot()

    # -------------------------------
    # Display Counts on Frame
    # -------------------------------
    y_text = 30
    for vehicle, ids_set in vehicle_count.items():
        count = len(ids_set)
        cv2.putText(annotated_frame,
                    f"{vehicle}: {count}",
                    (20, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)
        y_text += 30

    # -------------------------------
    # Smart Sound Alert
    # -------------------------------
    if vehicle_count:
        max_vehicle = max(vehicle_count.items(), key=lambda x: len(x[1]))[0]
        if max_vehicle != prev_max_vehicle:
            winsound.Beep(1000, 200)  # play beep
            prev_max_vehicle = max_vehicle

    # Show frame
    cv2.imshow("Advanced Vehicle Counting", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# STEP 6: Release Resources
# -------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Video tracking finished")

# -------------------------------
# STEP 7: Save Counts to CSV
# -------------------------------
# Overall vehicle count
counts = {vehicle: len(ids_set) for vehicle, ids_set in vehicle_count.items()}
with open("vehicle_counts.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(counts.keys())
    writer.writerow(counts.values())

# Lane-wise count
with open("lane_wise_counts.csv", "w", newline="") as file:
    writer = csv.writer(file)
    header = ["Lane", "Vehicle", "Count"]
    writer.writerow(header)
    for lane, vehicles in lane_count.items():
        for vehicle, ids_set in vehicles.items():
            writer.writerow([lane, vehicle, len(ids_set)])

print("✅ Vehicle counts saved to CSV")
print("Final counts:", counts)
