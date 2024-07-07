import cv2
import numpy as np
import pandas as pd
import datetime

# Load the video
video_path = 'video.mp4'  # Path to your video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize variables
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
quadrant_width = frame_width // 2
quadrant_height = frame_height // 2

# Define color ranges in HSV
colors = {
    'yellow': ((25, 100, 100), (35, 255, 255)),
    'white': ((0, 0, 200), (180, 30, 255)),
    'peach': ((5, 50, 50), (15, 255, 255)),
    'green': ((40, 50, 50), (80, 255, 255))
}

entries_exits = []
ball_positions = {}

def get_quadrant(x, y):
    if x < quadrant_width and y < quadrant_height:
        return 1
    elif x >= quadrant_width and y < quadrant_height:
        return 2
    elif x < quadrant_width and y >= quadrant_height:
        return 3
    else:
        return 4

def track_ball(ball_color, lower_bound, upper_bound):
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            (x, y, w, h) = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            current_quadrant = get_quadrant(center[0], center[1])
            ball_positions[ball_color] = center
            return current_quadrant, center
    return None, None

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print("Error: Could not open video writer.")
    exit()

start_time = datetime.datetime.now()

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Finished processing video.")
        break

    frame_count += 1
    if frame_count % 30 == 0:  # Process every 30th frame for faster execution
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        timestamp = (datetime.datetime.now() - start_time).total_seconds()

        for color_name, (lower, upper) in colors.items():
            current_quadrant, center = track_ball(color_name, lower, upper)
            if current_quadrant and center:
                if color_name not in ball_positions:
                    entries_exits.append([timestamp, current_quadrant, color_name, 'Entry'])
                    ball_positions[color_name] = current_quadrant
                elif ball_positions[color_name] != current_quadrant:
                    entries_exits.append([timestamp, ball_positions[color_name], color_name, 'Exit'])
                    entries_exits.append([timestamp, current_quadrant, color_name, 'Entry'])
                    ball_positions[color_name] = current_quadrant

                cv2.circle(frame, center, 10, (0, 255, 0), -1)
                cv2.putText(frame, f'{color_name} Ball', (center[0] - 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f'{timestamp:.2f}s {current_quadrant}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        print(f"Frame {frame_count} written to output.avi")

cap.release()
out.release()
print("Processing complete. Outputs saved to output.avi and entries_exits.csv.")

# Save entries and exits to a CSV file
df = pd.DataFrame(entries_exits, columns=['Time', 'Quadrant Number', 'Ball Colour', 'Type'])
df.to_csv('entries_exits.csv', index=False)
print("Entries and exits saved to entries_exits.csv.")

