import cv2

video_type = 'return'
video_path = f'motion_{video_type}.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open the video file.")
    exit()

frame_count = 0
saved_frame = None
points = []
y_line = None

# Mouse callback function to select points on the frame
def select_point(event, x, y, flags, param):
    global points, y_line, saved_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)} selected: {x, y}")
        elif y_line is None:
            y_line = y
            print(f"Line selected at y-coordinate: {y}")

        # Redraw the frame with the selected points and line
        redraw_frame()

# Function to redraw the frame with annotations
def redraw_frame():
    global saved_frame, points, y_line
    temp_frame = saved_frame.copy()

    # Draw selected points
    for idx, (px, py) in enumerate(points):
        cv2.circle(temp_frame, (px, py), 5, (0, 255, 0), -1)
        cv2.putText(temp_frame, f"Point {idx + 1}", (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw the y-line
    if y_line is not None:
        cv2.line(temp_frame, (0, y_line), (temp_frame.shape[1], y_line), (0, 0, 255), 2)
        cv2.putText(temp_frame, f"y-line: {y_line}", (10, y_line - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('Video Frame', temp_frame)

cv2.namedWindow('Video Frame')
cv2.setMouseCallback('Video Frame', select_point)

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or error reading frame.")
        break

    frame_count += 1
    print(f"Reading frame {frame_count}")

    cv2.imshow('Video Frame', frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        print("Exiting without saving.")
        break
    elif key == ord('p'):
        saved_frame = frame.copy()
        print("Paused at frame number:", frame_count)
        redraw_frame()

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("Exiting without saving.")
                break
            elif key == ord('r'):
                print("Resetting points and line.")
                points = []
                y_line = None
                redraw_frame()
            elif key == ord('s'):
                if len(points) == 4:
                    print("Saving points.")
                    break
                elif len(points) < 4:
                    print("Please select 4 points before saving.")
        if key == ord('s'):
            break

cap.release()
cv2.destroyAllWindows()


if len(points) == 4:
    x_coords = [x for x, _ in points]
    y_coords = [y for _, y in points]
    
    print(x_coords, y_coords)
    x1 = min(x_coords)
    y1 = min(y_coords)
    w = max(x_coords) - x1
    h = max(y_coords) - y1

    with open(f"motion_{video_type}_points.txt", "w") as f:
        f.write(f"{x1},{y1},{w},{h}\n")
        # f.write(f"{xc},{yc}\n")

    print(f"Points saved to motion_{video_type}_points.txt")
    print(f"x1, y1: {x1}, {y1}")
    print(f"w, h: {w}, {h}")
else:
    print("Insufficient data to save. Make sure to select 4 points.")

# Print the y-line
if y_line is not None:
    print(f"Selected y-line at y-coordinate: {y_line}")
else:
    print("No y-line was selected.")
