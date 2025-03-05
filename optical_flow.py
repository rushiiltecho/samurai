import numpy as np
import cv2

def generate_points_inside_polygon(polygon_coords, num_points=50):
    
        print(polygon_coords, type(polygon_coords))
        min_x = max_x = polygon_coords[0][0]
        min_y = max_y = polygon_coords[0][1]    
        for x, y in polygon_coords[1:]:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

        random_points = []

        while len(random_points) < num_points:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)

            if is_point_inside_polygon(x, y, polygon_coords):
                random_points.append((x, y))

        return random_points

def is_point_inside_polygon(x, y, polygon_coords):
    n = len(polygon_coords)
    inside = False

    p1x, p1y = polygon_coords[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_coords[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def lucas_kanade_polygon_tracking(video_path, segment):
    """
    Implement Lucas-Kanade optical flow for tracking points within a polygon segment.
    """
    # Initialize webcam
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get first frame to setup
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam")
        return
    
    frame_height, frame_width = first_frame.shape[:2]
    starting_y_coordinates = None
    # Define polygon vertices (example polygon - modify as needed)
    # polygon_points = np.array([
    #     [100, 100],  # Top-left
    #     [300, 100],  # Top-right
    #     [400, 200],  # Middle-right
    #     [300, 300],  # Bottom-right
    #     [100, 300],  # Bottom-left
    # ], dtype=np.float32)

    polygon_points = np.array(
        segment, dtype=np.float32
    )
    
    # Initialize points within the polygon
    p0 = generate_points_inside_polygon(polygon_points)
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Create random colors for visualization
    color = np.random.randint(0, 255, (len(p0), 3))
    
    # Create mask for drawing motion tracks
    mask = np.zeros_like(first_frame)
    
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            old_gray, 
            frame_gray, 
            p0, 
            None, 
            **lk_params
        )
        
        # Select good points
        if p1 is not None:
            good_new = p1[status == 1]
            good_old = p0[status == 1]

            if starting_y_coordinates is None:
                # starting_y_coordinates = good_new
                starting_y_coordinates = [np.mean([x[0] for x in good_new]), np.mean([x[1] for x in good_new])]
            # print(good_new)
            # ending_y_ccordinates = good_new
            ending_y_coordinates = [np.mean([x[0] for x in good_new]), np.mean([x[1] for x in good_new])]

            
            # Draw polygon outline
            cv2.polylines(frame, [polygon_points.astype(np.int32)], 
                         True, (0, 255, 0), 2)
            
            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                
                # Only draw if points are within frame bounds
                if (0 <= a < frame_width and 0 <= b < frame_height and
                    0 <= c < frame_width and 0 <= d < frame_height):
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), 
                                  color[i].tolist(), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 3, 
                                     color[i].tolist(), -1)
            
            img = cv2.add(frame, mask)
            
            # Display point count
            cv2.putText(img, f'Tracking points: {len(good_new)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            cv2.imshow('Polygon Tracking', img)
            
            # Update points
            p0 = good_new.reshape(-1, 1, 2)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('r'):  # Reset points
            mask = np.zeros_like(first_frame)
            p0 = generate_points_inside_polygon(polygon_points)
        
        # Update previous frame
        old_gray = frame_gray.copy()
    
    cap.release()
    cv2.destroyAllWindows()

    motion_length = ending_y_coordinates[0] - starting_y_coordinates[0]
    print(ending_y_coordinates,starting_y_coordinates)
    if motion_length < 0 :
        if abs(motion_length) > 20:
            motion_status = "grab"

        else:
            motion_status = "invalid"
    else:

        if abs(motion_length) > 20:
            motion_status = "return"
        else:
            motion_status = "inavlid"


    return motion_status