import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
from gesture_detector.hand_processing.landmark_ext_frame_lib import FrameAdder

X_LEN = 640
Y_LEN = 480
PLOT = True
MIN_DEPTH = 0.2
MAX_DEPTH = 1.2  # Adjust based on your environment
MAX_POINT_DISTANCE = 0.1  # Max acceptable movement between frames

def main():
    # Initialize MediaPipe and RealSense
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, X_LEN, Y_LEN, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, X_LEN, Y_LEN, rs.format.bgr8, 30)
    pipeline.start(config)

    # **Add alignment to align depth to color frame**
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Get camera intrinsics (after pipeline start)
    profile = pipeline.get_active_profile()
    depth_stream = profile.get_stream(rs.stream.depth)
    intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

    import rclpy
    from rclpy.node import Node
    from gesture_msgs.msg import Frame

    rclpy.init()
    rosnode = Node("frame_adder")
    frame_adder = FrameAdder()
    frame_publisher = rosnode.create_publisher(Frame, '/hand_frame', 5)

    # Initialize filters
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    # Initialize previous points dictionary
    previous_points = {}

    try:
        while True:
            # Capture frames
            frames = None
            while frames is None:
                try:
                    frames = pipeline.wait_for_frames()
                except RuntimeError:
                    print("Waiting for frames failed resetting")
                    ctx = rs.context()
                    devices = ctx.query_devices()
                    for dev in devices:
                        dev.hardware_reset()  

            # **Align the depth frame to color frame**
            aligned_frames = align.process(frames)
            
            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Apply filters to depth frame
            filtered_depth_frame = spatial.process(depth_frame)
            filtered_depth_frame = temporal.process(filtered_depth_frame)
            filtered_depth_frame = hole_filling.process(filtered_depth_frame)
            depth_frame = filtered_depth_frame.as_depth_frame()
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Process image with MediaPipe
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_image)

            if result.multi_hand_landmarks:
                hand_landmarks_3d = np.empty((len(result.multi_hand_landmarks), 21, 3))
                hand_landmarks_3d.fill(np.nan)
                for hand, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    for i, lm in enumerate(hand_landmarks.landmark):
                        # Convert normalized coordinates to pixel coordinates
                        h, w, _ = color_image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)

                        if not (0 <= cx < w) or not (0 <= cy < h):  # Corrected boundary check
                            continue 
                        # Get depth at the landmark's pixel
                        depth = depth_frame.get_distance(cx, cy)
                        if depth == 0 or depth < MIN_DEPTH or depth > MAX_DEPTH:
                            # Invalid depth, attempt to interpolate or skip
                            depth = get_valid_depth_average(depth_frame, cx, cy)
                            if depth is None:
                                continue  # Skip if still invalid

                        # Map to 3D point
                        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
                        hand_landmarks_3d[hand][i] = [-point_3d[0] * 1000, point_3d[2] * 1000, point_3d[1] * 1000] # [m] to [mm], tf to toolbox
                        
                        # Check for outliers
                        key = i  # Use landmark index as key
                        prev_point = previous_points.get(key)
                        if is_outlier(prev_point, point_3d, MAX_POINT_DISTANCE):
                            point_3d = prev_point if prev_point else point_3d
                        else:
                            # Smooth point
                            point_3d = smooth_point(prev_point, point_3d, alpha=0.5)
                            previous_points[key] = point_3d

                        
                        if PLOT:
                            # Draw landmark on the color image
                            cv2.circle(color_image, (cx, cy), 3, (0, 255, 0), -1)

                frame = frame_adder.add_frame(hand_landmarks_3d)
                frame_publisher.publish(frame.to_ros())

            # Display the image
            if PLOT:
                cv2.imshow('Hand Landmarks', color_image)
                if cv2.waitKey(1) == ord('q'):
                    break

    finally:
        hands.close()
        pipeline.stop()
        cv2.destroyAllWindows()

def is_outlier(prev_point, current_point, max_distance=0.1):
    if prev_point is None:
        return False
    distance = np.linalg.norm(np.array(current_point) - np.array(prev_point))
    return distance > max_distance

def smooth_point(prev_point, current_point, alpha):
    if prev_point is None:
        return current_point
    smoothed_point = [alpha * c + (1 - alpha) * p for p, c in zip(prev_point, current_point)]
    return smoothed_point

def get_valid_depth_average(depth_frame, x, y, window_size=3):
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    width = depth_intrin.width
    height = depth_intrin.height
    half_window = window_size // 2
    depths = []
    for i in range(-half_window, half_window + 1):
        for j in range(-half_window, half_window + 1):
            xi = x + i
            yj = y + j
            if 0 <= xi < width and 0 <= yj < height:
                d = depth_frame.get_distance(xi, yj)
                if MIN_DEPTH <= d <= MAX_DEPTH:
                    depths.append(d)
    if depths:
        return sum(depths) / len(depths)
    else:
        return None  # No valid depth in neighborhood

if __name__ == "__main__":
    main()