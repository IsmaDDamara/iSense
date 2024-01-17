import cv2
import time
import json
import argparse
import paho.mqtt.client as mqtt
from opencv_utils import draw_text, load_classes, load_colors, draw_detection

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv4 Tiny Crowd Detection")
    parser.add_argument("--weights", type=str, default="E:\\MAGANG\\iSense\\weights\\yolov4-tiny-416x416_best.weights", help="Path to YOLO weights file")
    parser.add_argument("--cfg", type=str, default="E:\\MAGANG\\iSense\\cfg\\yolov4-tiny-416x416.cfg", help="Path to YOLO configuration file")
    parser.add_argument("--classes", type=str, default="E:\\MAGANG\\iSense\\data\\obj.names", help="Path to YOLO classes file")
    parser.add_argument("--video", type=str, default="E:\\MAGANG\\iSense\\Test\\video\\video_test2.mp4", help="Path to video file")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load YOLOv4 Tiny model
    net = cv2.dnn.readNet(args.weights, args.cfg)

    # Load classes
    classes = load_classes(args.classes)

    # Load class colors
    class_colors = load_colors(classes)

    # Open video file
    cap = cv2.VideoCapture(args.video)

    # Get original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set the desired display resolution
    display_width = 1280
    display_height = 720

    # Define the codec and create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can change the codec as needed
    output_video_path = "output_video.mp4"  # Set your desired output video file path
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (display_width, display_height))  # Adjust the frame rate if needed
    
    # Inisialisasi MQTT Client
    client = mqtt.Client()
    client.connect("broker_address", 1883, 60)  

    frame_count = 0
    start_time = time.time()

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop if no more frames are available

        # Resize the frame to the desired display resolution
        resized_frame = cv2.resize(frame, (display_width, display_height))

        # Prepare the resized frame for inference
        blob = cv2.dnn.blobFromImage(resized_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Get output layer names
        output_layer_names = net.getUnconnectedOutLayersNames()

        # Perform inference
        detections = net.forward(output_layer_names)

        # Process and display the detections
        resized_frame, num_person_detections, num_head_detections = draw_detection(resized_frame, classes, class_colors, detections, confidence_threshold=0.5)

        # Calculate and display fps
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # Display the result with bounding box count and fps
        person_label = f'Person: {num_person_detections}'
        head_label = f'Head: {num_head_detections}'
        fps_label = f'FPS: {fps:.2f}'

        # Draw hologram effect for the "Person" text
        draw_text(resized_frame, person_label, (10, 30), (0, 255, 0))

        # Draw hologram effect for the "Head" text
        draw_text(resized_frame, head_label, (10, 70), (0, 255, 0))

        # Draw hologram effect for the "FPS" text
        draw_text(resized_frame, fps_label, (10, 110), (0, 255, 0))

        # Show the resized frame with normal window properties
        cv2.imshow("YOLOv4 Tiny Crowd Detection", resized_frame)

        # Write the frame to the output video file
        out.write(resized_frame)

        # Kirim data JSON melalui MQTT
        data_json = {"jumlah_orang": num_person_detections}
        json_data = json.dumps(data_json)
        client.publish("topic", json_data)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Tutup koneksi MQTT
    client.disconnect()

    # Calculate and display the average FPS
    average_fps = frame_count / (time.time() - start_time)
    print(f"Average FPS: {average_fps}")

    # Release the video capture object, video writer, and close the window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
