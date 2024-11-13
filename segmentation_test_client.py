import requests
from PIL import Image, ImageDraw
import io
import threading


# Path to the test image
image_path = "C://Users//Admin1//deep_wheels//sam_input_image//10000000b0350d82--2024-10-17-15-15-01--L_899s_26181.jpg"  # Replace with the path to your test image


def detect_objects(image_path):
    # Load the image and prepare for upload
    with open(image_path, "rb") as img_file:
        files = {"image": img_file}

        # Send POST request to the Flask server
        response = requests.post("http://127.0.0.1:5000/detect", files=files)

        if response.status_code == 200:
            detections = response.json()
            print("Detections:", detections)
            return detections
        else:
            print("Error:", response.status_code, response.text)
            return None


def draw_bounding_boxes(image_path, detections):
    # Open the image and draw the bounding boxes
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    for det in detections:
        bbox = det['bbox']
        confidence = det['confidence']
        class_id = det['class']

        # Draw rectangle and add label
        draw.rectangle(bbox, outline="red", width=2)
        draw.text((bbox[0], bbox[1] - 10), f"Class: {class_id}, Conf: {confidence:.2f}", fill="red")

    # Display the image
    img.show()


if __name__ == "__main__":
    # Detect objects and draw bounding boxes
    detections = detect_objects(image_path)
    if detections:
        draw_bounding_boxes(image_path, detections)
