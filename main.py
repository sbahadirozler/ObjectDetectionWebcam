import cv2
import numpy as np
import requests
from detecto.core import Model

# Replace this with the URL of your IP webcam stream
stream_url = "http://192.168.0.11:8080//shot.jpg"

# Create a detecto model
model = Model()

# Set the threshold
threshold = 0.85

# Start a loop to capture and process frames from the IP webcam stream
while True:
    # Capture a frame from the IP webcam stream
    img_resp = requests.get(stream_url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    # Detect objects in the frame
    predictions = model.predict(img)

    # Filter the predictions manually based on the threshold
    filtered_predictions = ([], [], [])
    for i in range(len(predictions[0])):
        label, bbox, score = predictions[0][i], predictions[1][i], predictions[2][i]
        if score >= threshold:
            filtered_predictions[0].append(label)
            filtered_predictions[1].append(bbox)
            filtered_predictions[2].append(score)

    # Draw bounding boxes around the detected objects
    for i in range(len(filtered_predictions[0])):
        label, bbox, score = filtered_predictions[0][i], filtered_predictions[1][i], filtered_predictions[2][i]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 3)
        cv2.putText(img, label, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("window", img)

    # Press the 'q' key to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Close all windows
cv2.destroyAllWindows()
