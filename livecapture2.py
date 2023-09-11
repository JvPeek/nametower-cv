import cv2
import os
import numpy as np
import argparse
import time

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Template Matching with Configurable Threshold")
parser.add_argument("--threshold", type=float, default=0.7, help="Matching threshold (0 to 1)")
args = parser.parse_args()

# Path to the templates folder
templates_folder = "templates"

# List all JPG files in the templates folder
template_files = [f for f in os.listdir(templates_folder) if f.lower().endswith(".jpg")]

# Create a video capture object
vid = cv2.VideoCapture(0)

# Flag to indicate if we are in detection mode (spacebar is pressed)
detecting = False

# Flag to indicate if the markers should be displayed
displayMarkers = True

# Initialize variables to keep track of the best matches
best_matches = {}

# List to store the coordinates of detected regions and their corresponding names
detected_regions = []

# Flag to indicate if we are in capture mode (s key is pressed)
capturing = False

# Flag to indicate we have a new picture captured
newImage = False
# Initialize variables for capturing a region
capture_start = None
capture_end = None

def capture_region(event, x, y, flags, param):
    global capturing, capture_start, capture_end
    if capturing:
        print ("capturing region")
        if event == cv2.EVENT_LBUTTONDOWN:
            print ("capture start")
            capture_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            print ("capture end")
            newImage = True
            capturing = False
            capture_end = (x, y)

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', capture_region)

while True:
    ret, frame = vid.read()

    # Draw a capturing mode indicator
    if capturing:
        cv2.putText(frame, "Capturing Mode ON", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # If in detection mode, perform template matching
    if detecting:
        detecting = False
        best_matches = {}
        detected_regions = []
        # Iterate through the template files
        for template_file in template_files:
            template_path = os.path.join(templates_folder, template_file)

            # Read the template image
            template = cv2.imread(template_path)

            # Match the template in the frame
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

            # Get the maximum match value and its location
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Use the threshold specified via command-line argument
            if max_val >= args.threshold:
                # Get the filename (without extension) of the template
                template_name = os.path.splitext(template_file)[0]

                # Check if this match is better than the best match for this template so far
                if template_name not in best_matches or max_val > best_matches[template_name][0]:
                    best_matches[template_name] = (max_val, max_loc)
                    
                    # Store the coordinates of the detected region and its name
                    detected_regions.append((max_loc, (max_loc[0] + template.shape[1], max_loc[1] + template.shape[0]), template_name))

    if capture_start and capture_end:
        x1, y1 = capture_start
        x2, y2 = capture_end
        captured_region = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

        # Check if the captured region is not empty before saving it
        if not captured_region.size == 0:
            # Generate a unique filename (e.g., based on a timestamp)
            timestamp = str(int(time.time()))
            capture_filename = os.path.join(templates_folder, f"captured_{timestamp}.jpg")
            
            # Save the captured region as a JPG image
            cv2.imwrite(capture_filename, captured_region)
            
            print(f"Saved captured region as {capture_filename}")

            # Reset capturing mode
            capturing = False
            capture_start = None
            capture_end = None

    if displayMarkers:
        # Draw rectangles and labels for the best matches and previously detected regions
        for template_name, (max_val, max_loc) in best_matches.items():
            template = cv2.imread(os.path.join(templates_folder, f"{template_name}.jpg"))
            top_left = max_loc
            bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, template_name, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Check for keypress events
    key = cv2.waitKey(1) & 0xFF

    # Check for the 'q' key to quit
    if key == ord('q'):
        break

    # Check for the 'm' key to toggle markers
    if key == ord('m'):
        displayMarkers = not displayMarkers

    # Check for the spacebar key to toggle detection mode
    elif key == 32:  # Spacebar key
        detecting = not detecting
        # If not in detection mode, clear previous matches and regions
        if detecting:
            best_matches = {}
            detected_regions = []

    # Check for the 's' key to capture a region
    elif key == ord('s'):
        capturing = not capturing
        capture_start = None
        capture_end = None

# Release the video capture object and close all windows
vid.release()
cv2.destroyAllWindows()
