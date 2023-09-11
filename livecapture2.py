import cv2
import os
import numpy as np
import argparse

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

while True:
    ret, frame = vid.read()



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

    if displayMarkers:
        # Draw rectangles and labels for the best matches and previously detected regions
        for template_name, (max_val, max_loc) in best_matches.items():
            template = cv2.imread(os.path.join(templates_folder, f"{template_name}.jpg"))
            top_left = max_loc
            bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, template_name, (top_left[0], top_left[1] -10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

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
# Release the video capture object and close all windows
vid.release()
cv2.destroyAllWindows()
