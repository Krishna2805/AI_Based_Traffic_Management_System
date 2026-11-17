
# ------------------------------------------------------------------------------------------
# MAIN CODE----------------------------------------------------------
import os 
import cv2 
from ultralytics import YOLO 
import numpy as np 
import pandas as pd 
from collections import Counter 
import cvzone 
from tkinter import filedialog
import tkinter as tk

# Load the YOLO model and class labels
script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, "yolov8m.pt")
model_source = weights_path if os.path.exists(weights_path) else "yolov8m.pt"
model = YOLO(model_source)

classes_path = os.path.join(script_dir, "classes.txt")
if os.path.exists(classes_path):
    with open(classes_path, "r") as my_file:
        class_list = my_file.read().split("\n")
else:
    with open("classes.txt", "r") as my_file:
        class_list = my_file.read().split("\n")


def object_detc(feed):
    outcome = model.predict(feed)
    detection_res = outcome[0].boxes.data
    df = pd.DataFrame(detection_res).astype("float")
    ob_classes = []

    for index, row in df.iterrows():
        left = int(row[0])
        top = int(row[1])
        right = int(row[2])
        bottom = int(row[3])
        id = int(row[5])
        class_of_ob = class_list[id]
        ob_classes.append(class_of_ob)

        # Draw bounding box and label on the image
        cv2.rectangle(feed, (left, top), (right, bottom), (255, 0, 255), 2)
        cvzone.putTextRect(feed, f'{class_of_ob}', (right, bottom), 1, 1)

    return ob_classes

def CountObInImg(classes_list):
    # Define the target objects to count
    target_objects = ["car", "motorcycle", "bus", "truck"]
    count_dict = {obj: 0 for obj in target_objects}

    counter = Counter(classes_list)
    
    for obj, count in counter.items():
        if obj in target_objects:
            count_dict[obj] = count
    
    # Print the count of target objects
    print("Object count (Car, Motorcycle, Bus, Truck):")
    for obj in target_objects:
        print(f"{obj}: {count_dict[obj]}")
    
    # Return the dictionary of counts for specific objects
    return count_dict

def CalculateTotalPriority(count_dict):
    # Define priority values for each vehicle type
    priority_values = {"truck": 4, "bus": 3, "car": 2, "motorcycle": 1}
    total_priority = 0
    
    # Calculate the total priority based on the count and priority of each vehicle
    for vehicle, count in count_dict.items():
        priority = priority_values.get(vehicle, 0)
        total_priority += count * priority

    print(f"Total priority for the image: {total_priority}")
    return total_priority

# Initialize tkinter for file dialog
root = tk.Tk()
root.withdraw()

# Store results for each image
total_priorities_per_image = []

# Select and process up to 4 images
for i in range(4):
    file = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp"),
                  ("All files", "*.*")]
    )

    if file:
        print(f"\nProcessing: {file}")
        feed = cv2.imread(file)
        if feed is None:
            print("Error loading image!")
            continue
        
        feed = cv2.resize(feed, (1020, 500))
        ob_classes = object_detc(feed)
        
        # Count the specific objects and calculate total priority
        count_dict = CountObInImg(ob_classes)
        total_priority = CalculateTotalPriority(count_dict)
        total_priorities_per_image.append(total_priority)
        
        # Display the image with detected objects in a separate window for each
        window_title = f"Image {i+1}"
        cv2.imshow(window_title, feed)

    print("\nProcessing complete!")

print("All selected images processed.")

# Print the total priorities for each image
for i, priority in enumerate(total_priorities_per_image):
    print(f"Total priority for Image {i+1}: {priority}")

# Keep all windows open until closed manually
while True:
    if cv2.waitKey(0) & 0xFF == ord('q'):  # Close all windows if 'q' is pressed 
        break 
 
cv2.destroyAllWindows()



