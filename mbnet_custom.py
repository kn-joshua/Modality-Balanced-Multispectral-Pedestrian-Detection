import cv2
import imutils
import argparse

# Function to convert image to thermal-like image
def convert_to_thermal_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thermal_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
    return thermal_image

# Function to detect people in an image
def detector(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    image = imutils.resize(image, width=min(400, image.shape[1]))

    # Detect pedestrians
    (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)

    # Draw rectangles around detected regions
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image

# Function to process the video
def detect(path):
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, image = cap.read()
        if ret:
            # Perform detection on RGB image
            rgb_detection = detector(image)

            # Convert to thermal and perform detection
            image = convert_to_thermal_image(image)
            thermal_detection = detector(image)

            # Show the output images
            cv2.imshow('RGB Detection', rgb_detection)
            cv2.imshow('Thermal Detection', thermal_detection)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect people in thermal and RGB videos.')
    parser.add_argument('input_video', help='Path to the input video file')
    
    args = parser.parse_args()
    detect(args.input_video)
