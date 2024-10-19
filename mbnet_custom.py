import cv2
import imutils
import argparse

def convert_to_thermal_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thermal_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
    return thermal_image

def detector(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    image = imutils.resize(image, width=min(400, image.shape[1]))
    (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image

def detect(path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, image = cap.read()
        if ret:
            rgb_detection = detector(image)
            image = convert_to_thermal_image(image)
            thermal_detection = detector(image)
            cv2.imshow('RGB Detection', rgb_detection)
            cv2.imshow('Thermal Detection', thermal_detection)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect people in thermal and RGB videos.')
    parser.add_argument('input_video')
    parser.add_argument('extra_argument')
    args = parser.parse_args()
    detect(args.input_video)
