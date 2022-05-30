import cv2
from apps import HandGesture

hg_start = HandGesture()

if __name__ == '__main__':
    hg_start.openCamera()

    while hg_start.cameraIsOpen:
        image = hg_start.readImage()

        if not hg_start.isReadImageSuccess:
            print("ignore empty frame")
            continue

        image = hg_start.trackGesture(image)
        hg_start.drawLandmark(image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    hg_start.camera.release()