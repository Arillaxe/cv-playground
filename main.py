import cv2
import time
import HandTracking
import PoseEstimation
import FaceDetection


def main():
  cap = cv2.VideoCapture(1)

  pTime = 0
  cTime = 0

  handDetector = HandTracking.HandDetector()
  poseDetector = PoseEstimation.PoseDetector()
  faceDetector = FaceDetection.FaceDectetor()

  while True:
    success, img = cap.read()

    faceBoxes = faceDetector.getBoundingBoxes(img)

    img = handDetector.findHands(img)
    img = poseDetector.findPose(img)

    if len(faceBoxes) != 0:
      for box in faceBoxes:
        cv2.rectangle(img, box, (255, 100, 100), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

if __name__ == "__main__":
  main()
