import cv2
import time
import HandTracking
import PoseEstimation

def main():
  cap = cv2.VideoCapture(1)

  pTime = 0
  cTime = 0

  handDetector = HandTracking.HandDetector()
  poseDetector = PoseEstimation.PoseDetector()

  while True:
    success, img = cap.read()

    img = handDetector.findHands(img)
    img = poseDetector.findPose(img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

if __name__ == "__main__":
  main()
