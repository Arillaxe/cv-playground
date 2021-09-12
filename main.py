import cv2
import FPS
import HandTracking
import PoseEstimation
import FaceDetection
import FaceMesh


def main():
  cap = cv2.VideoCapture(1)

  fpsCounter = FPS.FPSCounter()

  handDetector = HandTracking.HandDetector()
  poseDetector = PoseEstimation.PoseDetector()
  faceDetector = FaceDetection.FaceDectetor()
  faceMeshDetector = FaceMesh.FaceMeshDetector()

  while True:
    success, img = cap.read()

    faceBoxes = faceDetector.getBoundingBoxes(img)

    img = handDetector.findHands(img)
    img = poseDetector.findPose(img)
    img = faceMeshDetector.findFaces(img)

    if len(faceBoxes) != 0:
      for box in faceBoxes:
        cv2.rectangle(img, box, (255, 100, 100), 2)

    fps = fpsCounter.getFPS()

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

if __name__ == "__main__":
  main()
