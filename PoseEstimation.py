import cv2
import mediapipe as mp
import time
import FPS


class PoseDetector():
  def __init__(self):
    self.mpPose = mp.solutions.pose
    self.pose = self.mpPose.Pose()
    self.mpDraw = mp.solutions.drawing_utils

  def findPose(self, img, draw = True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.pose.process(imgRGB)

    if draw:
      if self.results.pose_landmarks:
        self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
          
    return img

  def findPosition(self, img):
    lmList = []

    if self.results.pose_landmarks:
      for id, lm in enumerate(self.results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])

    return lmList

# Example usage
def main():
  cap = cv2.VideoCapture(1)

  fpsCounter = FPS.FPSCounter()
  
  detector = PoseDetector()

  while True:
    success, img = cap.read()

    img = detector.findPose(img)
    positions = detector.findPosition(img, 3)

    if len(positions) != 0:
      cv2.circle(img, (positions[14][1], positions[14][2]), 15, (255, 100, 100), cv2.FILLED)

    fps = fpsCounter.getFPS()

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

if __name__ == "__main__":
  main()
