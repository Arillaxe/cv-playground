import cv2
import mediapipe as mp
import time
import FPS


class HandDetector():
  def __init__(self):
    self.mpHands = mp.solutions.hands
    self.hands = self.mpHands.Hands()
    self.mpDraw = mp.solutions.drawing_utils

  def findHands(self, img, draw = True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRGB)

    if draw:
      if self.results.multi_hand_landmarks:
        for handLms in self.results.multi_hand_landmarks:
          self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
          
    return img

  def findPosition(self, img, handN = 0):
    lmList = []

    if self.results.multi_hand_landmarks:
      hand = self.results.multi_hand_landmarks[handN]

      for id, lm in enumerate(hand.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])

    return lmList

# Example usage
def main():
  cap = cv2.VideoCapture(1)

  fpsCounter = FPS.FPSCounter()
  
  detector = HandDetector()

  while True:
    success, img = cap.read()

    img = detector.findHands(img)

    fps = fpsCounter.getFPS()

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

if __name__ == "__main__":
  main()
