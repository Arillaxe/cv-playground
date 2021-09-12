import cv2
import mediapipe as mp
import time
import FPS


class FaceDectetor():
  def __init__(self):
    self.mpFace = mp.solutions.face_detection
    self.face = self.mpFace.FaceDetection()
    self.mpDraw = mp.solutions.drawing_utils

  def getBoundingBoxes(self, img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.face.process(imgRGB)
    ih, iw, ic = img.shape

    boxesList = []

    if self.results.detections:
      for id, detection in enumerate(self.results.detections):
        box = detection.location_data.relative_bounding_box
        boxesList.append((int(box.xmin * iw), int(box.ymin * ih), int(box.width * iw), int(box.height * ih)))
          
    return boxesList

# Example usage
def main():
  cap = cv2.VideoCapture(1)

  fpsCounter = FPS.FPSCounter()
  
  detector = FaceDectetor()

  while True:
    success, img = cap.read()

    boxes = detector.getBoundingBoxes(img)

    if len(boxes) != 0:
      for box in boxes:
        cv2.rectangle(img, box, (255, 100, 100), 2)

    fps = fpsCounter.getFPS()

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

if __name__ == "__main__":
  main()
