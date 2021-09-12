import cv2
import mediapipe as mp
import FPS


class FaceMeshDetector():
  def __init__(self):
    self.mpFaceMesh = mp.solutions.face_mesh
    self.faceMesh = self.mpFaceMesh.FaceMesh()
    self.mpDraw = mp.solutions.drawing_utils
    self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

  def findFaces(self, img, draw = True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.faceMesh.process(imgRGB)

    if draw:
      if self.results.multi_face_landmarks:
        for faceLms in self.results.multi_face_landmarks:
          self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
          
    return img

# Example usage
def main():
  cap = cv2.VideoCapture(1)

  fpsCounter = FPS.FPSCounter()
  
  detector = FaceMeshDetector()

  while True:
    success, img = cap.read()

    img = detector.findFaces(img)

    fps = fpsCounter.getFPS()

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

if __name__ == "__main__":
  main()
