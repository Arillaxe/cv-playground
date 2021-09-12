import time


class FPSCounter():
  def __init__(self):
    self.pTime = 0
    self.cTime = 0

  def getFPS(self):
    self.cTime = time.time()
    fps = 1 / (self.cTime - self.pTime)
    self.pTime = self.cTime

    return fps

# Example usage
def main():
  fpsCounter = FPSCounter()

  while True:
    print(fpsCounter.getFPS())

if __name__ == "__main__":
  main()
