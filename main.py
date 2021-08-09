from kivy.graphics.texture import Texture
from kivymd.app import MDApp
from kivy.uix.image import Image
from kivy.clock import Clock
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.core.window import Window
import cv2
import mediapipe as mp
import numpy as np
import threading

Window.clearcolor = (1, 1, 1, 1)
Window.size = (600, 360)


class HandGestureCameraApp(MDApp):
    def build(self):
        self.layout = MDBoxLayout(orientation='vertical')
        self.image = Image()


        self.capture = cv2.VideoCapture(0)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.tipIds = [4, 8, 12, 16, 20]
        self.xp, self.yp = 0, 0
        self.imgCanvas = np.zeros((480, 640, 3), np.uint8)
        self.results = None

        thread = threading.Thread(target=self.load_video)
        thread.daemon = True
        thread.start()



        #Clock.schedule_interval(self.bufferImg, 1.0 / 60.0)

        return self.layout



    def load_video(self, *args):
        with self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
            while True:
                ret, frame = self.capture.read()

                if ret:
                    self.rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.rgbImage = cv2.flip(self.rgbImage, 1)
                    self.rgbImage.flags.writeable = False
                    self.results = hands.process(self.rgbImage)
                    self.rgbImage.flags.writeable = True
                    lmList = self.findPosition(self.rgbImage)

                    if self.results.multi_hand_landmarks:

                        for num, hand in enumerate(self.results.multi_hand_landmarks):
                            self.mp_drawing.draw_landmarks(self.rgbImage, hand, self.mp_hands.HAND_CONNECTIONS,
                                                           self.mp_drawing.DrawingSpec(color=(56, 58, 89), thickness=2,
                                                                                       circle_radius=4),
                                                           self.mp_drawing.DrawingSpec(color=(250, 44, 250),
                                                                                       thickness=2,
                                                                                       circle_radius=2),
                                                           )

                    if len(lmList) != 0:
                        self.x1, self.y1 = lmList[8][1:]
                        self.x2, self.y2 = lmList[12][1:]
                        finger = self.fingerUp()

                        if finger[0] == True and finger[1] == True and finger[2] == True and finger[3] == True and \
                                finger[4] == True:
                            if self.xp == 0 and self.yp == 0:
                                self.xp, self.yp = self.x1, self.y1

                            cv2.line(self.imgCanvas, (self.xp, self.yp), (self.x1, self.y1), (0, 0, 0), 50)
                            cv2.circle(self.rgbImage, (self.x2, self.y2), 35, (0, 0, 0), cv2.FILLED)

                        elif finger[1] == True and finger[2] == True and finger[0] == False and finger[3] == False and \
                                finger[4] == False:
                            self.xp, self.yp = 0, 0

                        elif finger[1] == True and finger[2] == False:
                            if self.xp == 0 and self.yp == 0:
                                self.xp, self.yp = self.x1, self.y1
                            cv2.line(self.rgbImage, (self.xp, self.yp), (self.x1, self.y1), (30, 144, 255), 15)
                            cv2.line(self.imgCanvas, (self.xp, self.yp), (self.x1, self.y1), (30, 144, 255), 15)
                        self.xp, self.yp = self.x1, self.y1

                    imgGrey = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
                    _, imgInv = cv2.threshold(imgGrey, 127, 255, cv2.THRESH_BINARY_INV)
                    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
                    self.rgbImage = cv2.bitwise_and(self.rgbImage, imgInv)
                    self.rgbImage = cv2.bitwise_or(self.rgbImage, self.imgCanvas)

                buffer = cv2.flip(self.rgbImage, 0).tobytes()
                texture = Texture.create(size=(self.rgbImage.shape[1], self.rgbImage.shape[0]))
                texture.blit_buffer(buffer, bufferfmt='ubyte')
                self.image.texture = texture


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.capture.release()
        self.layout.add_widget(self.image)



    def findPosition(self, img, handNo=0):
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                height, width, cho = img.shape

                cx, cy = int(lm.x * width), int(lm.y * height)
                self.lmList.append([id, cx, cy])

        return self.lmList

    def fingerUp(self):
        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


if __name__ == "__main__":
    HandGestureCameraApp().run()
    cv2.destroyAllWindows()
