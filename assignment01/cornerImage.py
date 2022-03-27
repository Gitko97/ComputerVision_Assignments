import cv2
import numpy as np


class CornerImage:
    def __init__(self, image, patchSize = 100):
        self.image = image
        self.patchSize = patchSize
        print(image.shape)

    def clickCorner(self):
        mouse_pressed = False

        show_img = np.copy(self.image)
        clickedPoint = []

        def mouse_callback(event, _x, _y, flags, param):
            nonlocal show_img, mouse_pressed, clickedPoint
            if event == cv2.EVENT_LBUTTONDOWN:
                print("Mouse Down")
                mouse_pressed = True
                cv2.circle(show_img, (_x, _y), int(self.patchSize/2)
                           , (0, 255, 0), 2)
                if len(clickedPoint) <= 4:
                    clickedPoint.append((_x, _y))
                    print(clickedPoint)

            elif event == cv2.EVENT_LBUTTONUP:
                print("Mouse UP")
                mouse_pressed = False


        cv2.namedWindow('image')
        cv2.setMouseCallback('image', mouse_callback)
        height, width, channel = show_img.shape
        while True:
            k = cv2.waitKey(1)
            if k == ord('a') and not mouse_pressed:
                print("종료")
                break

            if k == ord('z') and not mouse_pressed:
                if clickedPoint:
                    clickedPoint.pop()
                    print(clickedPoint)
                    show_img = np.copy(self.image)
                    for x, y in clickedPoint:
                        cv2.circle(show_img, (x, y), int(self.patchSize/2)
                                   , (0, 255, 0), 2)

            cv2.putText(show_img, "Press 'A' to save", (0, height - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)
            cv2.putText(show_img, "'Z' to go back",
                        (0, height),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)

            cv2.imshow('image', show_img)

        self.clickPoints = clickedPoint
        cv2.destroyAllWindows()

    def getColorHistogram(self):
        color = ('b', 'g', 'r')
        histograms = []
        for x, y in self.clickPoints:
            histo_dic = {}
            height, width, _ = self.image.shape
            startX = x - int(self.patchSize/2)
            startX = startX if startX >= 0 else 0

            endX = x + int(self.patchSize/2)
            endX = endX if endX < width else width - 1

            startY = y - int(self.patchSize/2)
            startY = startY if startY >=0 else 0

            endY = y + int(self.patchSize/2)
            endY = endY if endY < height else height - 1

            print(startX, startY, endX, endY)

            cropped_img = self.image[startY: endY, startX: endX]

            cv2.imshow("crop", cropped_img)
            cv2.waitKey()
            cv2.destroyAllWindows()

            for i, col in enumerate(color):
                histr = cv2.calcHist([cropped_img], [i], None, [256], [0, 256])
                cv2.normalize(histr, histr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                histo_dic[col] = histr
            histograms.append(histo_dic)
        self.histograms = histograms

    def compareHistogram(self, hist1, hist2):
        result = sum([(i-j) ** 2 for i, j in zip(hist1, hist2)])
        return result