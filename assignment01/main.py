import cv2
from cornerImage import CornerImage
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    image1 = cv2.imread("1st.jpg")
    image2 = cv2.imread("2nd.jpg")

    blurImage1 = cv2.GaussianBlur(image1, (3, 3), 0)
    blurImage2 = cv2.GaussianBlur(image2, (3, 3), 0)

    blurImage1 = cv2.resize(blurImage1, (0, 0), fx=0.3, fy=0.3)
    blurImage2 = cv2.resize(blurImage2, (0, 0), fx=0.3, fy=0.3)

    patchSize = 9

    cornerImage1 = CornerImage(blurImage1, patchSize)
    cornerImage2 = CornerImage(blurImage2, patchSize)

    cornerImage1.clickCorner()
    cornerImage2.clickCorner()

    cornerImage1.getColorHistogram()
    cornerImage2.getColorHistogram()

    color = ('b', 'g', 'r')
    for i, histogram in enumerate(cornerImage1.histograms):
        for col in color:
            plt.subplot(2, 4, i + 1)
            plt.title("point" + str(i % 4))
            plt.plot(histogram[col], color=col)
            plt.xlim([0, 256])

    for i, histogram in enumerate(cornerImage2.histograms):
        for col in color:
            plt.subplot(2, 4, i + 5)
            plt.title("point" + str(i % 4))
            plt.plot(histogram[col], color=col)
            plt.xlim([0, 256])

    plt.show()

    histoDiffResult = []
    for i, hist1 in enumerate(cornerImage1.histograms):
        for j, hist2 in enumerate(cornerImage2.histograms):
            diff = 0
            for col in color:
                diff = diff + cornerImage1.compareHistogram(hist1[col], hist2[col])
            histoDiffResult.append((i, j, diff))
    histoDiffResult.sort(key=lambda result: result[2])


    duplicate1 = []
    duplicate2 = []
    matched = []
    for i, j, _ in histoDiffResult:
        if i not in duplicate1:
            if j not in duplicate2:
                duplicate1.append(i)
                duplicate2.append(j)
                matched.append((i, j))

    newImage = np.concatenate((cornerImage1.image, cornerImage2.image), axis=1)
    for i, j in matched:
        width = blurImage1.shape[1]
        newPoint = (cornerImage2.clickPoints[j][0] + width , cornerImage2.clickPoints[j][1])
        cv2.line(newImage, cornerImage1.clickPoints[i], newPoint , (0, 0, 255), 10)
    cv2.imshow("Result", newImage)
    cv2.waitKey()
    cv2.destroyAllWindows()
