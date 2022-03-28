import cv2
from cornerImage import CornerImage
import gradient
import numpy as np

from matplotlib import pyplot as plt
def get_hog_list_with_cornerImage(cornerImage):
    image1_hog_list = []
    for image1_points in cornerImage.clickPoints:
        grayScale = cv2.cvtColor(cornerImage.getCropImage(image1_points), cv2.COLOR_BGR2GRAY)
        normalized_image = gradient.normalize_image(grayScale)

        ####
        gradX = gradient.compute_derivative_x(normalized_image)
        gradY = gradient.compute_derivative_y(normalized_image)

        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].set_title('GrayScale')
        axes[0].imshow(grayScale, cmap='gray')
        axes[1].set_title('Gradient X')
        axes[1].imshow(gradient.normalize_image(gradX), cmap='gray')
        axes[2].set_title('Gradient Y')
        axes[2].imshow(gradient.normalize_image(gradY), cmap='gray')
        plt.show()
        ####

        magnitude_result, orientation_result = gradient.get_gradient_magnitude_orientation(grayScale)
        magnitude_result = np.ravel(magnitude_result).astype(np.int32)
        orientation_result = np.ravel(orientation_result)

        hog = [0] * 9
        for magnitudeVal, orientationVal in zip(magnitude_result, orientation_result):
            quotient, remainder = divmod(orientationVal, 20)

            if remainder is 0:
                hog[quotient] = hog[quotient] + magnitudeVal
                continue
            val1 = magnitudeVal * (remainder / 20)
            val2 = magnitudeVal - val1

            if remainder <= 10:
                hog[quotient] = hog[quotient] + val2
                hog[quotient + 1 if quotient + 1 <= 8 else 0] = hog[quotient] + val1
            else:
                hog[quotient] = hog[quotient] + val1
                hog[quotient + 1 if quotient + 1 <= 8 else 0] = hog[quotient] + val2
        ###
        image1_hog_list.append(hog)
    return image1_hog_list

def test_gradient():
    img = np.array([[255, 0,0,0, 255],
                    [255, 0,0,0, 255],
                    [255, 0,0,0, 255]]
                   )
    gradX = gradient.compute_derivative_x(img)
    gradY = gradient.compute_derivative_y(img)
    magnitude_result, orientation_result = gradient.get_gradient_magnitude_orientation(img)

    print("**" * 20)
    print(gradX)
    print(gradY)
    print("==")
    print(magnitude_result)
    print(orientation_result)

    magnitude_result = np.ravel(magnitude_result).astype(np.int32)
    orientation_result = np.ravel(orientation_result)

    hog = [0] * 9
    for magnitudeVal, orientationVal in zip(magnitude_result, orientation_result):
        quotient, remainder = divmod(orientationVal, 20)
        print("Current magnitudeVal : ", magnitudeVal)
        print("Current orientationVal : ", orientationVal)
        print("quotient : ", quotient)
        print("remainder : ", remainder)

        if remainder is 0:
            hog[quotient] = hog[quotient] + magnitudeVal
            continue
        val1 = magnitudeVal * (remainder / 20)
        val2 = magnitudeVal - val1

        nextIndex = quotient + 1 if quotient + 1 <= 8 else 0
        if remainder <= 10:
            hog[quotient] = hog[quotient] + val2
            hog[nextIndex] = hog[nextIndex] + val1
        else:
            hog[quotient] = hog[quotient] + val1
            hog[nextIndex] = hog[nextIndex] + val2
        print("Val1 : ", val1," Val2 : ", val2)
        print("HOG : ", hog)
    ###
    print(hog)
if __name__ == '__main__':

    mode = 0 # 0 is total 1 is only color 2 is only gradient
    image1 = cv2.imread("1st.jpg")
    image2 = cv2.imread("2nd.jpg")

    blurImage1 = cv2.GaussianBlur(image1, (3, 3), 0)
    blurImage2 = cv2.GaussianBlur(image2, (3, 3), 0)

    blurImage1 = cv2.resize(blurImage1, (0, 0), fx=0.3, fy=0.3)
    blurImage2 = cv2.resize(blurImage2, (0, 0), fx=0.3, fy=0.3)

    patchSize = 51

    cornerImage1 = CornerImage(blurImage1, patchSize)
    cornerImage2 = CornerImage(blurImage2, patchSize)

    cornerImage1.clickPoints = [(410, 225), (42, 574), (527, 1080), (889, 725)]
    cornerImage2.clickPoints = [(765, 337), (235, 339), (201, 1064), (761, 1095)]

    # cornerImage1.clickCorner()
    # cornerImage2.clickCorner()

    ###
    test_gradient()
    ###
    cornerImage1.getColorHistogram()
    cornerImage2.getColorHistogram()

    color = ('b', 'g', 'r')
    newImage = np.concatenate((cornerImage1.image, cornerImage2.image), axis=1)
    for i, (point1, point2) in enumerate(zip(cornerImage1.clickPoints, cornerImage2.clickPoints)):
        width = cornerImage1.image.shape[1]
        newPoint = (point2[0] + width, point2[1])

        x, _x, y, _y = cornerImage1.get_rectangle_point(point1)
        cv2.rectangle(newImage, (x, y), (_x, _y), (0, 255, 0), 5)
        cv2.putText(newImage, str(i), point1,
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

        x, _x, y, _y = cornerImage2.get_rectangle_point(point2)
        cv2.rectangle(newImage, (x + width, y), (_x + width, _y), (0, 255, 0), 5)
        cv2.putText(newImage, str(i), newPoint,
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
    cv2.imshow("Patch Image", newImage)

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
    plt.suptitle("Color Histogram")
    plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()

    image1_hog = get_hog_list_with_cornerImage(cornerImage1)
    image2_hog = get_hog_list_with_cornerImage(cornerImage2)

    hogHistDiffResult = []
    hogHistMax = 0.
    hogHistMin = float('inf')
    for i, hist1 in enumerate(image1_hog):
        for j, hist2 in enumerate(image2_hog):
            diff = cornerImage1.compareHistogram(hist1, hist2)
            if hogHistMax < diff:
                hogHistMax = diff
            if hogHistMin > diff:
                hogHistMin = diff
            hogHistDiffResult.append((i, j, diff))

    #
    colorHistoDiffResult = []
    colorHistMax = 0.
    colorHistMin = float('inf')
    for i, hist1 in enumerate(cornerImage1.histograms):
        for j, hist2 in enumerate(cornerImage2.histograms):
            diff = 0
            for col in color:
                diff = diff + cornerImage1.compareHistogram(hist1[col], hist2[col])
            if colorHistMax < diff:
                colorHistMax = diff
            if colorHistMin > diff:
                colorHistMin = diff
            colorHistoDiffResult.append((i, j, diff))

    duplicate1 = []
    duplicate2 = []
    matched = []
    totalDiff = []
    for hogDiff, colorDiff in zip(hogHistDiffResult, colorHistoDiffResult):
        print(hogDiff," / ", colorDiff)
        normalizeHogVal = (hogDiff[2] - hogHistMin) / (hogHistMax - hogHistMin)
        normalizeColorVal = (colorDiff[2] - colorHistMin) / (colorHistMax - colorHistMin)
        if mode == 1:
            normalizeHogVal = 0.0
        if mode == 2:
            normalizeColorVal = 0.0
        totalDiff.append((hogDiff[0], hogDiff[1], normalizeHogVal+normalizeColorVal[0]))
    totalDiff.sort(key=lambda totalDiff: totalDiff[2])
    print("Total Difference",totalDiff)

    for i, j, _ in totalDiff:
        if i not in duplicate1:
            if j not in duplicate2:
                duplicate1.append(i)
                duplicate2.append(j)
                matched.append((i, j))

    newImage = np.concatenate((cornerImage1.image, cornerImage2.image), axis=1)
    for i, j in matched:
        width = blurImage1.shape[1]
        newPoint = (cornerImage2.clickPoints[j][0] + width , cornerImage2.clickPoints[j][1])

        x, _x, y, _y = cornerImage1.get_rectangle_point(cornerImage1.clickPoints[i])
        cv2.rectangle(newImage, (x, y), (_x, _y), (0, 255, 0), 5)
        cv2.putText(newImage, str(i), cornerImage1.clickPoints[i],
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

        x, _x, y, _y = cornerImage2.get_rectangle_point(cornerImage2.clickPoints[j])
        cv2.rectangle(newImage, (x + width, y), (_x + width, _y), (0, 255, 0), 5)
        cv2.putText(newImage, str(j), newPoint,
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

        cv2.line(newImage, cornerImage1.clickPoints[i], newPoint , (0, 0, 255), 10)
    cv2.imshow("Result", newImage)
    cv2.waitKey()
    cv2.destroyAllWindows()