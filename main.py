import cv2
import numpy as np

# print("package imported")  # to check if the package is imported or not
# kernel = np.ones((5, 5),np.uint8)  # (5,5) size of the kernel which contains only ones of unsigned integer

# chapter 1

# img = cv2.imread("Resources/img.jpg")
# cv2.imshow("output", img)
# cv2.waitKey(0)


# cap = cv2.VideoCapture("Resources/bf6.gif")
# while True:
#     success, img = cap.read()
#     cv2.imshow("video", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
# cap.set(10, 1000)
# while True:
#     success, img = cap.read()
#     cv2.imshow("Video", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# img = cv2.imread("Resources/img.jpg")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.GaussianBlur(img_gray, (33, 33), 0)  # kernel value should be odd number and sigma value should be zero
# img_canny = cv2.Canny(img, 100, 100)
# img_dialation = cv2.dilate(img_canny, kernel, iterations=1)  # iteration is the width of the lines, what dialation does is that it increases the width of the lines in the image so that is any line has some break in the image then by increasing the width of the image we can overcome it
# image_eroded = cv2.erode(img_dialation, kernel, iterations=1) # erode function does opposite of the dialation function
# cv2.imshow("Eroded image", image_eroded)
# cv2.imshow("Dialation image", img_dialation)
# cv2.imshow("Image canny", img_canny)
# cv2.imshow("Gray image", img_gray)
# cv2.imshow("Colored image", img)
# cv2.imshow("Blur image", img_blur)
# cv2.waitKey(0)

# img = cv2.imread("Resources/img.jpg")
# print(img.shape)   # (399, 600, 3) -> (height, width, channel(RGB->3))
# img_resize_large = cv2.resize(img, (1000, 2000,))   # (width, height)
# img_resize_small = cv2.resize(img, (300, 200,))
# img_cropped = img[0:200, 200:400]  # [height, width]  basically image is an array of boxes or we can call it image is a matrix
# print(img_resize_large.shape)
# print(img_resize_small.shape)
# # cv2.imshow("Resize image large", img_resize_large)
# # cv2.imshow("Resize image small", img_resize_small)
# cv2.imshow("Cropped image", img_cropped)
# cv2.imshow("Image", img)
# cv2.waitKey(0)


# chapter 4

#
# img_black = np.zeros((512, 512))  # zero for black
# img_white = np.ones((512, 512))   # ones for white
# img = np.zeros((512, 512, 3), np.uint8)  # 3 for the 3 channels i.e rgb as 512*512 is a gray scaled image
# cv2.imshow("Black matrix image", img_black)
# cv2.imshow("White matrix image", img_white)
# img[:] = 255,0,0  # 255 for blue
# img[:] = 255,255,0
# img[200:300, 200:300] = 255,0,0
# cv2.line(img, (0, 0), (300, 300), (0, 255, 0), 3)  # (img , (starting point), (ending point), (color number), thickness)
# cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 3)
# cv2.rectangle(img, (0, 0), (300, 300), (0, 0, 255), 3) # origin point to diagonal point
# cv2.rectangle(img, (0, 0), (300, 300), (0, 0, 255), cv2.FILLED)  # to fill the rectangle completely
# cv2.circle(img, (400, 50), 30, (255, 255, 0), 5)  # point radius
# cv2.circle(img, (400, 50), 30, (255, 255, 0), cv2.FILLED)
# cv2.putText(img, "HELLO WORLD", (200, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 3)   # (img, text, (x->widht, y->height), font, scale, color, iteration->thickness)
# cv2.imshow("img", img)
# meme_img = cv2.imread("Resources/Harold.jpg")
# cv2.putText(meme_img, "HIDE THE PAIN", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0, 255), 2)
# cv2.imshow("meme image", meme_img)
# print(img_white.shape)
# print(img)
# cv2.waitKey(0)


# chapter 5

# img = cv2.imread("Resources/card.jpg")
# cv2.imshow("Card image", img)
# width, height = 250, 350
# pts1 = np.float32([[640, 40], [930, 40], [640, 429], [929, 439]])
# pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
# img_output = cv2.warpPerspective(img, matrix, (width, height))
# cv2.imshow("output image", img_output)
# cv2.waitKey(0)



# chapter 6

img_1 = cv2.imread("Resources/img.jpg")
img_2 = cv2.imread("Resources/Harold.jpg")
print(img_1.shape)
img_2 = cv2.resize(img_2, (600, 399))

img_hor = np.hstack((img_1, img_1))
img_hor_2 = np.hstack((img_1, img_2))
img_ver = np.vstack((img_1, img_1))
img_ver_2 = np.vstack((img_1, img_2))
# cv2.imshow("Vertical image 2", img_ver_2)
# cv2.imshow("Vertical image", img_ver)
# cv2.imshow("Horizontal image", img_hor)
# cv2.imshow("Horozontal image2", img_hor_2)

cv2.waitKey(0)

