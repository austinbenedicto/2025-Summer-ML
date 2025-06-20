import cv2

image = cv2.imread('MainBefore.jpg')

cv2.imshow('Original Image', image)
cv2.waitKey(0)


height, width, _ = image.shape

"""
print("RGB values of each pixel:")
for y in range(height):
    for x in range(width):
        b, g, r = image[y, x]
        print(f'Pixel at ({x}, {y}): R={r}, G={g}, B={b}')
 """

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)


cv2.destroyAllWindows()