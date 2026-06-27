import cv2

image = cv2.imread("dataset/Training/glioma/Tr-gl_1.jpg")

print("Image Shape:", image.shape)
print("Image Data Type:", image.dtype)

cv2.imshow("MRI Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()