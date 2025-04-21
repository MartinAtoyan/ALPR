import cv2
import matplotlib.pyplot as plt

cube = cv2.imread("cube.jpg")

print(cube.shape)
print(cube.size)
print(cube.dtype)

# plt.imshow(cube, cmap="gray")
# plt.title("cube")
# plt.show()


reserved_channels = cube[:, :, ::-1]
# print(reserved_channels)

# b, g, r = cv2.split(cube)
# plt.figure(figsize=(20, 5))
# plt.subplot(141);plt.imshow(r, cmap="gray");plt.title("RED channel")
# plt.subplot(142);plt.imshow(g, cmap="gray");plt.title("GREEN channel")
# plt.subplot(143);plt.imshow(b, cmap="gray");plt.title("BLUE channel")
# merged = cv2.merge((b, g, r))
# plt.subplot(144);plt.imshow(merged[:, :, ::-1]);plt.title("Merged")
# plt.show()

cv2.destroyAllWindows()

hsv_image = cv2.cvtColor(cube, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_image)
plt.figure(figsize=(20, 5))
plt.subplot(141);plt.imshow(h, cmap="gray");plt.title("H channel")
plt.subplot(142);plt.imshow(s, cmap="gray");plt.title("S channel")
plt.subplot(143);plt.imshow(v, cmap="gray");plt.title("V channel")
merged_2 = cv2.merge((h, s, v))
plt.subplot(144);plt.imshow(merged_2[:, :, ::-1]);plt.title("Merged")
plt.show()


