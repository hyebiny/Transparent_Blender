import numpy as np
import cv2


h = 4048
w = 4048

# You can also easily set the number of squares per row
number= 17

# You can also easily set the colors
color_one = (0, 0, 0)
color_two = (255, 255, 255)

length_of_square = h/number
length_of_two_squares = h/number*2

# img = Image.new("RGB", (h, w), (255, 0, 0))  # create a new 15x15 image
pixels = np.ones((h,w,3))

for i in range(h):
    # for every 100 pixels out of the total 500 
    # if its the first 50 pixels
    if (i % length_of_two_squares) >= length_of_square:
        for j in range(w):
            if (j % length_of_two_squares) < length_of_square:
                pixels[i,j] = color_one
            else:
                pixels[i,j] = color_two

    # else its the second 50 pixels         
    else:
        for j in range(w):
            if (j % length_of_two_squares) >= length_of_square:
                pixels[i,j] = color_one
            else:
                pixels[i,j] = color_two

print("done")
cv2.imwrite(f"checkerboard_{h}_{w}_{number}.png", pixels)
