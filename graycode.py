import numpy as np
import cv2


h = 1024
w = 1024

# You can also easily set the number of squares per row
number= 22

# You can also easily set the colors
color_one = (0, 0, 0)
color_two = (255, 255, 255)

length_of_square = h/number
length_of_two_squares = h/number*2

# img = Image.new("RGB", (h, w), (255, 0, 0))  # create a new 15x15 image
pixels = np.ones((h,w,3))

for num in range(int(number/2)):
    length_of_square_h = h/(2**num)
    length_of_square_w = w/(2**num)

    pixels = np.ones((h,w,3))
    for i in range(h):
        # for every 100 pixels out of the total 500 
        # if its the first 50 pixels
        if (((i-(i%length_of_square_h))/length_of_square_h)) %2 == 0:
            pixels[i,:] = color_one
        else:
            pixels[i,:] = color_two

    cv2.imwrite(f"graycode_{h}_{w}_{0}_{num}.png", pixels)
        

    pixels = np.ones((h,w,3))
    for j in range(w):
        if (((j-(j%length_of_square_w))/length_of_square_w)) %2  == 0:
            pixels[:,j] = color_one
        else:
            pixels[:,j] = color_two

    cv2.imwrite(f"graycode_{h}_{w}_{1}_{num}.png", pixels)

print("done")
