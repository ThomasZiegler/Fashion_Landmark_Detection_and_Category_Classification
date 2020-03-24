import cv2
import numpy as np
import matplotlib.pyplot as plt

print('start')

#image = cv2.imread('./src/checker.jpg')
image = cv2.imread('./src/jeans.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224,224))







# define constants
img_height, img_width = image.shape[:2]
alpha = 600
sigma = 40
kernel_size = 2*int(4*sigma)+1



# define random generator
random_state = np.random.RandomState(17)

#dx = (random_state.rand(img_height, img_width)*2-1)
#dx_smooth = alpha * cv2.GaussianBlur(dx,
#                                    (kernel_size, kernel_size),
#                                    sigmaX=sigma,
#                                    sigmaY=sigma,
#                                    borderType=cv2.BORDER_CONSTANT)
#
#dy = (random_state.rand(img_height, img_width)*2-1)
#dy_smooth = alpha * cv2.GaussianBlur(dy,
#                                     (kernel_size, kernel_size),
#                                     sigmaX=sigma,
#                                     sigmaY=sigma,
#                                     borderType=cv2.BORDER_CONSTANT)
#
#
#dx_smooth = np.outer(np.arange(img_height),np.arange(img_width))
#dy_smooth = np.outer(np.arange(img_height),np.arange(img_width))

for k in range(15):
    dx = np.zeros((img_height, img_width))
    dy = np.zeros((img_height, img_width))

    for i in range(3):
        x_idx = int(random_state.rand(1)*img_width)
        y_idx = int(random_state.rand(1)*img_height)
        x_value = alpha*(random_state.rand(1)*2-1)
        y_value = alpha*(random_state.rand(1)*2-1)

        dx[y_idx, x_idx] = x_value
        dy[y_idx, x_idx] = y_value



    dx_smooth = alpha * cv2.GaussianBlur(dx,
                                        (kernel_size, kernel_size),
                                        sigmaX=sigma,
                                        sigmaY=sigma,
                                        borderType=cv2.BORDER_CONSTANT)

    dy_smooth = alpha * cv2.GaussianBlur(dy,
                                         (kernel_size, kernel_size),
                                         sigmaX=sigma,
                                         sigmaY=sigma,
                                         borderType=cv2.BORDER_CONSTANT)



    #fig = plt.figure(1)
    ## x vector field
    #ax1 = fig.add_subplot(221)
    #ax1.imshow(dx)
    #ax2 = fig.add_subplot(222)
    #ax2.imshow(dx_smooth)
    #
    ## y vector field
    #ax3 = fig.add_subplot(223)
    #ax3.imshow(dy)
    #ax4 = fig.add_subplot(224)
    #ax4.imshow(dy_smooth)


    # create coordinate maps src_coordinate + delta_coordinate
    x, y = np.meshgrid(np.arange(img_width), np.arange(img_height))
    map_x = np.float32(x+dx_smooth)
    map_y = np.float32(y+dy_smooth)

    #fig = plt.figure(2)
    #ax1 = fig.add_subplot(121)
    #ax1.imshow(map_x)
    #ax2 = fig.add_subplot(122)
    #ax2.imshow(map_y)
    #
    #fig = plt.figure(3)
    #plt.quiver(x, y, dx_smooth, dy_smooth)


    new_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    fig = plt.figure(4)
    ax1 = fig.add_subplot(121)
    ax1.imshow(image)
    ax2 = fig.add_subplot(122)
    ax2.imshow(new_image)

    #print('===================================')
    #print('===================================')
    #print(map_x.shape)
    #print('===================================')
    #print('===================================')
    plt.show(block=False)
    plt.pause(1)
    plt.close('all')
