import cv2
import numpy as np
import matplotlib.pyplot as plt

def check_black_percentage(image):
    total_pixels = image.shape[0] * image.shape[1]  
    black_pixels = np.sum(np.all(image == [0, 0, 0], axis=-1))  
    black_percentage = (black_pixels / total_pixels) * 100  
    return black_percentage
def highly_black(image,x):
    black_percentage = check_black_percentage(image)
    if black_percentage > 99.999999:
        x=x+1
        return x
    else:
        x=0
        return x

def red(image):
    num_filtered_contours_rose = 0
    x = 0
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([160, 100, 100])
    upper_red = np.array([180, 255, 255])
    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    result_image_rgb_red = cv2.bitwise_and(image, image, mask=mask)
    x = highly_black(result_image_rgb_red,x)
    if x!=0:
        return num_filtered_contours_rose
    cv2.imshow('Only Red Regions Image', result_image_rgb_red)
    cv2.imwrite('Only_Red_Regions_Image.png', result_image_rgb_red)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    result_image_red_or_black = np.zeros_like(result_image_rgb_red)
    result_image_red_or_black[np.where(mask != 0)] = [255, 255, 255]  
    x = highly_black(result_image_red_or_black,x)
    if x!=0:
        return num_filtered_contours_rose
    cv2.imshow('Converted Image Black & White', result_image_red_or_black)
    cv2.imwrite('Converted_Image_Black_&_White.png', result_image_red_or_black)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    gray_image = cv2.cvtColor(result_image_red_or_black, cv2.COLOR_RGB2GRAY)
    canny_edges = cv2.Canny(gray_image, 100, 200)
    cv2.imshow('Canny Edges Detected', canny_edges)
    cv2.imwrite('Canny_Edges_Detected.png', canny_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_image = result_image_rgb_red.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours From Canny Edge', contours_image)
    cv2.imwrite('Contours_From_Canny_Edge.png', contours_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= 19]
    contours_image_1 = result_image_rgb_red.copy()
    cv2.drawContours(contours_image_1, filtered_contours, -1, (0, 255, 0), 2)
    num_filtered_contours_rose = len(filtered_contours)
    for i, contour in enumerate(filtered_contours):
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(contours_image_1, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    plt.figure(figsize=(15, 10))
    cv2.imshow('Filtered Contours Image', contours_image_1)
    cv2.imwrite('Filtered_Contours_Image.png', contours_image_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title('Input Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Only Red Regions Image')
    plt.imshow(result_image_rgb_red)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Converted Image Black & White')
    plt.imshow(result_image_red_or_black)
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Canny Edges Detected')
    plt.imshow(canny_edges)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Contours From Canny Edge')
    plt.imshow(contours_image)
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('Filtered Contours Image Red Rose')
    plt.imshow(contours_image_1)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('red_image_results.png')
    plt.show()
    return num_filtered_contours_rose, contours_image_1

def yellow(image):
    num_filtered_contours_marigold = 0
    x = 0
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([0, 100, 100])  
    upper_yellow = np.array([30, 255, 255])  
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    result_image_rgb_yellow = cv2.bitwise_and(image, image, mask=mask)
    x = highly_black(result_image_rgb_yellow,x)
    if x!=0:
        return num_filtered_contours_marigold
    cv2.imshow('Only Yellow Regions Image', result_image_rgb_yellow)
    cv2.imwrite('yellow1.png', result_image_rgb_yellow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    result_image_yellow_or_black = np.zeros_like(result_image_rgb_yellow)
    result_image_yellow_or_black[np.where(mask != 0)] = [255, 255, 255]  
    x = highly_black(result_image_rgb_yellow,x)
    if x!=0:
        return num_filtered_contours_marigold
    cv2.imshow('Converted Image Black & White', result_image_yellow_or_black)
    cv2.imwrite('yellow2.png', result_image_yellow_or_black)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    gray_image = cv2.cvtColor(result_image_yellow_or_black, cv2.COLOR_RGB2GRAY)
    canny_edges = cv2.Canny(gray_image, 100, 200)
    cv2.imshow('Canny Edges Detected', canny_edges)
    cv2.imwrite('yellow3.png', canny_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_image = result_image_rgb_yellow.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours From Canny Edge', contours_image)
    cv2.imwrite('yellow4.png', contours_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= 19]
    contours_image_2 = result_image_rgb_yellow.copy()
    cv2.drawContours(contours_image_2, filtered_contours, -1, (0, 255, 0), 2)
    num_filtered_contours_marigold = len(filtered_contours)
    for i, contour in enumerate(filtered_contours):
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(contours_image_2, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    plt.figure(figsize=(15, 10))
    cv2.imshow('Filtered Contours Image', contours_image_2)
    cv2.imwrite('yellow5.png', contours_image_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title('Input Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Only Yellow Regions Image')
    plt.imshow(result_image_rgb_yellow)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Converted Image Black & White')
    plt.imshow(result_image_yellow_or_black)
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Canny Edges Detected')
    plt.imshow(canny_edges)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Contours From Canny Edge')
    plt.imshow(contours_image)
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('Filtered Contours Image Marigolds')
    plt.imshow(contours_image_2)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('yellow_image_results.png')
    plt.show()
    return num_filtered_contours_marigold,contours_image_2

def blue(image):
    num_filtered_contours_blue = 0
    x = 0
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 100])  # Adjust the lower and upper bounds for blue
    upper_blue = np.array([120, 255, 255])
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    result_image_rgb_blue = cv2.bitwise_and(image, image, mask=mask)
    # Assuming highly_black is a function defined elsewhere
    x = highly_black(result_image_rgb_blue, x)
    if x != 0:
        return num_filtered_contours_blue
    cv2.imshow('Only Blue Regions Image', result_image_rgb_blue)
    cv2.imwrite('blue1.png', result_image_rgb_blue)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    result_image_blue_or_black = np.zeros_like(result_image_rgb_blue)
    result_image_blue_or_black[np.where(mask != 0)] = [255, 255, 255]
    x = highly_black(result_image_blue_or_black, x)
    if x != 0:
        return num_filtered_contours_blue
    cv2.imshow('Converted Image Black & White', result_image_blue_or_black)
    cv2.imwrite('blue2.png', result_image_blue_or_black)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray_image = cv2.cvtColor(result_image_blue_or_black, cv2.COLOR_RGB2GRAY)
    canny_edges = cv2.Canny(gray_image, 100, 200)
    cv2.imshow('Canny Edges Detected', canny_edges)
    cv2.imwrite('blue3.png', canny_edges)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_image = result_image_rgb_blue.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours From Canny Edge', contours_image)
    cv2.imwrite('blue4.png', contours_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= 19]
    contours_image_3 = result_image_rgb_blue.copy()
    cv2.drawContours(contours_image_3, filtered_contours, -1, (0, 255, 0), 2)
    num_filtered_contours_blue = len(filtered_contours)
    for i, contour in enumerate(filtered_contours):
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(contours_image_3, str(i + 1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    plt.figure(figsize=(15, 10))
    cv2.imshow('Filtered Contours Image', contours_image_3)
    cv2.imwrite('blue5.png', contours_image_3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title('Input Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Only Blue Regions Image')
    plt.imshow(result_image_rgb_blue)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Converted Image Black & White')
    plt.imshow(result_image_blue_or_black)
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Canny Edges Detected')
    plt.imshow(canny_edges)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Contours From Canny Edge')
    plt.imshow(contours_image)
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('Filtered Contours Image Blue')
    plt.imshow(contours_image_3)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('blue_image_results.png')
    plt.show()

    return num_filtered_contours_blue,contours_image_3


def cyan(image):
    num_filtered_contours_cyan = 0
    x = 0
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_cyan = np.array([80, 100, 100])  # Adjust the lower and upper bounds for cyan
    upper_cyan = np.array([100, 255, 255])
    mask = cv2.inRange(hsv_image, lower_cyan, upper_cyan)
    result_image_rgb_cyan = cv2.bitwise_and(image, image, mask=mask)
    # Assuming highly_black is a function defined elsewhere
    x = highly_black(result_image_rgb_cyan, x)
    if x != 0:
        return num_filtered_contours_cyan
    cv2.imshow('Only Cyan Regions Image', result_image_rgb_cyan)
    cv2.imwrite('cyan1', result_image_rgb_cyan)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mask = cv2.inRange(hsv_image, lower_cyan, upper_cyan)
    result_image_cyan_or_black = np.zeros_like(result_image_rgb_cyan)
    result_image_cyan_or_black[np.where(mask != 0)] = [255, 255, 255]
    x = highly_black(result_image_cyan_or_black, x)
    if x != 0:
        return num_filtered_contours_cyan
    cv2.imshow('Converted Image Black & White', result_image_cyan_or_black)
    cv2.imwrite('cyan2.png', result_image_cyan_or_black)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray_image = cv2.cvtColor(result_image_cyan_or_black, cv2.COLOR_RGB2GRAY)
    canny_edges = cv2.Canny(gray_image, 100, 200)
    cv2.imshow('Canny Edges Detected', canny_edges)
    cv2.imwrite('cyan3.png', canny_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_image = result_image_rgb_cyan.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours From Canny Edge', contours_image)
    cv2.imwrite('cyan4.png', contours_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= 19]
    contours_image_4 = result_image_rgb_cyan.copy()
    cv2.drawContours(contours_image_4, filtered_contours, -1, (0, 255, 0), 2)
    num_filtered_contours_cyan = len(filtered_contours)
    for i, contour in enumerate(filtered_contours):
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(contours_image_4, str(i + 1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    plt.figure(figsize=(15, 10))
    cv2.imshow('Filtered Contours Image', contours_image_4)
    cv2.imwrite('cyan5.png', contours_image_4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title('Input Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Only Cyan Regions Image')
    plt.imshow(result_image_rgb_cyan)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Converted Image Black & White')
    plt.imshow(result_image_cyan_or_black)
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Canny Edges Detected')
    plt.imshow(canny_edges)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Contours From Canny Edge')
    plt.imshow(contours_image)
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('Filtered Contours Image Cyan')
    plt.imshow(contours_image_4)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('cyan_image_results.png')
    plt.show()

    return num_filtered_contours_cyan,contours_image_4


def magenta(image):
    num_filtered_contours_magenta = 0
    x = 0
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_magenta = np.array([140, 100, 100])  # Adjust the lower and upper bounds for magenta
    upper_magenta = np.array([160, 255, 255])
    mask = cv2.inRange(hsv_image, lower_magenta, upper_magenta)
    result_image_rgb_magenta = cv2.bitwise_and(image, image, mask=mask)
    # Assuming highly_black is a function defined elsewhere
    x = highly_black(result_image_rgb_magenta, x)
    if x != 0:
        return num_filtered_contours_magenta
    cv2.imshow('Only Magenta Regions Image', result_image_rgb_magenta)
    cv2.imwrite('magenta1.png', result_image_rgb_magenta)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mask = cv2.inRange(hsv_image, lower_magenta, upper_magenta)
    result_image_magenta_or_black = np.zeros_like(result_image_rgb_magenta)
    result_image_magenta_or_black[np.where(mask != 0)] = [255, 255, 255]
    x = highly_black(result_image_magenta_or_black, x)
    if x != 0:
        return num_filtered_contours_magenta
    cv2.imshow('Converted Image Black & White', result_image_magenta_or_black)
    cv2.imwrite('magenta2.png', result_image_magenta_or_black)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray_image = cv2.cvtColor(result_image_magenta_or_black, cv2.COLOR_RGB2GRAY)
    canny_edges = cv2.Canny(gray_image, 100, 200)
    cv2.imshow('Canny Edges Detected', canny_edges)
    cv2.imwrite('magenta3.png', canny_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_image = result_image_rgb_magenta.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours From Canny Edge', contours_image)
    cv2.imwrite('magenta4.png', contours_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= 19]
    contours_image_2 = result_image_rgb_magenta.copy()
    cv2.drawContours(contours_image_2, filtered_contours, -1, (0, 255, 0), 2)
    num_filtered_contours_magenta = len(filtered_contours)
    for i, contour in enumerate(filtered_contours):
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(contours_image_2, str(i + 1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    plt.figure(figsize=(15, 10))
    cv2.imshow('Filtered Contours Image', contours_image_2)
    cv2.imwrite('magenta5.png', contours_image_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title('Input Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Only Magenta Regions Image')
    plt.imshow(result_image_rgb_magenta)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Converted Image Black & White')
    plt.imshow(result_image_magenta_or_black)
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Canny Edges Detected')
    plt.imshow(canny_edges)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Contours From Canny Edge')
    plt.imshow(contours_image)
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('Filtered Contours Image Magenta')
    plt.imshow(contours_image_2)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('magenta_image_results.png')
    plt.show()

    return num_filtered_contours_magenta,contours_image_2

def white(image):
    num_filtered_contours_white = 0
    x = 0
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the HSV range for white
    lower_white = np.array([0, 0, 200])  # Lower bound for white
    upper_white = np.array([180, 30, 255])  # Upper bound for white
    
    mask = cv2.inRange(hsv_image, lower_white, upper_white)
    result_image_rgb_white = cv2.bitwise_and(image, image, mask=mask)
    
    x = highly_black(result_image_rgb_white, x)
    if x != 0:
        return num_filtered_contours_white
    
    save_and_show('Only White Regions Image', 'Only_White_Regions_Image.png', result_image_rgb_white)
    
    result_image_white_or_black = create_black_and_white_image(mask, result_image_rgb_white)
    x = highly_black(result_image_white_or_black, x)
    if x != 0:
        return num_filtered_contours_white
    
    save_and_show('Converted Image Black & White', 'Converted_Image_Black_&_White.png', result_image_white_or_black)
    
    gray_image = cv2.cvtColor(result_image_white_or_black, cv2.COLOR_RGB2GRAY)
    canny_edges = cv2.Canny(gray_image, 100, 200)
    
    save_and_show('Canny Edges Detected', 'Canny_Edges_Detected.png', canny_edges)
    
    contours_image, contours = draw_contours(canny_edges, result_image_rgb_white, 'Contours From Canny Edge', 'Contours_From_Canny_Edge.png')
    
    filtered_contours, contours_image_1 = filter_and_label_contours(contours, result_image_rgb_white)
    num_filtered_contours_white = len(filtered_contours)
    
    save_and_show('Filtered Contours Image', 'Filtered_Contours_Image.png', contours_image_1)
    
    plot_results(image, result_image_rgb_white, result_image_white_or_black, canny_edges, contours_image, contours_image_1, 'white_image_results.png')
    
    return num_filtered_contours_white, contours_image_1

def save_and_show(title, filename, image):
    cv2.imshow(title, image)
    cv2.imwrite(filename, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_black_and_white_image(mask, image):
    result_image = np.zeros_like(image)
    result_image[np.where(mask != 0)] = [255, 255, 255]
    return result_image

def draw_contours(edges, image, title, filename):
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_image = image.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    save_and_show(title, filename, contours_image)
    return contours_image, contours

def filter_and_label_contours(contours, image):
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= 19]
    contours_image = image.copy()
    for i, contour in enumerate(filtered_contours):
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(contours_image, str(i + 1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return filtered_contours, contours_image

def plot_results(image, white_image, bw_image, edges, contours_image, filtered_contours_image, filename):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.title('Input Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title('Only White Regions Image')
    plt.imshow(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title('Converted Image Black & White')
    plt.imshow(bw_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title('Canny Edges Detected')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title('Contours From Canny Edge')
    plt.imshow(cv2.cvtColor(contours_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.title('Filtered Contours Image White Region')
    plt.imshow(cv2.cvtColor(filtered_contours_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


import webcolors
from sklearn.cluster import MeanShift, estimate_bandwidth
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Manual dictionary of basic colors
BASIC_COLORS = {
    'black': '#000000',
    'white': '#FFFFFF',
    'red': '#FF0000',
    'green': '#00FF00',
    'blue': '#0000FF',
    'yellow': '#FFFF00',
    'cyan': '#00FFFF',
    'magenta': '#FF00FF'
}

# Function to open file dialog and select an image file
def select_image():
    # Creating the root Tkinter window and hide it
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Ensure the dialog is on top
    # Opening the file dialog
    filename = askopenfilename(title="Select an image file",
                               filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp")],
                               parent=root)  # Ensuring dialog is a child of root
    root.destroy()  # Destroying the root window after selection
    return filename

# Function to get the closest color name using a manual dictionary
def closest_color(requested_color):
    min_colors = {}
    for name, hex_value in BASIC_COLORS.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

# Function to get color name
def get_color_name(requested_color):
    try:
        closest_name = webcolors.rgb_to_name(requested_color)
    except ValueError:
        closest_name = closest_color(requested_color)
    return closest_name

# Selecting the image file
image_path = select_image()

# Reading the selected image file
image = cv2.imread(image_path)
# Convert the image from BGR to RGB (OpenCV loads images in BGR format)
image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshaping the image to a 2D array of pixels
pixels = image_copy.reshape(-1, 3)

# Estimating the bandwidth of the input data
bandwidth = estimate_bandwidth(pixels, quantile=0.2, n_samples=500)

# Performing mean-shift clustering
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
mean_shift.fit(pixels)

# Getting the cluster centers (main colors)
main_colors = mean_shift.cluster_centers_
main_colors = main_colors.astype(int)

# Get the labels for each pixel
labels = mean_shift.labels_

# Create a new image where each pixel is assigned the color of its cluster center
clustered_image = np.zeros_like(pixels)
for i, label in enumerate(labels):
    clustered_image[i] = main_colors[label]

# Reshape the clustered image to the original image shape
clustered_image = clustered_image.reshape(image.shape)

# Convert the clustered image back to BGR format for displaying with OpenCV
clustered_image_bgr = cv2.cvtColor(clustered_image, cv2.COLOR_RGB2BGR)

contours_image_red=0
contours_image_yellow=0
contours_image_blue=0
contours_image_cyan=0
contours_image_magenta=0
h,w = image.shape[0],image.shape[1]
if h<w:
    new_size = (600,400)
else:
    new_size = (400,600)
image = cv2.resize(image, new_size)
cv2.imshow('Input Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the clustered image
if h<w:
    new_size = (600,400)
else:
    new_size = (400,600)
clustered_image_bgr = cv2.resize(clustered_image_bgr, new_size)
cv2.imshow('Clustered Image', clustered_image_bgr)
cv2.imwrite('Clustered_Image.png', clustered_image_bgr)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# Getting the color names for each main color
color_names = [get_color_name(tuple(color)) for color in main_colors]

# Printing the main colors and their names
print("Main Colors and their Names:")
for color, name in zip(main_colors, color_names):
    print(f"Color: {color}, Name: {name}")

# Plotting the main colors
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_copy)
plt.title('Selected Image')
plt.axis('off')

plt.subplot(1, 2, 2)
for i, (color, name) in enumerate(zip(main_colors, color_names)):
    plt.plot([0, 1], [i, i], color=color / 255, linewidth=25)
    plt.text(1.1, i, name, verticalalignment='center')

plt.title('Main Colors')
plt.axis('off')
plt.savefig('main_colors_image_results.png')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

for color, name in zip(main_colors, color_names):
    if name=="yellow":
        num_filtered_contours_yellow,contours_image_yellow = yellow(image)
        print("Number of Yellow Flowers:", num_filtered_contours_yellow)
    if name=="blue":
        num_filtered_contours_blue,contours_image_blue = blue(image)
        print("Number of Blue Flowers:", num_filtered_contours_blue)
    if name=="cyan":
        num_filtered_contours_cyan,contours_image_cyan = cyan(image)
        print("Number of Cyan Flowers:", num_filtered_contours_cyan)
    if name=="red":
        num_filtered_contours_red,contours_image_red = red(image)
        print("Number of Red Flowers:", num_filtered_contours_red)
    if name=="magenta":
        num_filtered_contours_magenta,contours_image_magenta = magenta(image)
        print("Number of Magenta Flowers:", num_filtered_contours_magenta)
alpha = 0.7  # Weight of the red flowers image
beta = 1.3 - alpha  # Weight of the yellow flowers image
# Blending the images
blended_image = cv2.addWeighted(contours_image_red, alpha, contours_image_yellow + contours_image_blue + contours_image_cyan + contours_image_magenta, beta, 0.0)
cv2.imshow('Blended Image', blended_image)
cv2.imwrite('Blended_Image.png', blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.title('Output Image')
plt.imshow(blended_image)
plt.axis('off')
