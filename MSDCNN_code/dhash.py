import cv2
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def calculate_hash(image):
    #calculate the dhash of an image
    difference = __difference(image)
    # 1 => 8, 8 => 16
    decimal_value = 0 #十进制的变量
    hash_string = ""
    for index, value in enumerate(difference): #enumerate（）用于遍历different列表的下标index和元素value
        if value:
            decimal_value += value * (2 ** (index % 8))
        if index % 8 == 7:
            # 0xf=>0x0f
            hash_string += str(hex(decimal_value)[2:].rjust(2, "0"))
            decimal_value = 0
    hash=np.array(list(str(format(int(hash_string,16),'b'))))
    pad_len=64-len(hash)
    hash=np.pad(hash, (0, pad_len), 'constant', constant_values=0)
    # print(len(hash))
    return hash


def __difference(image):
    #find the difference with image
    resize_width = 9
    resize_height = 8
    # resize enough to hide the details
    smaller_image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_AREA) #调整大小
    grayscale_image = cv2.cvtColor(smaller_image, cv2.COLOR_BGR2GRAY) #灰度化
    # difference calculation
    pixels = grayscale_image.flatten() #把图片降成一维的
    difference = [] #定义差异列表
    for row in range(resize_height):
        row_start_index = row * resize_width
        for col in range(resize_width - 1):
            left_pixel_index = row_start_index + col
            difference.append(
                pixels[left_pixel_index] > pixels[left_pixel_index + 1])
    return difference

if __name__=='__main__': #这语句下面是测试用的，通文件夹import这个py文件，这个语句下面不会被执行
    path1='D:\\TensorFlow program\\MDCNN2\\test_photo\\frame1.jpg'
    path2='D:\\TensorFlow program\\MDCNN2\\test_photo\\frame3.jpg'
    img1=cv2.imread(path1)
    img2=cv2.imread(path2)
    print(calculate_hash(img1))
    print(calculate_hash(img2))
