import cv2
import os
def save_img():

    vc = cv2.VideoCapture('project_video.mp4') #读入视频文件
    c=0
    rval=vc.isOpened()
    folder_name = 'video_images'

    while rval:   #循环读取视频帧
        c = c + 1
        rval, frame = vc.read()
        pic_path = folder_name+'/'
        if rval:
            cv2.imwrite(pic_path + str(c) + '.jpg', frame) #存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
            cv2.waitKey(1)
        else:
            break
    vc.release()
    print('save_success')
    print(folder_name)
save_img()