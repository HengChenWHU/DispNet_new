import tensorflow as tf
import numpy as np
import os
import operator
from IO import read,write
from model_v2 import *
import matplotlib.pyplot as plt
import cv2
import random
import gc

def get_flingthings3d_list(fly3d_path,disparity_path):
    image_left=[]
    image_right=[]
    image_gt_l=[]
    image_gt_r=[]
    
    image_left_name=[]
    image_right_name=[]
    image_gt_l_name=[]
    image_gt_r_name=[]
    
    for file1 in os.listdir(fly3d_path):
        
        if file1=='frames_finalpass':
                    for file2 in os.listdir(fly3d_path+"/"+file1):
                        
                        #if file2 =='TRAIN':
                            for file3 in os.listdir(fly3d_path+"/"+file1+"/"+file2):
                                
                                for file4 in os.listdir(fly3d_path+"/"+file1+"/"+file2+"/"+file3):#A,B,C
                                    
                                    for file5 in os.listdir(fly3d_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4):#0000,0001
                                        
                                            if file5=='left':
                                                for file6_left in os.listdir(fly3d_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4+"/"+file5):
                                                    image_left.append(fly3d_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4+"/"+file5+"/"+file6_left)
                                                    image_left_name.append(file5+"/"+file6_left)
                                            if file5=='right':
                                                for file6_right in os.listdir(fly3d_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4+"/"+file5):
                                                    image_right_name.append(file5+"/"+file6_left)
                                                    image_right.append(fly3d_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4+"/"+file5+"/"+file6_right)
    
    for file1 in os.listdir(disparity_path):
        
        #if file1 =='TRAIN':
            
            for file2 in os.listdir(disparity_path+"/"+file1):#A,B,C
                #print("check")
                #print(file2)
                for file3 in os.listdir(disparity_path+"/"+file1+"/"+file2):#0000,0001
                    for file4 in os.listdir(disparity_path+"/"+file1+"/"+file2+"/"+file3):#LEFT,RIGHT
                            if file4=='left':
                                for file5 in os.listdir(disparity_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4):
                                    image_gt_l.append(disparity_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4+"/"+file5)
                                    image_gt_l_name.append(file4+"/"+file5)
                            if file4=='right':
                                for file5 in os.listdir(disparity_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4):
                                    image_gt_r.append(disparity_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4+"/"+file5)
                                    image_gt_r_name.append(file4+"/"+file5)
    
    print("lenth of flingthings3d")
    print(len(image_left))
    print(len(image_right))
    print(len(image_gt_l))
    print(len(image_gt_r))
    return image_left,image_right,image_gt_l,image_gt_r



                     
    

def get_monkaa__disparity_list(monkaa_path,disparity_path):
    #print(monkaa_path)
    #print(disparity_path)
    image_left=[]
    image_right=[]
    image_gt_l=[]
    image_gt_r=[]
    
    image_left_name=[]
    image_right_name=[]
    image_gt_l_name=[]
    image_gt_r_name=[]
    
    for file1 in os.listdir(monkaa_path):
        #print(file1)
        if file1=='frames_finalpass':
                    for file2 in os.listdir(monkaa_path+"/"+file1):
                        #print(file2)
                        for file3 in os.listdir(monkaa_path+"/"+file1+"/"+file2):
                            if file3=='left':
                                for file4_left in os.listdir(monkaa_path+"/"+file1+"/"+file2 +"/"+file3):
                                    image_left.append(monkaa_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4_left)
                                    #str1=monkaa_path+"/"+file1+"/"+file2+"/"+file3_left
                                    #print("hello")
                                    image_left_name.append(file3+"/"+file4_left)
                            if file3=='right':
                                for file4_right in os.listdir(monkaa_path+"/"+file1+"/"+file2 +"/"+file3):
                                    image_right_name.append(file3+"/"+file4_right)
                                    image_right.append(monkaa_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4_right)
                        
                        
                                        
                                            
    
    for file1 in os.listdir(disparity_path):
        
        #if file1 =='TRAIN':
            
            for file2 in os.listdir(disparity_path+"/"+file1):#A,B,C
                #print("check")
                #print(file2)
                for file3 in os.listdir(disparity_path+"/"+file1+"/"+file2):#0000,0001
                            if file3=='left':
                                for file4 in os.listdir(disparity_path+"/"+file1+"/"+file2+"/"+file3):
                                    image_gt_l.append(disparity_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4)
                                    image_gt_l_name.append(file3+"/"+file4)
                            if file3=='right':
                                for file4 in os.listdir(disparity_path+"/"+file1+"/"+file2+"/"+file3):
                                    image_gt_r.append(disparity_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4)
                                    image_gt_r_name.append(file3+"/"+file4)
    print("lenth of monkka")
    print(len(image_left))
    print(len(image_right))
    print(len(image_gt_l))
    print(len(image_gt_r))
    return image_left,image_right,image_gt_l,image_gt_r
                
def get_driving__disparity_list(driving,disparity_path):
    print(driving)
    print(disparity_path)
    image_left=[]
    image_right=[]
    image_gt_l=[]
    image_gt_r=[]
    
    image_left_name=[]
    image_right_name=[]
    image_gt_l_name=[]
    image_gt_r_name=[]
    
    for file1 in os.listdir(driving):
        #print(file1)
        if file1=='frames_finalpass':
                    for file2 in os.listdir(driving+"/"+file1):
                        #print(file2)
                        for file3 in os.listdir(driving+"/"+file1+"/"+file2):
                            #print(file3)
                            for file4 in os.listdir(driving+"/"+file1+"/"+file2+"/"+file3):
                                #print(file4)
                                for file5 in os.listdir(driving+"/"+file1+"/"+file2+"/"+file3+"/"+file4):
                                    #print(file5)
                                    if file5=='left':
                                        for file6_left in os.listdir(driving+"/"+file1+"/"+file2 +"/"+file3+"/"+file4+"/"+file5):
                                            #sss=driving+"/"+file1+"/"+file2 +"/"+file3+"/"+file4+"/"+file5+"/"+file6_left
                                            #print(sss)
                                            image_left.append(driving+"/"+file1+"/"+file2 +"/"+file3+"/"+file4+"/"+file5+"/"+file6_left)
                                            #str1=monkaa_path+"/"+file1+"/"+file2+"/"+file3_left
                                            #print("hello")
                                            image_left_name.append(file5+"/"+file6_left)
                                    if file5=='right':
                                        for file6_right in os.listdir(driving+"/"+file1+"/"+file2 +"/"+file3+"/"+file4+"/"+file5):
                                            image_right_name.append(file5+"/"+file6_right)
                                            image_right.append(driving+"/"+file1+"/"+file2 +"/"+file3+"/"+file4+"/"+file5+"/"+file6_right)
                        
                        
                                        
    
    for file1 in os.listdir(disparity_path):
        
        #if file1 =='TRAIN':
            
            for file2 in os.listdir(disparity_path+"/"+file1):#A,B,C
                #print("check")
                #print(file2)
                for file3 in os.listdir(disparity_path+"/"+file1+"/"+file2):#0000,0001
                    for file4 in os.listdir(disparity_path+"/"+file1+"/"+file2+"/"+file3):#0000,0001
                        for file5 in os.listdir(disparity_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4):#0000,0001
                            if file5=='left':
                                for file6_left in os.listdir(disparity_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4+"/"+file5):
                                    image_gt_l.append(disparity_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4+"/"+file5+"/"+file6_left)
                                    image_gt_l_name.append(file5+"/"+file6_left)
                            if file5=='right':
                                for file6_right in os.listdir(disparity_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4+"/"+file5):
                                    image_gt_r.append(disparity_path+"/"+file1+"/"+file2+"/"+file3+"/"+file4+"/"+file5+"/"+file6_right)
                                    image_gt_r_name.append(file5+"/"+file6_right)
        
    
    print("lenth of driving")
    print(len(image_left))
    print(len(image_right))
    print(len(image_gt_l))
    print(len(image_gt_r))
    return image_left,image_right,image_gt_l,image_gt_r
    

def get_MPI_Sintel_stereo_training_list(MPI_path):
    image_left=[]
    image_right=[]
    image_gt=[]
    
    image_left_name=[]
    image_right_name=[]
    image_gt_name=[]
    
    for file1 in os.listdir(MPI_path):
        if file1=='final_left':
            for file2 in os.listdir(MPI_path+"/"+file1):
                for file3 in os.listdir(MPI_path+'/'+file1+"/"+file2):
                    image_left.append(MPI_path+'/'+file1+"/"+file2+"/"+file3)
                    image_left_name.append(file2+"/"+file3)
        if file1=='final_right':
            for file2 in os.listdir(MPI_path+'/'+file1):
                for file3 in os.listdir(MPI_path+'/'+file1+"/"+file2):
                    image_right.append(MPI_path+'/'+file1+"/"+file2+"/"+file3)
                    image_right_name.append(file2+"/"+file3)
        if file1=='disparities':
            for file2 in os.listdir(MPI_path+'/'+file1):
                for file3 in os.listdir(MPI_path+'/'+file1+"/"+file2):
                    image_gt.append(MPI_path+'/'+file1+"/"+file2+"/"+file3)
                    image_gt_name.append(file2+"/"+file3)
    
    return image_left,image_right,image_gt

def get_KITTI2015_list(KITTI2015_PATH):
    image_left=[]
    image_right=[]
    image_gt=[]
    
    image_left_name=[]
    image_right_name=[]
    image_gt_name=[]
    
    for file1 in os.listdir(KITTI2015_PATH):
        if file1=='image_2':
            for file2 in os.listdir(KITTI2015_PATH+"/"+file1):
                    image_left.append(KITTI2015_PATH+'/'+file1+"/"+file2)
                    image_left_name.append(file2)
        if file1=='image_3':
            for file2 in os.listdir(KITTI2015_PATH+'/'+file1):
                    image_right.append(KITTI2015_PATH+'/'+file1+"/"+file2)
                    image_right_name.append(file2)
        if file1=='disp_noc_0':
            for file2 in os.listdir(KITTI2015_PATH+'/'+file1):
                    image_gt.append(KITTI2015_PATH+'/'+file1+"/"+file2)
                    image_gt_name.append(file2)
    
    image_left=np.hstack((image_left))
    image_right=np.hstack((image_right))
    image_gt=np.hstack((image_gt))
    
    temp = np.array([image_left, image_right,image_gt])
    temp = temp.transpose()
    #np.random.shuffle(temp)

    #print(operator.eq(image_left_name,image_right_name))
    #print(operator.eq(image_left_name,image_gt_name))
    #print(operator.eq(image_gt_name,image_right_name))
  
    #print("已经输出文件列表")
    #将所有的img和lab转换成list
    image_left=list(temp[:,0])
    image_right=list(temp[:,1])
    image_gt=list(temp[:,2])
    
    #all_image_list=[image_left,image_right,image_gt]
    
    
    return image_left,image_right,image_gt

def collect_all_images(image_left,image_right,image_gt_l,image_gt_r):
    
        image_left=np.hstack((image_left))
        image_right=np.hstack((image_right))
        image_gt_l=np.hstack((image_gt_l))
        image_gt_r=np.hstack((image_gt_r))
        
        temp = np.array([image_left, image_right,image_gt_l,image_gt_r])
        temp = temp.transpose()
        np.random.shuffle(temp)

  
        print("已经输出文件列表")
        #将所有的img和lab转换成list
        image_left=list(temp[:,0])
        image_right=list(temp[:,1])
        image_gt_l=list(temp[:,2])
        image_gt_r=list(temp[:,3])
    
        all_image_list=[image_left,image_right,image_gt_l,image_gt_r]
        #print("all the images number")
        #print(len(all_image_list))
    
        return all_image_list
    
def progress_picture(image_train,image_h,image_w):
    #数据加强过程
    
    #image_train = tf.image.resize_image_with_crop_or_pad(image_train, image_h, image_w)
    #随机设置图片的亮度
    image_train = tf.image.random_brightness(image_train,max_delta=5)
    #随机设置图片的对比度
    image_train = tf.image.random_contrast(image_train,lower=0.2,upper=1.8)
    
    #随机设置图片的色度
    image_train = tf.image.random_hue(image_train,max_delta=0.3)
    
    #随机设置图片的饱和度
    image_train = tf.image.random_saturation(image_train,lower=0.2,upper=1.8)
    
    #image_train=tf.squeeze(image_train)
    #print()
    # 图像标准化，
    #image_train = tf.image.per_image_standardization(image_train)
    
    #image_train=tf.expand_dims(image_train,0)
    image_train = tf.cast(image_train, tf.float32)  # 转换数据类型并归一化
    
    return image_train

#构建mini-batch

def get_mini_batch(all_image_list,BATCH_SIZE):
    n=len(all_image_list[0])#获得全部文件的个数
    #print(dataL.shape)
    mini_batches = []
    
    image_left=all_image_list[0]
    image_right=all_image_list[1]
    image_gt_l=all_image_list[2]
    image_gt_r=all_image_list[3]
    
    batch_num = int(n / BATCH_SIZE)
    for k in range(0, batch_num):
        mini_batch_L = image_left[k*BATCH_SIZE:(k+1)*BATCH_SIZE]
        mini_batch_R = image_right[k*BATCH_SIZE:(k+1)*BATCH_SIZE]
        mini_batch_D_L = image_gt_l[k*BATCH_SIZE:(k+1)*BATCH_SIZE]
        mini_batch_D_R = image_gt_r[k*BATCH_SIZE:(k+1)*BATCH_SIZE]
        mini_batches.append([mini_batch_L, mini_batch_R, mini_batch_D_L,mini_batch_D_R])

    if n % BATCH_SIZE != 0:
        mini_batch_L = image_left[batch_num*BATCH_SIZE:]
        mini_batch_R = image_right[batch_num*BATCH_SIZE:]
        mini_batch_D_L = image_gt_l[batch_num*BATCH_SIZE:]
        mini_batch_D_R = image_gt_r[batch_num*BATCH_SIZE:]
        mini_batches.append([mini_batch_L, mini_batch_R, mini_batch_D_L,mini_batch_D_R])
    
    return mini_batches

def read_batch_image(mini_batches,IMG_H,IMG_W):
    imgll=[]
    imgrr=[]
    imggg_L=[]
    imggg_R=[]
    
    dirl=mini_batches[0]
    #print(dirl)
    dirr=mini_batches[1]
    #print(dirr)
    dirg_l=mini_batches[2]
    
    dirg_r=mini_batches[3]
    #print(dirg)
    n=len(dirl)

    for i in range(n):
        #print("start")
        temp_l_img=cv2.imread(dirl[i])
        temp_l_img=cv2.resize(temp_l_img,(IMG_W,IMG_H))
        #temp_l_img=imgnorm(temp_l_img)
        
        temp_r_img=cv2.imread(dirr[i])
        temp_r_img=cv2.resize(temp_r_img,(IMG_W,IMG_H))
        #temp_r_img=imgnorm(temp_r_img)
        
        temp_g_img_l=read(dirg_l[i])
        temp_g_img_l=cv2.resize(temp_g_img_l,(IMG_W,IMG_H))
        #temp_g_img_l=imgnorm(temp_g_img_l)
        
        temp_g_img_r=read(dirg_r[i])
        temp_g_img_r=cv2.resize(temp_g_img_r,(IMG_W,IMG_H))
        #temp_g_img_r=imgnorm(temp_g_img_r)
        
        #print("after read shape")
        #print(temp_l_img.shape)
        #print(temp_r_img.shape)
        #print(temp_g_img.shape)
        
        temp_g_img_l=temp_g_img_l[:,:,np.newaxis]
        temp_g_img_r=temp_g_img_r[:,:,np.newaxis]
        #print(temp_g_img.shape)
        
        
        
        
        imgll.append(temp_l_img)
        imgrr.append(temp_r_img)
        imggg_L.append(temp_g_img_l)
        imggg_R.append(temp_g_img_r)
        
        #del temp_l_img
        #del temp_r_img
        #del temp_g_img
        
        #gc.collect()
    return imgll,imgrr,imggg_L,imggg_R
    
def imgnorm(img):
  return (img - np.mean(img)) / np.std(img)


        
    
def showimgs(image_batch):#测试获取的batch中的图片能否正常显示
    n=len(image_batch[0])
    print('show a batch of image')
    img_l=image_batch[0]
    img_r=image_batch[1]
    img_g_l=image_batch[2]
    img_g_r=image_batch[3]
    
    print(len(img_l))
    #print(img_g[0].shape)
    
    for i in range(n):
        print("left")
        plt.imshow((img_l[i]))
        plt.show()
        print("img_l shape")
        print(img_l[i].shape)
        
        print("right")
        plt.imshow((img_r[i]))
        plt.show()
        print("img_r shape")
        print(img_r[i].shape)
        
        print("ground truth left")
        t=img_g_l[i]
        t=t[:,:,0]
        print(t.shape)
        plt.imshow(t)
        plt.show()
        print("img_g shape")
        #print(t)
        
        print("ground truth right")
        t2=img_g_r[i]
        t2=t2[:,:,0]
        print(t2.shape)
        plt.imshow(t2)
        plt.show()
        print("img_g shape")
        #print(t)
        

########################CNN defination###############
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import psutil
    import gc
    IMG_W = 1024
    IMG_H = 512
    IMG_SIZE=256
    BATCH_SIZE = 3
    CAPACITY = 200
    CHANNEL=3
    
    tf.reset_default_graph()#这一句话非常重要，如果没有这句话，就会出现重复定义变量的错误

    logs_train_dir='E:/study/graduate_document/my_densematching_program/log_train'

    left_dir = 'E:/study/graduate_document/my_densematching_program//train/image_2'
    right_dir = 'E:/study/graduate_document/my_densematching_program/train/image_3'
    groundtruth_dir = 'E:/study/graduate_document/my_densematching_program/train//disp_noc_0'
    output_dir='E:/study/graduate_document/my_densematching_program/model'

    test_left_dir = 'E:/study/graduate_document/my_densematching_program//test/image_2'
    test_right_dir = 'E:/study/graduate_document/my_densematching_program/test/image_3'
    test_groundtruth_dir = 'E:/study/graduate_document/my_densematching_program/test/disp_noc_0'
    logs_test_dir='E:/study/graduate_document/my_densematching_program/log_test'
    
    MPI_path='./MPI-Sintel-stereo-training-20150305/training'
    
    fly3d_path='D:/flyingthings3d__frames_finalpass'
    disparity_path='D:/disparity'
    
    
    monkaa_path='D:/monkaa__frames_finalpass'
    disparity_path_monkaa='D:/SceneFlow/monkaa__disparity'
    
    driving_final_fram="D:/driving__frames_finalpass"
    disparity_disparity='D:/SceneFlow/driving__disparity'
    
    image_left1,image_right1,image_gt_l1,image_gt_r1=get_flingthings3d_list(fly3d_path,disparity_path)
    #print("show me your shape")
    #print(len(image_left1))
    #image_left,image_right,image_gt=get_MPI_Sintel_stereo_training_list(MPI_path)
    image_left2,image_right2,image_gt_l2,image_gt_r2=get_monkaa__disparity_list(monkaa_path,disparity_path_monkaa)
    #print("show me your shape")
    #print(len(image_left2))
    image_left3,image_right3,image_gt_l3,image_gt_r3=get_driving__disparity_list(driving_final_fram,disparity_disparity)
    #print("show me your shape")
    #print(len(image_left3))
    
    all_left=image_left1+image_left2+image_left3
    all_right=image_right1+image_right2+image_right3
    all_gt_l=image_gt_l1+image_gt_l2+image_gt_l3
    all_gt_r=image_gt_r1+image_gt_r2+image_gt_r3
    all_images=collect_all_images(all_left,all_right,all_gt_l,all_gt_r)
    print("the number of images are:%d"%(len(all_images[0])))
    #all_image_list4=get_driving__disparity_list(driving_final_fram,disparity_disparity)
    #all_image_list=get_trainlist(test_left_dir,test_right_dir,test_groundtruth_dir)
    mini_batches=get_mini_batch(all_images,BATCH_SIZE)
    n=len(mini_batches)
    left_batch=[]
    right_batch=[]
    gt_batch=[]
    
    for i in range(10000):
        batch_index=random.randint(0, n-1)#就是这个SB数据出错
        
        left_batch,right_batch,gt_batch_l,gt_batch_r=read_batch_image(mini_batches[batch_index],IMG_H,IMG_W)
        image_batch=[left_batch,right_batch,gt_batch_l,gt_batch_r]
        
        info = psutil.virtual_memory()
        print(u'内存使用：',psutil.Process(os.getpid()).memory_info().rss)
        print(u'内存占比：',info.percent)  
        
        showimgs(image_batch)
        del left_batch
        gc.collect()
        del right_batch
        gc.collect()
        del gt_batch_l
        gc.collect()
        del gt_batch_r
        gc.collect()
        del image_batch
        gc.collect()
        
        print("顺利读取图片") 
        
        
        