import time
from load_data import *
from model_v2 import *
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
from IO import read,write
import random
import gc
import psutil

pwd=os.getcwd()
train_dir = pwd+'\\data\\train_2000\\train_images\\train_set\\'
allimages=pwd+'\\data\\train_2000\\train_images\\allimages\\'
devlopment_set=pwd+'\\data\\train_2000\\train_images\\development\\'
logs_train_dir =pwd+ '\\data\\train_2000\\train_images\\train_label_2000_new.txt'
logs_dir = pwd+'\\logs_20181230\\'     # 检查点保存路径
MPI_path=pwd+'\\MPI-Sintel-stereo-training-20150305\\training'
KITTI2015_PATH=pwd+'\\test3'
OUTPUT_PATH=pwd+'\\output\\'


fly3d_path='D:/flyingthings3d__frames_finalpass'
disparity_path='D:/disparity'

monkaa_path='D:/monkaa__frames_finalpass'
disparity_path_monkaa='D:/SceneFlow/monkaa__disparity'
    
driving_final_fram="D:/driving__frames_finalpass"
disparity_disparity='D:/SceneFlow/driving__disparity'

test_image_l='F:/study/DispNet/testone/left/0006.png'
test_image_r='F:/study/DispNet/testone/right/0006.png'
test_image_gl='F:/study/DispNet/testone/gtl/0006.pfm'
test_image_gr='F:/study/DispNet/testone/gtr/0006.pfm'

# 训练模型
def training():
    IMG_SIZE = 256
    IMG_H=256
    IMG_W=512
    BATCH_SIZE = 4#batch size can not smaller than 2
    CAPACITY = 20
    MAX_STEP = 600000
    LEARNING_RATE = 1e-3
   
    training_flag = tf.placeholder(tf.bool)#
    
    global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0),trainable=False)#创建global_step参数

    #产生一个会话
    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存 
    config.allow_soft_placement=True
    config.log_device_placement=True
    config.gpu_options.allocator_type='BFC'
    sess = tf.Session(config=config)

    image_left = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_H, IMG_W, 3], name='image_left')
    image_right = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_H, IMG_W, 3], name='image_right') 
    ground_truth_l = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_H, IMG_W, 1], name='ground_truth')
    ground_truth_r = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_H, IMG_W, 1], name='ground_truth')
    
    
    
    #image_left=progress_picture(image_left,IMG_H,IMG_W)
    #image_right=progress_picture(image_left,IMG_H,IMG_W)
    #ground_truth=progress_picture(image_left,IMG_H,IMG_W)
    
    
    image_left=tf.to_float(image_left, name='ToFloat')
    image_right=tf.to_float(image_right, name='ToFloat')
    ground_truth_l=tf.to_float(ground_truth_l, name='ToFloat')
    ground_truth_r=tf.to_float(ground_truth_r, name='ToFloat')
    
    #ground_truth = tf.image.resize_image_with_crop_or_pad(image_gt, IMG_H, IMG_W)
    #image_right = tf.image.resize_image_with_crop_or_pad(image_right, IMG_H, IMG_W)
    #image_left = tf.image.resize_image_with_crop_or_pad(image_left, IMG_H, IMG_W)
    
    total_loss,final_left,final_right= inference2(image_left,image_right,ground_truth_l,ground_truth_r,BATCH_SIZE,'training')
    pre_l=tf.squeeze(final_left[5])
    pre_r=tf.squeeze(final_right[5])
    #ema = tf.train.ExponentialMovingAverage(0.95, global_step)
    #l2_loss=tf.add_n(tf.get_collection("losses"))
    total_loss=total_loss
    #ema_op = ema.apply([total_loss]) 
    tf.summary.scalar('loss',total_loss)
    ################################采用退化学习率##################################################
    decay_learning_rate=tf.train.exponential_decay(LEARNING_RATE,global_step,decay_steps=10000,decay_rate=0.5)
    
    ######################################采用adam优化函数######################################
    train_op = tf.train.AdamOptimizer(decay_learning_rate).minimize(total_loss)
     
    ######################################计算参数的数目######################################
    
    
    test_left_img=cv2.imread(test_image_l)
    test_right_img=cv2.imread(test_image_r)
    test_gt_l=read(test_image_gl)
    test_gt_r=read(test_image_gr)
    
    test_left_img=cv2.resize(test_left_img,(IMG_W,IMG_H))
    test_right_img=cv2.resize(test_right_img,(IMG_W,IMG_H))
    test_gt_l=cv2.resize(test_gt_l,(IMG_W,IMG_H))
    test_gt_r=cv2.resize(test_gt_r,(IMG_W,IMG_H))
    
    test_left_img=test_left_img[np.newaxis,:,:,:]
    test_right_img=test_right_img[np.newaxis,:,:,:]
    test_gt_l=test_gt_l[np.newaxis,:,:,np.newaxis]
    test_gt_r=test_gt_r[np.newaxis,:,:,np.newaxis]
    
    test_image_left=tf.convert_to_tensor(np.asarray(test_left_img))
    test_image_right=tf.convert_to_tensor(np.asarray(test_right_img))
    test_ground_truth_l=tf.convert_to_tensor(np.asarray(test_gt_l))
    test_ground_truth_r=tf.convert_to_tensor(np.asarray(test_gt_r))
    
    test_image_left=tf.to_float(test_image_left, name='ToFloat')
    test_image_right=tf.to_float(test_image_right, name='ToFloat')
    test_ground_truth_l=tf.to_float(test_ground_truth_l, name='ToFloat')
    test_ground_truth_r=tf.to_float(test_ground_truth_r, name='ToFloat')
    
    
    test_loss,test_left,test_right=inference2(test_image_left,test_image_right,test_ground_truth_l,test_ground_truth_r,1,'testing')
    test_pre_l=tf.squeeze(test_left[5])
    test_pre_r=tf.squeeze(test_right[5])
    
    var_list = tf.trainable_variables()
    paras_count = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_list])
    print('参数数目:%d' % sess.run(paras_count), end='\n\n')
    
    #######################真正读取图片###################################
    image_left1,image_right1,image_gt_l1,image_gt_r1=get_flingthings3d_list(fly3d_path,disparity_path)
    image_left2,image_right2,image_gt_l2,image_gt_r2=get_monkaa__disparity_list(monkaa_path,disparity_path_monkaa)
    image_left3,image_right3,image_gt_l3,image_gt_r3=get_driving__disparity_list(driving_final_fram,disparity_disparity)
    
    all_left=image_left1+image_left2+image_left3
    all_right=image_right1+image_right2+image_right3
    all_gt_l=image_gt_l1+image_gt_l2+image_gt_l3
    all_gt_r=image_gt_r1+image_gt_r2+image_gt_r3
    all_images=collect_all_images(all_left,all_right,all_gt_l,all_gt_r)
    
    mini_batches=get_mini_batch(all_images,BATCH_SIZE)
    n=len(mini_batches)
    #batch_index=random.randint(0, n-1)#就是这个SB数据出错
    #imgl,imgr,imgg=read_batch_image(mini_batches[batch_index],IMG_H,IMG_W)
    #imgl,imgr,imgg=read_batch_image(mini_batches[0],IMG_H,IMG_W)
    
    
   
    saver = tf.train.Saver(max_to_keep=1)
   

    add_global = global_step.assign_add(1)
    
    
    ckpt = tf.train.get_checkpoint_state(logs_dir)

    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')
        sess.run(tf.global_variables_initializer())
        
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter("val_record/", sess.graph)
    
    s_t = time.time()
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            #for batch_index in range(n):
            batch_index=random.randint(0, n-1)#就是这个SB数据出错
            #batch_index=6
            imgl,imgr,imggl,imggr=read_batch_image(mini_batches[batch_index],IMG_H,IMG_W)
            
            info = psutil.virtual_memory()
            print(u'内存使用：',psutil.Process(os.getpid()).memory_info().rss)
            
            imgl=np.asarray(imgl)
            imgr=np.asarray(imgr)
            imggl=np.asarray(imggl)
            imggr=np.asarray(imggr)
            
            #print("the shape of imgg")
            #print(imgg.shape)
            
            imgl=imgnorm(imgl)
            imgr=imgnorm(imgr)
            #imggl=imgnorm(imggl)
            #imggr=imgnorm(imggr)
                
            rate,g=sess.run([decay_learning_rate,add_global])
            #l2_loss_=sess.run(l2_loss)
            test_loss_=sess.run(test_loss)
            #ema_op_=sess.run(ema_op,feed_dict={image_left:imgl,image_right:imgr,ground_truth:imgg})
            pre_l_, pre_r_,total_loss_,train_op_ = sess.run([pre_l,pre_r,total_loss,train_op],feed_dict={image_left:imgl,image_right:imgr,ground_truth_l:imggl,ground_truth_r:imggr})
            
            #print(id(image_left))
            
            #print("predict shape")
            #print(pr1_.shape)
            
            if g % 1 == 0:  # 实时记录训练过程并显示
                runtime = time.time() - s_t
                #print('Step: %1f, loss: %.8f,l2_loss:%.5f,test_loss:%.5f,batch_index:%.4f,learning rate:%.10f'% (g, total_loss_,l2_loss_,test_loss_,batch_index,rate))
                print('Step: %1f, loss: %.8f,test_loss:%.5f,batch_index:%.4f,learning rate:%.10f'% (g, total_loss_,test_loss_,batch_index,rate))
                #print('Step: %6f,batch_index:%.4f,learning rate:%.10f'% (g,batch_index,rate))
                s_t = time.time()
                
            if g % 20 == 0:  # 实时记录训练过程并显示
                plt.imshow(pre_l_[0])
                plt.show()
                plt.imshow(imggl[0,:,:,0])
                plt.show()
                plt.imshow(pre_r_[0])
                plt.show()
                plt.imshow(imggr[0,:,:,0])
                plt.show()
                
                #rs=sess.run(merged)
                #writer.add_summary(rs, g)
            
                
                #eval2()

            if g % 500 == 0 or g == MAX_STEP - 1:  # 保存检查点
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                print("have already saved the model")
            del imgl,imgr,imggl,imggr,pre_l_,pre_r_
            gc.collect()
                    #eval()

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()

def eval2():
    IMG_SIZE = 50
    BATCH_SIZE = 1
    CAPACITY = 200
    MAX_STEP = 1
    IMG_H=512
    IMG_W=1024
    logs_dir = 'logs_20181230'     # 检查点目录
    
    sess2 = tf.Session()

    image_left_list,image_right_list,image_gt_list=get_KITTI2015_list(KITTI2015_PATH)
    n=len(image_left_list)
    print("all images are %.3d"%(n))
    
    image_left = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_H, IMG_W, 3], name='image_left')
    image_right = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_H, IMG_W, 3], name='image_right') 
    ground_truth_l = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_H, IMG_W, 1], name='ground_truth_l')
    ground_truth_r = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_H, IMG_W, 1], name='ground_truth_r')
    
    ground_truth_l = tf.image.resize_image_with_crop_or_pad(ground_truth_l, IMG_H, IMG_W)
    ground_truth_r = tf.image.resize_image_with_crop_or_pad(ground_truth_r, IMG_H, IMG_W)
    image_right = tf.image.resize_image_with_crop_or_pad(image_right, IMG_H, IMG_W)
    image_left = tf.image.resize_image_with_crop_or_pad(image_left, IMG_H, IMG_W)
        
    image_left = tf.cast(image_left, tf.float32)
    image_right = tf.cast(image_right, tf.float32)
    ground_truth_l = tf.cast(ground_truth_l, tf.float32)
    ground_truth_r = tf.cast(ground_truth_r, tf.float32)
        
    total_loss,final_left,final_right= inference2(image_left,image_right,ground_truth_l,ground_truth_r,BATCH_SIZE,'testing')
    
    pre_l=tf.squeeze(final_left[5])
    pre_r=tf.squeeze(final_right[5])
    
    # 载入检查点
    
    saver = tf.train.Saver()
    print('\n载入检查点...')
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess2, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')
        
    for i in range(n):
        
        imgl=cv2.imread(image_left_list[i])
        print(image_left_list[i])
        imgl=cv2.resize(imgl,(IMG_W,IMG_H))
        imgl=[imgl]
        
        imgr=cv2.imread(image_right_list[i])
        imgr=cv2.resize(imgr,(IMG_W,IMG_H))
        imgr=[imgr]
        
        imgl=imgnorm(imgl)
        imgr=imgnorm(imgr)
        
        imgg=read(image_gt_list[i])
        #imgg=cv2.imread(image_gt_list[i])#KITTI
        imgg=cv2.resize(imgg,(IMG_W,IMG_H))
        
        #imgg=imgg[:,:,0]#KITTI
        
        imgg=imgg[np.newaxis,:,:,np.newaxis]
    
        imgl=np.asarray(imgl)
        imgr=np.asarray(imgr)
        imgg=np.asarray(imgg)
        
        #image_left=tf.convert_to_tensor(np.asarray(imgl))
        ##image_right=tf.convert_to_tensor(np.asarray(imgr))
        #ground_truth=tf.convert_to_tensor(np.asarray(imgg))
        
        pre_l_, pre_r_,total_loss_ = sess2.run([pre_l,pre_r,total_loss],feed_dict={image_left:imgl,image_right:imgr,ground_truth_l:imgg,ground_truth_r:imgg})
        pr1_=pre_l_
        
        
        plt.imshow(imgg[0,:,:,0])
        plt.show()
        plt.imshow(pre_l_)
        plt.show()
        print(pre_l_)
                
        print("loss")
        print(total_loss_)
        
        
        name=OUTPUT_PATH+str(i)+".png"
        cv2.imwrite(name,pr1_)
    sess2.close()
    
# 测试检查点
def test():
    IMG_SIZE = 256
    IMG_H=256
    IMG_W=1024
    BATCH_SIZE = 3
    CAPACITY = 200
    MAX_STEP = 10000
    LEARNING_RATE = 1e-5
    MOVING_AVERAGE_DECAY = 0.99
    UPDATE_OPS_COLLECTION = 'update_ops'
    weight_decay=0.95

    logs_dir = 'logs_1'     # 检查点目录
    test_left='000055_10l.jpg'
    test_right='000055_10r.jpg'
    test_gt='000055_10d.jpg'
    left_image=read(test_left)
    right_image=read(test_right)
    gt_image=read(test_gt)
    
    left_image=cv2.resize(left_image,(IMG_W,IMG_H))
    right_image=cv2.resize(right_image,(IMG_W,IMG_H))
    gt_image=cv2.resize(gt_image,(IMG_W,IMG_H))
    
    image_left=tf.convert_to_tensor(np.asarray(left_image))
    image_right=tf.convert_to_tensor(np.asarray(right_image))
    ground_truth=tf.convert_to_tensor(np.asarray(gt_image))
    
    image_left=tf.expand_dims(image_left,0)
    image_right=tf.expand_dims(image_right,0)
    ground_truth=tf.expand_dims(ground_truth,0)
    
    image_left = tf.cast(image_left, tf.float32)
    image_right = tf.cast(image_right, tf.float32)
    ground_truth = tf.cast(ground_truth, tf.float32)
    
    left=[image_left,image_left,image_left]
    right=[image_right,image_right,image_right]
    gt=[ground_truth,ground_truth,ground_truth]
    
    
    
    total_loss,pr1 = inference(left,right,gt,BATCH_SIZE,'testing')
    
    sess = tf.Session()
    # 载入检查点
    saver = tf.train.Saver()
    print('\n载入检查点...')
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    correct_rate=0.00
    index=0
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            pr1_, total_loss_ = sess.run([pr1, total_loss])
        
        print("---------------finished testing------------")
        cv.imwrite('01.png',pr1_)
    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()


if __name__ == '__main__':
    tf.reset_default_graph()
    
    training()
    #eval2()
    #test()