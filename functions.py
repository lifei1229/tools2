import glob,os
from json import load
import imghdr
import json
import numpy as np
import cv2
from PIL import Image
################# count files numbers #################################

def countFiles(dir):
    dirs = os.listdir(dir)
    res = 0
    for dir_i in dirs:
    # path_file_number=glob.glob('D:/case/test/testcase/checkdata/*.py')#或者指定文件下个数
        path_file_number=glob.glob(pathname= dir  + dir_i + '/labels/*.json') #获取当前文件夹下个数
    
        res += len(path_file_number)

    return res
""" dir = '/data_nas/detection_data/SyntheticFrom3D/Multilingual/'
print( countFiles(dir) ) """

############################## files to unreal_files dir ############################

""" dir = '/data_nas/detection_data/SyntheticFrom3D/'
f=open("unreal_total.txt","a")
idx = 0
for synth_name in os.listdir(dir)[:2]:
    resDir = dir + synth_name #'/data_nas/detection_data/SyntheticFrom3D/Latin'
    for dir_sub in os.listdir(resDir):
        subs = os.path.join(resDir,dir_sub) #'/data_nas/detection_data/SyntheticFrom3D/Latin/sub_2'
        for sub in os.listdir(subs)[:1]:
            for file in os.listdir(os.path.join(subs,sub)):
                img_dir = os.path.join(subs,sub,file)
                print (f'write {img_dir} imgs {idx} ') 
                assert 'jpg' in file
                gt_dir = img_dir.replace('imgs','labels').replace('jpg','json')
                if os.path.exists(gt_dir) and os.path.getsize(gt_dir):
                    f.writelines(img_dir+";" + gt_dir + '\n' )
                    idx += 1
f.close() """

################################# read json label ########################################
""" gt = '/data_nas/detection_data/SyntheticFrom3D/Latin/sub_2/labels/106.json'
with open(gt,'r') as load_f:
    load_dict = json.load(load_f)
res = {}
for idx, line in enumerate(load_dict['bbox']):
    item = {}
    poly = np.array(list(map(float, line))).reshape((-1, 2)).tolist() #转化成为二维数组，对�?4个坐�?
    label = load_dict['text'][idx]
    item['poly'] = poly
    item['text'] = label
    lines.append(item)
res.append(lines)    """


################################ show gt of unreal text #############################

# gt = '/data_nas/detection_data/SyntheticFrom3D/Latin/sub_2/labels/135.json'
# img = gt.replace('labels','imgs').replace('json','jpg')
# im = cv2.imread(img)

# with open(gt,'r') as load_f:
#     load_dict = json.load(load_f)
# for idx, line in enumerate(load_dict['bbox']):
#   #  line.reshape((-1, 1, 2))
#     poly = np.array(list(map(int, line))).reshape((-1, 1,2)) #转化成为二维数组，对�?4个坐�?
#    # print (poly )
#     cv2.polylines(im, [poly], True, (0, 255, 255), 1)  ##[box]！！�?

# cv2.imwrite('135.jpg',im)
# cv2.waitKey(0)  


############################## random split train ####################################
#coding:utf-8

import random
import time

txt_dir = "/data_nas/detection_data/ANNOTATIONS/icdar2017-rctw/icdar2017-rctw.txt"
train_dir = "/data_nas/detection_data/ANNOTATIONS/icdar2017-rctw/icdar2017-rctw_train.txt"
tets_dir = "/data_nas/detection_data/ANNOTATIONS/icdar2017-rctw/icdar2017-rctw_test.txt"
f = open(txt_dir,"r")         #源文件
ft = open(train_dir,"w")     #待写文件
fv = open(tets_dir,"w")     #待写文件

def test():
    start = time.clock()
    raw_list = f.readlines()
    random.shuffle(raw_list)
    for i in range(len(raw_list)):    #随机抽取数目 n 可以替代range里面
        if i < 0.8 * len(raw_list):
            ft.writelines(raw_list[i])
        else:
            fv.writelines(raw_list[i])
    f.close()
    ft.close()
    fv.close()
    end = time.clock()
    print("cost time is %f" %(end - start))

test()

####################################### png 转化 jpg #########################
def PNG_JPG(PngPath):
    img = cv2.imread(PngPath, 0)
    w, h = img.shape[::-1]
    infile = PngPath
    outfile = os.path.splitext(infile)[0] + ".jpg"
    img = Image.open(infile)
    # img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=70)
            os.remove(PngPath)
        else:
            img.convert('RGB').save(outfile, quality=70)
            os.remove(PngPath)
        return outfile
    except Exception as e:
        print("PNG转换JPG 错误", e)

########################## icdar2019 mlt #############################
# dir = '/data_nas/detection_data/icdar2019-mlt/icdar2019-mlt_revise/ImagesPart2/ImagesPart2/'
# txt_dir = '/data_nas/detection_data/icdar2019-mlt/icdar2019-mlt_revise/icdar2019-mlt.txt'
# f=open(txt_dir,"a+")
# idx = 0
# for synth_name in os.listdir(dir):
#     img_dir = dir + synth_name #'/data_nas/detection_data/icdar2019-mlt/icdar2019-mlt_revise/ImagesPart1/ImagesPart1/tr_img_01002.jpg'
#     print (f'write {img_dir} imgs {idx} ') 
#     if img_dir.endswith('.png'):
#         print('#########################################')
#         PNG_JPG(img_dir)
#         img_dir = img_dir.replace('png','jpg')
#     assert 'jpg' in img_dir and os.path.exists(img_dir)
#     gt_dir = img_dir.replace('ImagesPart2','1-2_gt').replace('jpg','txt')
#     if os.path.exists(gt_dir) and os.path.getsize(gt_dir):
#         f.writelines(img_dir+";" + gt_dir + '\n' )
#         # print (f'write {gt_dir} imgs {idx} ') 
#         idx += 1
# f.close()

##################################### icdar2019 lsvt ################################ 
# label_dir = '/data_nas/detection_data/icdar2019-lsvt/detection/train_full_labels.json'
# txt_dir = '/data_nas/detection_data/icdar2019-lsvt/detection/gts/'
# with open(label_dir,'r') as f:
#     load_dict = json.load(f)
# idx = 0
# for key, line in load_dict.items():
#     with open(txt_dir+key+ '.txt','w') as f:
#         for item in line:
#             for xy in item['points']:
#                 f.write(str(xy[0]) + ',' + str(xy[1]) + ',')
#             f.write(item['transcription'] + '\n')
#     print (f'writen {key} .txt {idx}' )
#     idx += 1
        
# dir = '/data_nas/detection_data/icdar2019-lsvt/detection/imgs/'
# txt_dir = '/data_nas/detection_data/icdar2019-lsvt/detection/icdar2019-lsvt.txt'
# with open(txt_dir, 'a+') as f:
#     idx = 0
#     for synth_name in os.listdir(dir):
#         img_dir = dir + synth_name #'/data_nas/detection_data/icdar2019-lsvt/detection/imgs/gt_10294.jpg'
#         print (f'write {img_dir} imgs {idx} ') 
#         assert 'jpg' in img_dir and os.path.exists(img_dir)
#         gt_dir = img_dir.replace('imgs','gts').replace('jpg','txt')
#         if os.path.exists(gt_dir) and os.path.getsize(gt_dir):
#             f.writelines(img_dir+";" + gt_dir + '\n' )
#             # print (f'write {gt_dir} imgs {idx} ') 
#             idx += 1

##################################### icdar2019 art ################################ 
# label_dir = '/data_nas/detection_data/icdar2019-art/detection/gt/train_labels.json'
# txt_dir = '/data_nas/detection_data/icdar2019-art/detection/gts/'
# with open(label_dir,'r') as f:
#     load_dict = json.load(f)
# idx = 0
# for key, line in load_dict.items():
#     with open(txt_dir+key+ '.txt','w') as f:
#         for item in line:
#             for xy in item['points']:
#                 f.write(str(xy[0]) + ',' + str(xy[1]) + ',')
#             f.write(item['transcription'] + '\n')
#     print (f'writen {key} .txt {idx}' )
#     idx += 1

# dir = '/data_nas/detection_data/icdar2019-art/detection/train_images/'
# txt_dir = '/data_nas/detection_data/icdar2019-art/detection/icdar2019-art.txt'
# with open(txt_dir, 'a+') as f:
#     idx = 0
#     for synth_name in os.listdir(dir):
#         img_dir = dir + synth_name #'/data_nas/detection_data/icdar2019-lsvt/detection/imgs/gt_10294.jpg'
#         print (f'write {img_dir} imgs {idx} ') 
#         assert 'jpg' in img_dir and os.path.exists(img_dir)
#         gt_dir = img_dir.replace('train_images','gts').replace('jpg','txt')
#         if os.path.exists(gt_dir) and os.path.getsize(gt_dir):
#             f.writelines(img_dir+";" + gt_dir + '\n' )
#             # print (f'write {gt_dir} imgs {idx} ') 
#             idx += 1

###################################### icdar2017 rctw ################################ 
# label_dir = '/data_nas/detection_data/icdar2017-rctw/detection/train.json'
# txt_dir = '/data_nas/detection_data/icdar2017-rctw/detection/gts/'
# with open(label_dir,'r') as f:
#     load_dict = json.load(f)
# idx = 0
# for  line_img in load_dict['data_list']:
#     txt_name = line_img['img_name'].replace('jpg','txt')
#     with open(txt_dir+txt_name,'w') as f:
#         line = line_img['annotations']
#         for item in line:
#             for xy in item['polygon']:
#                 f.write(str(int(xy[0])) + ',' + str(int(xy[1])) + ',')
#             f.write(item['text'] + '\n')
#     print (f'writen {txt_name} .txt {idx}' )
#     idx += 1

# dir = '/data_nas/detection_data/icdar2017-rctw/detection/imgs/'
# txt_dir = '/data_nas/detection_data/icdar2017-rctw/detection/icdar2017-rctw.txt'
# with open(txt_dir, 'a+') as f:
#     idx = 0
#     for synth_name in os.listdir(dir):
#         if synth_name.endswith('txt'):
#             continue
#         img_dir = dir + synth_name #'/data_nas/detection_data/icdar2017-rctw/detection/imgs/image_10.jpg'
#         print (f'write {img_dir} imgs {idx} ') 
#         assert 'jpg' in img_dir and os.path.exists(img_dir)
#         gt_dir = img_dir.replace('imgs','gts').replace('jpg','txt')
#         if os.path.exists(gt_dir) and os.path.getsize(gt_dir):
#             f.writelines(img_dir+";" + gt_dir + '\n' )
#             # print (f'write {gt_dir} imgs {idx} ') 
#             idx += 1

########################## icdar2017 mlt #############################
# dir = '/data_nas/detection_data/icdar2017-mlt/ch8_validation_localization_transcription_gt_v2/'
# txt_dir = '/data_nas/detection_data/icdar2017-mlt/val_gt/'
# idx = 0
# for synth_name in os.listdir(dir):
#     resDir = dir + synth_name #'/data_nas/detection_data/SyntheticFrom3D/Latin'
#     with open(resDir, 'r') as f1:
#         lines = f1.readlines()
#         with open(txt_dir+synth_name , 'w') as f2:
#             for line in lines:
#                 line = line.strip().split(',')
#                 if line[-1] == '###':
#                     new_line = ','.join(line[:-2]) + ',' + line[-1] + '\n'
#                 else:
#                     new_line = ','.join(line[:-1]) + '\n'
#                 f2.write(new_line )
#         print (f'write {synth_name} txt {idx} ') 
#     idx += 1

# dir = '/data_nas/detection_data/icdar2017-mlt/ch8_validation_images/'
# txt_dir = '/data_nas/detection_data/icdar2017-mlt/icdar2017-mlt.txt'
# with open(txt_dir, 'a+') as f:
#     idx = 0
#     for synth_name in os.listdir(dir):
#         if synth_name.endswith('txt'):
#             continue
#         img_dir = dir + synth_name #'/data_nas/detection_data/icdar2017-rctw/detection/imgs/image_10.jpg'
#         if img_dir.endswith('.png'):
#             print('#########################################')
#             PNG_JPG(img_dir)
#             img_dir = img_dir.replace('png','jpg')

#         print (f'write {img_dir} imgs {idx} ') 
#         assert 'jpg' in img_dir and os.path.exists(img_dir)
#         gt_dir = img_dir.replace('ch8_validation_images','val_gt').replace('jpg','txt').replace('img_','gt_img_')
#         if os.path.exists(gt_dir) and os.path.getsize(gt_dir):
#             f.writelines(img_dir+";" + gt_dir + '\n' )
#             # print (f'write {gt_dir} imgs {idx} ') 
#             idx += 1

####################################### icdar 2015 ###########################
# dir = '/data_nas/detection_data/icdar2015/detection/test/imgs/'
# txt_dir = '/data_nas/detection_data/icdar2015/detection/icdar2015_test.txt'
# with open(txt_dir, 'a+') as f:
#     idx = 0
#     for synth_name in os.listdir(dir):
#         if synth_name.endswith('txt'):
#             continue
#         img_dir = dir + synth_name #'/data_nas/detection_data/icdar2017-rctw/detection/imgs/image_10.jpg'
#         if img_dir.endswith('.png'):
#             print('#########################################')
#             PNG_JPG(img_dir)
#             img_dir = img_dir.replace('png','jpg')

#         print (f'write {img_dir} imgs {idx} ') 
#         assert 'jpg' in img_dir and os.path.exists(img_dir)
#         gt_dir = img_dir.replace('imgs','gt').replace('jpg','txt').replace('img_','gt_img_')
#         if os.path.exists(gt_dir) and os.path.getsize(gt_dir):
#             f.writelines(img_dir+";" + gt_dir + '\n' )
#             # print (f'write {gt_dir} imgs {idx} ') 
#             idx += 1


################################ show gt of Cwhandwriting #############################
# gts = '/data_nas/detection_data/CwHandwirting/gts/all_is_gt/'
# imgdir = '/data_nas/detection_data/CwHandwirting/images/'
# for gt in os.listdir(gts):
#     im = cv2.imread(imgdir + gt.replace('txt','jpg'))
#     with open(gts + gt, 'r') as fid:
#         lines = fid.readlines()
#         for line in lines:
#             line = line.split(',')
#             poly = np.array(list(map(int, line[:-1]))).reshape((-1,1, 2))#转化成为二维数组，对应4个坐标
#     # print (poly )
#             cv2.polylines(im, [poly], True, (0, 255, 255), 1) 

#     cv2.imwrite('./gts/' + gt+'.jpg',im)
#     # cv2.waitKey(0)  

############################# Cwhandwriting ###########################
# gts = '/data_nas/detection_data/CwHandwirting/gts/all_is_gt/'
# dir = '/data_nas/detection_data/CwHandwirting/images/'
# txt_dir = '/data_nas/detection_data/CwHandwirting/cwHandwirting.txt'
# with open(txt_dir, 'a+') as f:
#     idx = 0
#     for synth_name in os.listdir(dir):
#         if synth_name.endswith('txt'):
#             continue
#         img_dir = dir + synth_name #'/data_nas/detection_data/icdar2017-rctw/detection/imgs/image_10.jpg'
#         if img_dir.endswith('.png'):
#             print('#########################################')
#             PNG_JPG(img_dir)
#             img_dir = img_dir.replace('png','jpg')

#         print (f'write {img_dir} imgs {idx} ') 
#         assert 'jpg' in img_dir and os.path.exists(img_dir)
#         gt_dir = img_dir.replace('images','gts/all_is_gt').replace('jpg','txt')
#         if os.path.exists(gt_dir) and os.path.getsize(gt_dir):
#             f.writelines(img_dir+";" + gt_dir + '\n' )
#             # print (f'write {gt_dir} imgs {idx} ') 
#             idx += 1


#################### total text #####################
dir = '/data_nas/detection_data/total_text/test_images/'
txt_dir = '/data_nas/detection_data/total_text/total_text_test.txt'
with open(txt_dir, 'a+') as f:
    idx = 0
    for synth_name in os.listdir(dir):
        if synth_name.endswith('txt'):
            continue
        img_dir = dir + synth_name #'/data_nas/detection_data/icdar2017-rctw/detection/imgs/image_10.jpg'
        if img_dir.endswith('.png'):
            print('#########################################')
            PNG_JPG(img_dir)
            img_dir = img_dir.replace('png','jpg')

        print (f'write {img_dir} imgs {idx} ') 
        assert 'jpg' in img_dir or 'JPG' in img_dir  and os.path.exists(img_dir)
        gt_dir = img_dir.replace('test_images','test_gts') + '.txt'
        if os.path.exists(gt_dir) and os.path.getsize(gt_dir):
            f.writelines(img_dir+";" + gt_dir + '\n' )
            # print (f'write {gt_dir} imgs {idx} ') 
            idx += 1