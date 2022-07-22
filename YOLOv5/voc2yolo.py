#%%
import yaml
import os
import shutil
import argparse 
from pathlib import Path
import numpy as np
import cv2
from random import shuffle
from math import trunc

# from IPython.display import display

import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image

from xml.etree.ElementTree import parse
#%%
# dataset_config_file = '../../../dataset/test/data.yaml'
# output_path = '../../../dataset/test/'
# dataset_path = '/home/gbox3d/work/datasets/digit/'
no_log = False
voc_path = 'voc'
config_file = '/home/gbox3d/work/visionApp/daisy_project/trainer/yolo_v5/config/digit_set_7.yaml'
# output_path = 'set_7'


#%%
parser = argparse.ArgumentParser()
# parser.add_argument('--dataset-path','-dp',type=str, default=dataset_path ,help='data set path')
parser.add_argument('--config-file','-cf',type=str, default=config_file ,help='config file')
parser.add_argument('--no-log',action='store_true', help='Dont display log')
# parser.add_argument('--voc-path','-vp',type=str, default=voc_path ,help='voc path')
# parser.add_argument('--output-path','-op',type=str, default=output_path ,help='output path')

opt = parser.parse_args()
# dataset_path = opt.dataset_path
no_log = opt.no_log
# voc_path = opt.voc_path
config_file = opt.config_file
# output_path = opt.output_path
# print(no_log)
# print('data set path : ' + dataset_path)

# %%
def _doLabelTxt( dataset_path,voc_path,train_rt=0.7,valid_rt=0.2,no_log=True) :
    # _test_dir = './test'
    # out_path = f'{dataset_path}'
    # _out_path = out_path + '/labels'

    src_path = os.path.join(dataset_path,voc_path)
    print(f'voc path : {src_path}')

    if train_rt + valid_rt < 1 :

        train_path =  os.path.join(dataset_path,'train')# f'{dataset_path}/train'
        val_path =  os.path.join(dataset_path,'valid')# f'{dataset_path}/valid'
        test_path =  os.path.join(dataset_path,'test')# f'{dataset_path}/test'
        # val_path = f'{dataset_path}/valid'
        # test_path = f'{dataset_path}/test'

        path_list = [train_path,val_path,test_path]

        for _path in path_list :
            if os.path.exists(_path):
                shutil.rmtree(_path)  # delete output folder
            os.makedirs(_path) 
            os.makedirs(_path+'/labels') 
            os.makedirs(_path+'/images') 

        _path = Path(src_path)
        
        files = _path.glob('*')
        _file_list = list(files)
        xml_files = [x for x in _file_list if str(x).split('.')[-1].lower() in ['xml']]
        
        #검증세트 분리하기 
        shuffle(xml_files)

        _size = len(xml_files)

        _start_index = 0
        _end_index = trunc(_size * train_rt)
        _train_list = xml_files[ _start_index : _end_index]
        _start_index = _end_index
        _end_index =  trunc( _size * (train_rt + valid_rt) )
        _valid_list = xml_files[ _start_index : _end_index ]
        _test_list = xml_files[ _end_index :  ]
        
        _file_list_set = (_train_list,_valid_list,_test_list)
        print(f'train : {len(_train_list)} , valid : {len(_valid_list)} , test : {len(_test_list)}')
    else : # test set skip
        train_path =  os.path.join(dataset_path,'train')# f'{dataset_path}/train'
        val_path =  os.path.join(dataset_path,'valid')# f'{dataset_path}/valid'
        # test_path =  os.path.join(dataset_path,'test')# f'{dataset_path}/test'
        # val_path = f'{dataset_path}/valid'
        # test_path = f'{dataset_path}/test'

        path_list = [train_path,val_path]

        for _path in path_list :
            if os.path.exists(_path):
                shutil.rmtree(_path)  # delete output folder
            os.makedirs(_path) 
            os.makedirs(_path+'/labels') 
            os.makedirs(_path+'/images') 

        _path = Path(src_path)
        
        files = _path.glob('*')
        _file_list = list(files)
        xml_files = [x for x in _file_list if str(x).split('.')[-1].lower() in ['xml']]
        
        #검증세트 분리하기 
        shuffle(xml_files)

        _size = len(xml_files)

        _start_index = 0
        _end_index = trunc(_size * train_rt)
        _train_list = xml_files[ _start_index : _end_index]
        _start_index = _end_index
        _end_index =  trunc( _size * (train_rt + valid_rt) )
        _valid_list = xml_files[ _start_index : _end_index ]
        # _test_list = xml_files[ _end_index :  ]
        
        _file_list_set = (_train_list,_valid_list)
        print(f'train : {len(_train_list)} , valid : {len(_valid_list)} ')


    # for _i in range(0, 3) :
    for _i,_file_list in enumerate(_file_list_set) :
        _out_path = path_list[_i] + '/labels'
        # _file_list = _file_list_set[_i]

        print(f'output :{ _out_path} , {len(_file_list)}')
        
        for _file in _file_list :
            # print(_file.stem)
            tree = parse(_file)
            rootNode = tree.getroot()
            # _fname = rootNode.find('filename').text.split('.')[0]
            _fname = _file.stem

            _fPath = f'{_out_path}/{_fname}.txt'
            if no_log is not True:
                print(_fname)
            
            _objs = rootNode.findall('object')
            for _obj in _objs :
                _label = _obj.find('name').text
                
                _bbox = _obj.find('bndbox')
                if _bbox is not None :
                    xmin =  float(_bbox.find('xmin').text)
                    ymin =  float(_bbox.find('ymin').text)
                    xmax =  float(_bbox.find('xmax').text)
                    ymax =  float(_bbox.find('ymax').text)
                else :
                    # 바운딩 박스로 만들기 
                    segment = [ [float(v.find('x').text),float(v.find('y').text)] for v in _obj.findall('segmentation') ]
                    xmin = min(segment, key=lambda x: x[0])[0]
                    ymin = min(segment, key=lambda x: x[1])[1]
                    xmax = max(segment, key=lambda x: x[0])[0]
                    ymax = max(segment, key=lambda x: x[1])[1]

                _imgW = float(rootNode.find('size').find('width').text)
                _imgH = float(rootNode.find('size').find('height').text)

                _xcenter = (((xmin + xmax)/2) / _imgW)
                _ycenter = (((ymin + ymax)/2) / _imgH)
                _w = ((xmax - xmin) / _imgW )
                _h = ((ymax - ymin) / _imgH )

                if _label in label_dic :
                    _out = f'{label_dic[_label]} {round(_xcenter,4)} {round(_ycenter,4)} {round(_w,4)} {round(_h,4)} \n'
                    # print(_out)
                                
                    with open(_fPath,'a') as fd:
                        fd.write(_out)

            # 이미지 파일 카피 
            _out_path_img = path_list[_i] + '/images'

            _img_type_list = ['jpeg', 'png','jpg']
            _index = 0

            for _img_type in _img_type_list:
                _image_file = f'{src_path}/{_fname}.{_img_type}'
                if Path(_image_file).exists() == True :
                    if no_log is not True: 
                        # print(_out,end='')
                        print(f'copy {_image_file} to {_out_path_img}')
                    shutil.copy(_image_file,_out_path_img)
                    break

                _index += 1
    

#%%
config_data = {}
label_dic = {}
# dataset_config_file = f'{dataset_path}/data.yaml'


try:
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
        
        for _index,_lb in enumerate(config_data['names'] ):
            # print(_lb)
            label_dic[_lb] = _index
        # dataset_path = 
        
    # print(config_data)

    _doLabelTxt(
        dataset_path = config_data['path'],
        voc_path=config_data['voc'],
        train_rt=config_data['split']['train'],
        valid_rt=config_data['split']['val'],
        no_log=no_log)
    print('complete')
except  Exception as ex:
    print('error : ')   
    config_data = None 
    print(ex)
