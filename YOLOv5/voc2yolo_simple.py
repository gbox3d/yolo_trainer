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
ds_path = '/home/gbox3d/work/datasets/digit'
src_dir = 'voc_val'
dest_dir = 'valid'
no_log = False

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--ds-path', type=str,help='data set path')
parser.add_argument('--src', type=str,help='voc file path')
parser.add_argument('--dest', type=str ,help='yolo format dest path')
parser.add_argument('--no-log',action='store_true', help='Dont display log')

opt = parser.parse_args()
src_dir = opt.src
dest_dir = opt.dest
ds_path = opt.ds_path
no_log = opt.no_log

# print(no_log)
# print('data set path : ' + dataset_path)

# %%
def _doLabelTxt(src_path,dest_path,no_log=True) :
    # _test_dir = './test'
    # out_path = f'{dataset_path}'
    # _out_path = out_path + '/labels'

    # src_path = f'{dataset_path}/voc'

    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)  # delete output folder
    os.makedirs(dest_path) 
    os.makedirs(dest_path+'/labels') 
    os.makedirs(dest_path+'/images') 

    _path = Path(src_path)
    
    files = _path.glob('*')
    _file_list = list(files)
    xml_files = [x for x in _file_list if str(x).split('.')[-1].lower() in ['xml']]

    

    # _file_list_set = 

    # for _i in range(0, 3) :
    _out_path = dest_path + '/labels'
    # _file_list = _file_list_set[_i]

    # print(f'output :{ _out_path} , {len(_file_list)}')
    
    for _file in xml_files :
        
        # print(_file.stem)

        tree = parse(_file)
        rootNode = tree.getroot()
        # _fname = rootNode.find('filename').text.split('.')[0]
        _fname = _file.stem

        _fPath = f'{_out_path}/{_fname}.txt'
        
        if no_log is not True:
            print(f'parse {_fname}')
        
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
        _out_path_img = dest_path + '/images'

        _img_type_list = ['jpeg', 'png','jpg']
        _index = 0

        for _img_type in _img_type_list:
            _image_file = f'{src_path}/{_fname}.{_img_type}'
            if Path(_image_file).exists() == True :
                if no_log is not True: 
                    print(f'copy {_image_file} to {_out_path_img}')
                shutil.copy(_image_file,_out_path_img)
                break
            _index += 1

                
    print(f'done {len(xml_files)} files')

#%%
config_data = {}
label_dic = {}
dataset_config_file = f'{ds_path}/data.yaml'
try:
    with open(dataset_config_file, 'r') as f:
        config_data = yaml.safe_load(f)
        
        for _index,_lb in enumerate(config_data['names'] ):
            # print(_lb)
            label_dic[_lb] = _index
    # print(config_data)

    _doLabelTxt(
        src_path=f'{ds_path}/{src_dir}',
        dest_path=f'{ds_path}/{dest_dir}',
        no_log=no_log)
    print('complete')
except  Exception as ex:
    print('error : ')   
    config_data = None 
    print(ex)


# %%
