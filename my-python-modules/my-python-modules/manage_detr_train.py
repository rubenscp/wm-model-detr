"""
Institution: Institute of Computing - University of Campinas (IC/Unicamp)
Project: White Mold 
Description: Implements the DETR neural network model for step of inference.
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at DCC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 14/05/2024
Version: 1.0
This implementation is based on this notebook: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb 
"""

# Setting folder of common libraries of White Mold Project
# This folder is updated at Github repository named wm-common
# import sys
# sys.path.append('/home/lovelace/proj/proj939/rubenscp/research/white-mold-application/commonxxxxxxxxxxxxxx')

# Basic python and ML Libraries
import os
from datetime import datetime
import shutil
import pandas as pd 

# torchvision libraries
import torch 
import torchvision 
from torch.utils.data import DataLoader
import pytorch_lightning
from transformers import DetrImageProcessor
from functools import partial

# Import python code from White Mold Project 
from common.manage_log import *
from common.tasks import Tasks
from common.entity.AnnotationsStatistic import AnnotationsStatistic
from common.convert_pascal_voc_to_coco_format import *
# from create_yaml_file import * 
from model import *
from dataset import *
from train import *
from eval import *

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'
NEW_FILE = True

# ###########################################
# Application Methods
# ###########################################

# ###########################################
# Methods of Level 1
# ###########################################

def main():
    """
    Main method that perform training of the neural network model.

    All values of the parameters used here are defined in the external file "wm_model_yolo_v8_parameters.json".

    """

    # creating Tasks object 
    processing_tasks = Tasks()

    # setting dictionary initial parameters for processing
    full_path_project = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-model-detr'

    # getting application parameters 
    processing_tasks.start_task('Getting application parameters')
    parameters_filename = 'wm_model_detr_parameters.json'
    parameters = get_parameters(full_path_project, parameters_filename)
    processing_tasks.finish_task('Getting application parameters')

    # setting new values of parameters according of initial parameters
    processing_tasks.start_task('Setting input image folders')
    set_input_image_folders(parameters)
    processing_tasks.finish_task('Setting input image folders')

    # getting last running id
    processing_tasks.start_task('Getting running id')
    running_id = get_running_id(parameters)
    processing_tasks.finish_task('Getting running id')

    # setting output folder results
    processing_tasks.start_task('Setting result folders')
    set_results_folder(parameters)
    processing_tasks.finish_task('Setting result folders')
    
    # creating log file 
    processing_tasks.start_task('Creating log file')
    logging_create_log(
        parameters['training_results']['log_folder'], parameters['training_results']['log_filename']
    )
    processing_tasks.finish_task('Creating log file')
    
    logging_info('White Mold Research')
    logging_info('Training the DETR model' + LINE_FEED)

    logging_info(f'')
    logging_info(f'>> Set input image folders')
    logging_info(f'')
    logging_info(f'>> Get running id')
    logging_info(f'running id: {str(running_id)}')   
    logging_info(f'')
    logging_info(f'>> Set result folders')

    # creating yaml file with parameters used by Ultralytics
    # processing_tasks.start_task('Creating yaml file for ultralytics')
    # create_yaml_file_for_ultralytics(parameters)
    # processing_tasks.finish_task('Creating yaml file for ultralytics')

    # getting device CUDA
    processing_tasks.start_task('Getting device CUDA')
    device = get_device(parameters)
    processing_tasks.finish_task('Getting device CUDA')

    # creating new instance of parameters file related to current running
    processing_tasks.start_task('Saving processing parameters')
    save_processing_parameters(parameters_filename, parameters)
    processing_tasks.finish_task('Saving processing parameters')

    # copying pre-trained files 
    processing_tasks.start_task('Copying pre-trained files')
    copy_pretrained_model_files(parameters)
    processing_tasks.finish_task('Copying pre-trained files') 
    
    # converting Pascal VOC annotations to COCO format
    processing_tasks.start_task('Converting Pascal VOC annotations to COCO format')
    convert_pascal_voc_to_coco_format(parameters)
    processing_tasks.finish_task('Converting Pascal VOC annotations to COCO format')

    # loading datasets and dataloaders of image dataset for processing
    processing_tasks.start_task('Loading dataloaders of image dataset')
    dataset_train, dataset_valid, dataset_test, \
        dataloader_train, dataloader_valid, processor = get_datasets_and_dataloaders(parameters, device)
    processing_tasks.finish_task('Loading dataloaders of image dataset')

    # creating neural network model 
    processing_tasks.start_task('Creating neural network model')
    model = get_neural_network_model(parameters, device, dataloader_train, dataloader_valid)
    processing_tasks.finish_task('Creating neural network model')

    # getting statistics of input dataset
    if parameters['processing']['show_statistics_of_input_dataset']:
        processing_tasks.start_task('Getting statistics of input dataset')
        annotation_statistics = get_input_dataset_statistics(parameters)
        show_input_dataset_statistics(parameters, annotation_statistics)
        processing_tasks.finish_task('Getting statistics of input dataset')

    # training neural netowrk model
    processing_tasks.start_task('Training neural netowrk model')
    model = train_detr_model(parameters, device, model, dataloader_train, dataloader_valid)
    processing_tasks.finish_task('Training neural netowrk model')

    # training neural netowrk model
    processing_tasks.start_task('Evaluation neural netowrk model')
    model = validate_detr_model(parameters, device, model, processor, dataset_valid, dataloader_valid)
    processing_tasks.finish_task('Evaluation neural netowrk model')

    # saving the best weights file 
    # processing_tasks.start_task('Saving best weights')
    # save_best_weights(parameters)
    # processing_tasks.finish_task('Saving best weights')

    # validating model 
    # processing_tasks.start_task('Validating model')
    # model = validate_yolo_v8_model(parameters, device, model)
    # processing_tasks.finish_task('Validating model')
    
    # showing input dataset statistics
    # if parameters['processing']['show_statistics_of_input_dataset']:
    #     show_input_dataset_statistics(annotation_statistics)

    # finishing model training 
    logging_info('')
    logging_info('Finished the training of the model DETR ' + LINE_FEED)

    # creating plot of training loss
    # create_plot_training_loss(parameters)    

    # printing tasks summary 
    processing_tasks.finish_processing()
    logging_info(processing_tasks.to_string())

    # copying processing files to log folder 
    # copy_processing_files_to_log(parameters)


# ###########################################
# Methods of Level 2
# ###########################################

def get_parameters(full_path_project, parameters_filename):
    '''
    Get dictionary parameters for processing
    '''    
    # getting parameters 
    path_and_parameters_filename = os.path.join(full_path_project, parameters_filename)
    parameters = Utils.read_json_parameters(path_and_parameters_filename)

    # returning parameters 
    return parameters


def set_input_image_folders(parameters):
    '''
    Set folder name of input images dataset
    '''    
    
    # getting image dataset folder according processing parameters 
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    image_dataset_folder = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['input']['input_dataset']['input_dataset_path'],
        parameters['input']['input_dataset']['annotation_format'],
        input_image_size + 'x' + input_image_size,
    )

    # setting image dataset folder in processing parameters 
    parameters['processing']['image_dataset_folder'] = image_dataset_folder
    parameters['processing']['image_dataset_folder_train'] = \
        os.path.join(image_dataset_folder, 'train')
    parameters['processing']['image_dataset_folder_valid'] = \
        os.path.join(image_dataset_folder, 'valid')
    parameters['processing']['image_dataset_folder_test'] = \
        os.path.join(image_dataset_folder, 'test')


def get_running_id(parameters):
    '''
    Get last running id to calculate the current id
    '''    

    # setting control filename 
    running_control_filename = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        parameters['processing']['running_control_filename'],
    )

    # getting control info 
    running_control = Utils.read_json_parameters(running_control_filename)

    # calculating the current running id 
    running_control['last_running_id'] = int(running_control['last_running_id']) + 1

    # updating running control file 
    running_id = int(running_control['last_running_id'])

    # saving file 
    Utils.save_text_file(running_control_filename, \
                         Utils.get_pretty_json(running_control), 
                         NEW_FILE)

    # updating running id in the processing parameters 
    parameters['processing']['running_id'] = running_id
    parameters['processing']['running_id_text'] = 'running-' + f'{running_id:04}'

    # returning the current running id
    return running_id

def set_results_folder(parameters):
    '''
    Set folder name of output results
    '''

    # resetting test results 
    parameters['test_results'] = {}

    # creating results folders 
    main_folder = os.path.join(
        parameters['processing']['research_root_folder'],     
        parameters['training_results']['main_folder']
    )
    parameters['training_results']['main_folder'] = main_folder
    Utils.create_directory(main_folder)

    # setting and creating model folder 
    parameters['training_results']['model_folder'] = parameters['neural_network_model']['model_name']
    model_folder = os.path.join(
        main_folder,
        parameters['training_results']['model_folder']
    )
    parameters['training_results']['model_folder'] = model_folder
    Utils.create_directory(model_folder)

    # setting and creating experiment folder
    experiment_folder = os.path.join(
        model_folder,
        parameters['input']['experiment']['id']
    )
    parameters['training_results']['experiment_folder'] = experiment_folder
    Utils.create_directory(experiment_folder)

    # setting and creating action folder of training
    action_folder = os.path.join(
        experiment_folder,
        parameters['training_results']['action_folder']
    )
    parameters['training_results']['action_folder'] = action_folder
    Utils.create_directory(action_folder)

    # setting and creating running folder 
    running_id = parameters['processing']['running_id']
    running_id_text = 'running-' + f'{running_id:04}'
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    parameters['training_results']['running_folder'] = running_id_text + "-" + input_image_size + 'x' + input_image_size   
    running_folder = os.path.join(
        action_folder,
        parameters['training_results']['running_folder']
    )
    parameters['training_results']['running_folder'] = running_folder
    Utils.create_directory(running_folder)

    # setting and creating others specific folders
    processing_parameters_folder = os.path.join(
        running_folder,
        parameters['training_results']['processing_parameters_folder']
    )
    parameters['training_results']['processing_parameters_folder'] = processing_parameters_folder
    Utils.create_directory(processing_parameters_folder)

    pretrained_model_folder = os.path.join(
        running_folder,
        parameters['training_results']['pretrained_model_folder']
    )
    parameters['training_results']['pretrained_model_folder'] = pretrained_model_folder
    Utils.create_directory(pretrained_model_folder)
    
    weights_folder = os.path.join(
        running_folder,
        parameters['training_results']['weights_folder']
    )
    parameters['training_results']['weights_folder'] = weights_folder
    Utils.create_directory(weights_folder)

    # setting the base filename of weights
    weights_base_filename = parameters['neural_network_model']['model_name'] + '-' + \
                            running_id_text + "-" + input_image_size + 'x' + input_image_size + '.pth'
    parameters['training_results']['weights_base_filename'] = weights_base_filename

    metrics_folder = os.path.join(
        running_folder,
        parameters['training_results']['metrics_folder']
    )
    parameters['training_results']['metrics_folder'] = metrics_folder
    Utils.create_directory(metrics_folder)

    log_folder = os.path.join(
        running_folder,
        parameters['training_results']['log_folder']
    )
    parameters['training_results']['log_folder'] = log_folder
    Utils.create_directory(log_folder)

    results_folder = os.path.join(
        running_folder,
        parameters['training_results']['results_folder']
    )
    parameters['training_results']['results_folder'] = results_folder
    Utils.create_directory(results_folder)

    # setting folder of YOLO models 
    # model_folder = os.path.join(
    #     parameters['processing']['research_root_folder'],
    #     parameters['processing']['project_name_folder'],
    #     parameters['neural_network_model']['model_folder'],
    # )
    # parameters['neural_network_model']['model_folder'] = model_folder
    
# def create_yaml_file_for_ultralytics(parameters):

#     # preparing parameters 
#     yolo_v8_yaml_filename = parameters['processing']['yolo_v8_yaml_filename_train']
#     path_and_filename_white_mold_yaml = os.path.join(
#         parameters['processing']['research_root_folder'],
#         parameters['processing']['project_name_folder'],
#         yolo_v8_yaml_filename
#     )
#     image_dataset_folder = parameters['processing']['image_dataset_folder']
#     number_of_classes = parameters['neural_network_model']['number_of_classes']
#     classes = (parameters['neural_network_model']['classes'])[:(number_of_classes+1)]
    
#     # creating yaml file 
#     create_project_yaml_file_for_train_valid(
#         path_and_filename_white_mold_yaml,
#         image_dataset_folder,
#         classes,    
#     )

def get_device(parameters):
    '''
    Get device CUDA to train models
    '''    

    logging_info(f'')
    logging_info(f'>> Get device')

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    parameters['processing']['device'] = f'{device}'

    logging_info(f'Device: {device}')

    # returning current device 
    return device     

def save_processing_parameters(parameters_filename, parameters):
    '''
    Update parameters file of the processing
    '''    
    # setting full path and log folder  to write parameters file 
    path_and_parameters_filename = os.path.join(
        parameters['training_results']['processing_parameters_folder'], 
        parameters_filename)

    # saving current processing parameters in the log folder 
    Utils.save_text_file(path_and_parameters_filename, \
                        Utils.get_pretty_json(parameters), 
                        NEW_FILE)

def copy_pretrained_model_files(parameters):
    '''
    Copy pretrained model files for this running
    '''    

    input_path = parameters['neural_network_model']['pretrained_model_path']
    output_path = parameters['training_results']['pretrained_model_folder']

    filename = 'config.json'
    Utils.copy_file_same_name(filename, input_path, output_path)
    filename = 'model.safetensors'
    Utils.copy_file_same_name(filename, input_path, output_path)
    filename = 'preprocessor_config.json'
    Utils.copy_file_same_name(filename, input_path, output_path)
    filename = 'pytorch_model.bin'
    Utils.copy_file_same_name(filename, input_path, output_path)

def convert_pascal_voc_to_coco_format(parameters):
    # checking if converts annotations from Pascal VOC to Coco detection format
    if not parameters['input']['input_dataset']['convert_pascal_voc_to_coco_format']:
        return

    # base folders 
    research_root_folder = parameters['processing']['research_root_folder']
    input_dataset_path = parameters['input']['input_dataset']['input_dataset_path']

    # input folders 
    pascal_voc_folder = os.path.join('ssd_pascal_voc', '300x300')
    train_input_folder = os.path.join(research_root_folder, input_dataset_path, 
                                      pascal_voc_folder, 'train')
    valid_input_folder = os.path.join(research_root_folder, input_dataset_path,
                                      pascal_voc_folder, 'valid')
    test_input_folder = os.path.join(research_root_folder, input_dataset_path,
                                      pascal_voc_folder, 'test')

    # output folders
    coco_folder = os.path.join('coco_detection_json', '300x300')
    train_output_folder = os.path.join(research_root_folder, input_dataset_path, 
                                      coco_folder, 'train')
    valid_output_folder = os.path.join(research_root_folder, input_dataset_path,
                                      coco_folder, 'valid')
    test_output_folder = os.path.join(research_root_folder, input_dataset_path,
                                      coco_folder, 'test')

    # creating outfolders 
    Utils.create_directory(train_output_folder)
    Utils.create_directory(valid_output_folder)
    Utils.create_directory(test_output_folder)

    # parser = argparse.ArgumentParser(
    #     description='This script support converting voc format xmls to coco format json')
    # parser.add_argument('--ann_dir', type=str, default=None,
    #                     help='path to annotation files directory. It is not need when use --ann_paths_list')
    # parser.add_argument('--ann_ids', type=str, default=None,
    #                     help='path to annotation files ids list. It is not need when use --ann_paths_list')
    # parser.add_argument('--ann_paths_list', type=str, default=None,
    #                     help='path of annotation paths list. It is not need when use --ann_dir and --ann_ids')
    # parser.add_argument('--labels', type=str, default=None,
    #                     help='path to label list.')
    # parser.add_argument('--output', type=str, default='output.json', help='path to output json file')
    # parser.add_argument('--ext', type=str, default='', help='additional extension of annotation file')
    # args = parser.parse_args()
    
    # getting dictionary of labels(classes) and its ids
    label2id_dict = get_label2id_list(labels_list=parameters['neural_network_model']['classes'])
    logging_info(f'label2id_dict: {label2id_dict}')
    logging_info(f'')

    # convert annotations of the training dataset
    copy_images_to_folder(train_input_folder, train_output_folder)
    train_ann_paths = get_annpaths(annpaths_list_path=train_input_folder)
    logging_info(f'Annotation files of training dataset  : {len(train_ann_paths)}')
    train_output_file = os.path.join(train_output_folder, 'custom_train.json')
    convert_xmls_to_cocojson(
        annotation_paths=train_ann_paths,
        label2id=label2id_dict,
        output_folder=train_output_folder,
        output_jsonpath=train_output_file,
        extract_num_from_imgid=False
    )

    # convert annotations of the validation dataset
    copy_images_to_folder(valid_input_folder, valid_output_folder)
    valid_ann_paths = get_annpaths(annpaths_list_path=valid_input_folder)
    logging_info(f'Annotation files of validation dataset: {len(valid_ann_paths)}')
    valid_output_file = os.path.join(valid_output_folder, 'custom_valid.json')
    convert_xmls_to_cocojson(
        annotation_paths=valid_ann_paths,
        label2id=label2id_dict,
        output_folder=valid_output_folder,
        output_jsonpath=valid_output_file,
        extract_num_from_imgid=False
    )

    # convert annotations of the training dataset
    copy_images_to_folder(test_input_folder, test_output_folder)
    test_ann_paths = get_annpaths(annpaths_list_path=test_input_folder)
    logging_info(f'Annotation files of test dataset      : {len(test_ann_paths)}')
    test_output_file = os.path.join(test_output_folder, 'custom_test.json')
    convert_xmls_to_cocojson(
        annotation_paths=test_ann_paths,
        label2id=label2id_dict,
        output_folder=test_output_folder,
        output_jsonpath=test_output_file,
        extract_num_from_imgid=False
    )

    total_ann_paths = len(train_ann_paths) + len(valid_ann_paths) + len(test_ann_paths)
    logging_info(f'Annotation files of full dataset      : {total_ann_paths}')

def get_datasets_and_dataloaders(parameters, device):
    '''
    Get datasets and dataloaders of training, validation and testing from image dataset 
    '''

    # getting image dataset folders
    image_dataset_folder_train = parameters['processing']['image_dataset_folder_train']
    image_dataset_folder_valid = parameters['processing']['image_dataset_folder_valid']
    image_dataset_folder_test = parameters['processing']['image_dataset_folder_test']

    # setting parameters for training and validation datasets
    pretrained_model_name_or_path = parameters['neural_network_model']['pretrained_model_path']
    cache_dir = parameters['neural_network_model']['model_cache_dir']

    # getting image processor
    processor = DetrImageProcessor.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            revision="no_timm",
            local_files_only=True,            
        )
    logging.info(f'processor: {processor}')

    # getting datasets for training and validation
    dataset_train = CocoDetection(img_folder=image_dataset_folder_train, processor=processor, dataset_type='train')
    dataset_valid = CocoDetection(img_folder=image_dataset_folder_valid, processor=processor, dataset_type='valid')
    dataset_test  = CocoDetection(img_folder=image_dataset_folder_test, processor=processor, dataset_type='test')

    logging.info(f'Getting datasets')
    logging.info(f'Number of training images  : {len(dataset_train)}')
    logging.info(f'Number of validation images: {len(dataset_valid)}')
    logging.info(f'Number of testing images   : {len(dataset_test)}')
    logging.info(f'Total of images            : {len(dataset_train) + len(dataset_valid) + len(dataset_test)}')
    logging_info(f'')

    # creating dataloaders 
    batch_size = parameters['neural_network_model']['batch_size']
    num_workers = parameters['neural_network_model']['number_workers']
    dataloader_train = DataLoader(
        dataset_train, 
        collate_fn=partial(collate_fn, processor=processor), 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True)
    dataloader_valid = DataLoader(
        dataset_valid, 
        collate_fn=partial(collate_fn, processor=processor), 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)

    logging.info(f'Getting dataloaders')
    logging.info(f'Number of training loaders   : {len(dataloader_train)}')
    logging.info(f'Number of validation loaders : {len(dataloader_valid)}')
    logging_info(f'')

    logging_info(f'Details train batch')
    batch = next(iter(dataloader_train))
    logging_info(f'dataloader_train - batch.keys: {batch.keys()}')
    logging_info(f'')

    logging_info(f'Details valid batch')
    batch = next(iter(dataloader_valid))
    logging_info(f'dataloader_valid - batch.keys: {batch.keys()}')
    logging_info(f'')

    pixel_values, target = dataset_train[0]
    logging_info(f'Details dataset_train[0]')
    logging_info(f'pixel_values.shape: {pixel_values.shape}')
    logging_info(f'target: {target}')   
    logging_info(f'')

    pixel_values, target = dataset_valid[0]
    logging_info(f'Details dataset_valid[0]')
    logging_info(f'pixel_values.shape: {pixel_values.shape}')
    logging_info(f'target: {target}')   
    logging_info(f'')

    # returning dataloaders from datasets for processing 
    return dataset_train, dataset_valid, dataset_test, dataloader_train, dataloader_valid, processor

def get_neural_network_model(parameters, device, dataloader_train, dataloader_valid):
    '''
    Get neural network model
    '''      
    
    logging_info(f'')
    logging_info(f'>> Get neural network model')
    
    model_name = parameters['neural_network_model']['model_name']

    logging_info(f'Model used: {model_name}')

    learning_rate = parameters['neural_network_model']['learning_rate_initial']
    learning_rate_backbone = parameters['neural_network_model']['learning_rate_backbone']
    weight_decay = parameters['neural_network_model']['weight_decay']
    num_labels = parameters['neural_network_model']['number_of_classes']
    pretrained_model_name_or_path = parameters['neural_network_model']['pretrained_model_path']
    cache_dir = parameters['neural_network_model']['model_cache_dir']
    model = Detr(lr=learning_rate, 
                 lr_backbone=learning_rate_backbone, 
                 weight_decay=weight_decay,
                 pretrained_model_name_or_path=pretrained_model_name_or_path,
                 cache_dir=cache_dir,
                 num_labels=num_labels,
                 train_dataloader=dataloader_train,
                 val_dataloader=dataloader_valid)

    logging.info(f'{model}')
    # logging.info(f'')
    # logging.info(f'next(model.parameters()).device')
    # logging_info(next(model.parameters()).device)
    # logging_info(f'next(model.parameters()).is_cuda: {next(model.parameters()).is_cuda}')
    # logging.info(f'logging model.cuda()')
    # logging.info(f'model.cuda(): {model.cuda()}')

    # moving model into GPU 
    logging_info(f'Status of GPU')

    # logging_info(f'ANTES')
    # logging_info(f'torch.__version__: {torch.__version__}')
    # logging_info(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    
    # logging.info(f'Moving model to device: {device}')
    # moving model to GPU 
    model = model.to(device)

    # logging_info(f'DEPOIS ')
    logging_info(f'torch.__version__: {torch.__version__}')
    logging_info(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    logging.info(f'next(model.parameters()).device')
    logging_info(next(model.parameters()).device)
    logging_info(f'next(model.parameters()).is_cuda: {next(model.parameters()).is_cuda}')

    # returning neural network model
    return model

# getting statistics of input dataset 
def get_input_dataset_statistics(parameters):
    
    annotation_statistics = AnnotationsStatistic()
    steps = ['train', 'valid', 'test'] 
    annotation_statistics.processing_statistics(parameters, steps)
    return annotation_statistics

def show_input_dataset_statistics(parameters, annotation_statistics):

    logging_info(f'Input dataset statistic')
    logging_info(annotation_statistics.to_string())
    path_and_filename = os.path.join(
        parameters['training_results']['metrics_folder'],
        parameters['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_annotations_statistics.xlsx',
    )
    annotation_format = parameters['input']['input_dataset']['annotation_format']
    input_image_size = parameters['input']['input_dataset']['input_image_size']
    classes = (parameters['neural_network_model']['classes'])[1:5]
    annotation_statistics.save_annotations_statistics(
        path_and_filename,
        annotation_format,
        input_image_size,
        classes
    )

def train_detr_model(parameters, device, model, dataloader_train, dataloader_valid):
    '''
    Execute training of the neural network model
    '''

    logging_info(f'')
    logging_info(f'>> Training model')   

    # Train the model
    model = training_the_model(parameters, model, dataloader_train, dataloader_valid)
    logging_info(f'Training of the model finished')

    # returning trained model
    return model 

def validate_detr_model(parameters, device, model, processor, dataset_valid, dataloader_valid):
    '''
    Execute evaluation of the neural network model
    '''

    logging_info(f'')
    logging_info(f'>> Evaluation of the model')

    # moving model to GPU 
    # logging.info(f'ANTES ')
    # logging.info(f'next(model.parameters()).device')
    # logging_info(next(model.parameters()).device)
    # logging_info(f'next(model.parameters()).is_cuda: {next(model.parameters()).is_cuda}')
    model = model.to(device)
    # logging.info(f'DEPOIS')
    # logging.info(f'next(model.parameters()).device')
    # logging_info(next(model.parameters()).device)
    # logging_info(f'next(model.parameters()).is_cuda: {next(model.parameters()).is_cuda}')

    # Validating the model
    model = validate_the_model(parameters, device, model, processor, dataset_valid, dataloader_valid)
    logging_info(f'Evaluation of the model finished')

    # returning trained model
    # return model 

# def save_best_weights(parameters):

#     # setting folders and filenames for weights file 
#     input_filename  = 'best.pt'
#     input_path      = os.path.join(parameters['training_results']['results_folder'], 'train/weights')
#     output_filename =  parameters['training_results']['weights_base_filename']
#     output_path     = parameters['training_results']['weights_folder']
#     logging_info(f'input_filename: {input_filename}')
#     logging_info(f'input_path: {input_path}')
#     logging_info(f'output_filename: {output_filename}')
#     logging_info(f'output_path: {output_path}')

#     # copying weights file to be used in the inference step 
#     Utils.copy_file(input_filename, input_path, output_filename, output_path)

# def validate_yolo_v8_model(parameters, device, model):
#     '''
#     Execute validation in the model previously trained
#     '''

#     logging_info(f'')
#     logging_info(f'>> Validating model')   

#     # Train the model using the 'coco128.yaml' dataset for 3 epochs
#     # data_file_yaml = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-model-detr/white_mold_yolo_v8.yaml'    
#     data_file_yaml = os.path.join(
#         parameters['processing']['research_root_folder'],
#         parameters['processing']['project_name_folder'],
#         parameters['processing']['yolo_v8_yaml_filename_train']
#     )
#     metrics = model.val(
#         data=data_file_yaml, 
#         imgsz=parameters['input']['input_dataset']['input_image_size'],
#         project=parameters['training_results']['results_folder'],
#         conf=parameters['neural_network_model']['threshold'],
#         iou=parameters['neural_network_model']['iou_threshold_for_validation'],
#         device=device,
#         verbose=True, 
#         show=True,
#         save=True,
#         save_conf=True,
#         plots=True,
#     )

#     # metrics = model.val(save_json=True, save_hybrid=True) 
#     logging_info(f'metrics: {metrics}')
#     # logging_info(f'box.map: {metrics.box.map}')
#     # logging_info(f'box.map50: {metrics.box.map50}')
#     # logging_info(f'box.map75: {metrics.box.map75}')
#     # logging_info(f'box.maps: {metrics.box.maps}')
    
#     logging_info(f'Validating of the model finished')
    
#     # returing trained model
#     return model

# def copy_processing_files_to_log(parameters):
#     input_path = os.path.join(
#         parameters['processing']['research_root_folder'],
#         parameters['processing']['project_name_folder'],
#     )
#     output_path = parameters['training_results']['log_folder']    
#     input_filename = output_filename = 'yolo_v8_train_errors'
#     Utils.copy_file(input_filename, input_path, output_filename, output_path)

#     input_filename = output_filename = 'yolo_v8_train_output'
#     Utils.copy_file(input_filename, input_path, output_filename, output_path)


def create_plot_training_loss(parameters):
    '''
    Create plot of training loss.

    This function reads the training loss results from a CSV file, 
    creates a plot of the training loss over epochs, and saves the plot 
    as an image file. It also saves the training loss values in an Excel file.

    Parameters:
    - parameters (dict): A dictionary containing various parameters for the function.

    Returns:
    - None
    '''

    # Construct the path and filename for the results CSV file
    path_and_filename = os.path.join(
        parameters['training_results']['results_folder'],
        'train',
        'results.csv'
    )

    # Log the path and filename
    logging_info(f'path_and_filename: {path_and_filename}')    

    # Read the results CSV file into a DataFrame
    results_df = pd.read_csv(path_and_filename)

    # Remove left spaces from the column names
    results_df.columns = results_df.columns.str.replace(' ','');

    # Get the training losses
    epochs = results_df['epoch'].tolist()
    losses_per_epoch = results_df['train/box_loss'].tolist()
    train_loss_list_excel = []
    for epoch, loss in enumerate(losses_per_epoch):
        train_loss_list_excel.append([epoch+1, loss])

    # Construct the path and filename for the training loss plot
    path_and_filename = os.path.join(
        parameters['training_results']['metrics_folder'],     
        parameters['neural_network_model']['model_name'] + \
                    '_train_loss.png'
    )
    title = f'Training Loss for model {parameters["neural_network_model"]["model_name"]}'
    x_label = "Epochs"
    y_label = "Train Loss"

    # Log the path and filename
    logging_info(f'path_and_filename: {path_and_filename}')

    # Save the training loss plot
    Utils.save_plot(losses_per_epoch, path_and_filename, title, x_label, y_label)

    # Log the final train loss list
    logging_info(f'train_loss_list_excel final: {train_loss_list_excel}')    

    # Construct the path and filename for the training loss Excel file
    path_and_filename = os.path.join(
        parameters['training_results']['metrics_folder'],
        parameters['neural_network_model']['model_name'] + '_train_loss.xlsx'
    )

    # Log the path and filename
    logging_info(f'path_and_filename: {path_and_filename}')

    # Save the training loss list to an Excel file
    Utils.save_losses(train_loss_list_excel, path_and_filename)

# ###########################################
# Methods of Level 3
# ###########################################

# ###########################################
# Main method
# ###########################################
if __name__ == '__main__':
    main()
