from parsers.config_parser import config_parser
import os

config_path = './config.ini'
section = 'path'
config_param_data_dir = 'data_dir'
data_dir = config_parser(config_path, section,config_param_data_dir)

dataset_name_template_string = "eng.{}"
dataset_small_template_string = "eng.{}.small"
train_set = "train"
dev_set = "dev"
test_set = "test"

def get_train_set_path():
    return os.path.join(data_dir, dataset_name_template_string.format(train_set))

def get_dev_set_path():
    return os.path.join(data_dir, dataset_name_template_string.format(dev_set))

def get_test_set_path():
    return os.path.join(data_dir, dataset_name_template_string.format(test_set))

def get_small_train_set_path():
    return os.path.join(data_dir, dataset_small_template_string.format(train_set))

def get_small_dev_set_path():
    return os.path.join(data_dir, dataset_small_template_string.format(dev_set))

def get_small_test_set_path():
    return os.path.join(data_dir, dataset_small_template_string.format(test_set))
