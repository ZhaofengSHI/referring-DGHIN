from loader.loader import Generator
from yacs.config import CfgNode as CN
import json
import os.path

def load_dataset(config, split):
    with open(config[split], 'rb') as f:
        data_lines = json.load(f)

    set_num = len(data_lines)
    print('Dataset Loaded: %s,  Len: %d' % (split, set_num))
    return data_lines, set_num


def load_data(config,train_mode = True, set = 'testB_set'): #test_set: evaluate_set testA_set testB_set test_set

    if train_mode:
        dataset_train, dataset_len_train = load_dataset(config,'train_set')
        dataset_val, dataset_len_val = load_dataset(config,set)

        train_generator = Generator(dataset_train, config, train_mode=True)
        val_generator = Generator(dataset_val,config, train_mode=False)

        return train_generator,val_generator

    else:
        dataset_val, dataset_len_val = load_dataset(config, set)
        val_generator = Generator(dataset_val, config, train_mode=False)

        return val_generator

def load_config(cfg_dir,dataset):
    #load and merge config

    with open(os.path.join(cfg_dir,'base.yaml'), 'r') as f:
        _C = CN.load_cfg(f)

    config_path = os.path.join(cfg_dir,dataset,'example.yaml')
    config = _C.clone()
    config.merge_from_file(config_path)
    config.freeze()

    return config