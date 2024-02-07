import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOCAL = {
    'weights_path': {

    },
    'weights_save_path': r'D:\my_phd\Model_Weights\Stage2',
    'ECPD': {
        'base_dir': r'D:\chrom_download\grouped2_ECP',
        'txt_dir': 'dataset_txt'
    }
}

CLOUD = {
    'weights_path': {

    },
    'weights_save_path': r'/content/weights_path',
    'ECPD': {
        'base_dir': r'/content/data',
        'txt_dir': 'dataset_txt'
    }
}














