import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOCAL = {
    'weights_save_path': r'D:\my_phd\Model_Weights\Stage2',
    # 格式为 model-dataset
    'weights_path': {
        'vgg-ECPD': r'D:\my_phd\Model_Weights\Stage2\vgg-ECPD-050-0.97701919.pth'
    },
    'D1': {
        'base_dir': r'D:\chrom_download\grouped2_ECP',
        'image_dir': '',
    },
    'D2': {

    },
    'D3': {
        'base_dir': r'D:\my_phd\dataset\D3_BDD',
        'image_dir': r'D:\my_phd\dataset\D3_BDD\bdd100k\images\100k',
    },
    'D4': {

    }
}

CLOUD = {
    'weights_save_path': r'/content/weights_path',
    'weights_path': {

    },
    'D1': {
        'base_dir': r'/content/data',
        'txt_dir': 'dataset_txt'
    },
    'D2': {

    },
    'D3': {

    },
    'D4': {

    }
}














