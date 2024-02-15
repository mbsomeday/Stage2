import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOCAL = {
    'dataset_base_dir': r'D:\my_phd\dataset',
    'weights_save_path': r'D:\my_phd\Model_Weights\Stage2',
    # 格式为 model-dataset
    'weights_path': {
        'vgg-ECPD': r'D:\my_phd\Model_Weights\Stage2\vgg-ECPD-050-0.97701919.pth'
    },

    'D1_ECPDaytime': {
        'base_dir': r'D:\my_phd\dataset\D1_ECPDaytime',
        'txt_dir': 'dataset_txt'
    },
    'D2_CityPersons': {
        'base_dir': r'D:\my_phd\dataset\D2_CityPersons',
        'txt_dir': 'dataset_txt'
    },
    'D3_ECPNight': {
        'base_dir': r'D:\my_phd\dataset\D3_ECPNight',
        'image_dir': r'D:\my_phd\dataset\D3_ECPNight\bdd100k\images\100k',
    },
    'D4_BDD100K': {
        'base_dir': r'D:\my_phd\dataset\D4_BDD100K',
        'image_dir': r'D:\my_phd\dataset\D4_BDD100K\bdd100k\images\100k',
    }
}

CLOUD = {
    'dataset_base_dir': r'D:\my_phd\dataset',
    'weights_save_path': r'/content/weights_path',
    'weights_path': {

    },
    'D1_ECPDaytime': {
        'base_dir': r'/content/D1_ECPDaytime',
        'txt_dir': 'dataset_txt'
    },
    'D2_CityPersons': {
        'base_dir': r'/content/D2_CityPersons',
        'txt_dir': 'dataset_txt'
    },
    'D3_ECPNight': {
        'base_dir': r'/content/D3_ECPNight',
        'txt_dir': 'dataset_txt'
    },
    'D4_BDD100K': {
        'base_dir': r'/content/D4_BDD100K',
        'txt_dir': 'dataset_txt'
    }
}














