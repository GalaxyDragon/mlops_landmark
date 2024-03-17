from colorama import Fore, Style
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

ROOT_DIR = "../dataset"
TRAIN_DIR = "../dataset/train"
TEST_DIR = "../dataset/validation"
MODEL_PATH = "../models"

g_ = Fore.GREEN
c_ = Fore.CYAN
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

CONFIG = dict(
    seed = 42,
    model_name = 'tf_mobilenetv3_small_100',
    train_batch_size = 384,
    valid_batch_size = 768,
    img_size = 224,
    epochs = 1,
    learning_rate = 5e-4,
    scheduler = None,
    weight_decay = 1e-6,
    n_accumulate = 1,
    n_fold = 5,
    num_classes = 81313,
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    competition = 'GOOGL',
    _wandb_kernel = 'deb'
)
TRAIN_TRANSFORM = A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.)

VALID_TRANSFORM = A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.)