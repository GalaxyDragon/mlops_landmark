import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


from config import ROOT_DIR, TRAIN_DIR
from config import CONFIG, TRAIN_TRANSFORM, VALID_TRANSFORM
from utils import get_train_file_path

df = pd.read_csv(f"{ROOT_DIR}/train.csv")

def split_datasets():

    df['file_path'] = df['id'].apply(get_train_file_path)

    df_train, df_test = train_test_split(df, test_size=0.4, stratify=df.landmark_id,
                                         shuffle=True, random_state=CONFIG['seed'])
    df_valid, df_test = train_test_split(df_test, test_size=0.5, shuffle=True,
                                         random_state=CONFIG['seed'])
    return df_train, df_test, df_valid

class LandmarkDataset(Dataset):
    def __init__(self, root_dir, df, transforms=None):
        self.root_dir = root_dir
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df['landmark_id'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return img, label



def prepare_loaders():

    df_train, df_test, df_valid = split_datasets()

    data_transforms = {
        "train": TRAIN_TRANSFORM,

        "valid": VALID_TRANSFORM
    }

    train_dataset = LandmarkDataset(TRAIN_DIR, df_train, transforms=data_transforms['train'])

    valid_dataset = LandmarkDataset(TRAIN_DIR, df_valid, transforms=data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'],
                              num_workers=4, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'],
                              num_workers=4, shuffle=False, pin_memory=True)

    return train_loader, valid_loader
