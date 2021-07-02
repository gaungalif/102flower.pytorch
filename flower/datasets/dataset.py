import json

from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image
from typing import *

class FlowerDataset(Dataset):
    def __init__(self, root: str, jsfile: str, ext: str = "jpg", transform: Optional[Callable] = None):
        self.root = Path(root)
        self.jsfile = Path(jsfile)
        self.ext = ext
        self.transform = transform
        
        self._load_files()
        self._load_classes()
        
    def _load_files(self):
        self.files = sorted(list(self.root.glob(f"*/*.{self.ext}")))
    
    def _load_classes(self):
        cat2name = self._load_json(str(self.jsfile))
        idx2class, class2idx, data_dict = [], {}, {}

        for k,v in cat2name.items(): data_dict[v] = int(k)
        list_pair = sorted(data_dict.items(), key=lambda x: x[1])
        for (name,cat) in list_pair:
            class2idx[name] = cat - 1
            idx2class.append(name)
        
        self.class2idx = class2idx
        self.idx2class = idx2class

    def _load_json(self, path):
        with open(path, 'r') as jsfile:
            data = json.load(jsfile)
        return data
    
    def _load_image(self, path):
        img = Image.open(path)
        return img
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        impath = self.files[idx]
        img = self._load_image(str(impath))
        cat = int(impath.parent.name)
        
        if self.transform:
            img = self.transform(img)

        return img, cat
