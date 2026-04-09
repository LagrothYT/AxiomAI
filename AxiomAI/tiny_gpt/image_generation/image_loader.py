import os
import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    import torchvision.transforms as T
except ImportError:
    pass

class ImageDataset(Dataset):
    def __init__(self, data_dir, config):
        self.data_dir = data_dir
        self.config = config
        self.samples = []
        
        # Transform ensures images match correct math sizes and format
        self.transform = T.Compose([
            T.Resize((self.config.image_height, self.config.image_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Crawl Photos directory
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                
                # Setup recursive check for sub-directories vs root files
                if os.path.isdir(item_path):
                    files_to_check = [(f, os.path.join(item_path, f), item) for f in os.listdir(item_path)]
                else:
                    files_to_check = [(item, item_path, os.path.splitext(item)[0])]
                    
                for fname, img_path, default_caption in files_to_check:
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        base_name = os.path.splitext(fname)[0]
                        parent_dir = os.path.dirname(img_path)
                        txt_path = os.path.join(parent_dir, base_name + '.txt')
                        
                        caption = default_caption
                        if os.path.exists(txt_path):
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                caption = f.read().strip()
                                
                        self.samples.append((img_path, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        
        try:
            image = Image.open(img_path)
            # Crucial: Automaticaly convert grayscale (L) to RGB (3-channel)
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            img_tensor = self.transform(image)
        except Exception as e:
            img_tensor = torch.zeros(3, self.config.image_height, self.config.image_width)
            print(f"[!] Warning: Could not load image {img_path}. Error: {e}")
            
        return img_tensor, caption
