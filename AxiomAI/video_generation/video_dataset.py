import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from tokenizer.my_tokenizer import CharTokenizer

class VideoDataset(Dataset):
    """
    Loads .mp4 and .txt pairs from the data/Videos/ directory.
    Supports OpenCV for frame extraction and normalization.
    """
    def __init__(self, config, tokenizer_path, width=None, height=None, max_caption_len=None):
        self.data_path = config.get('video_data_path', 'data/Videos/')
        self.width = int(width if width is not None else config.get('width', 64))
        self.height = int(height if height is not None else config.get('height', 64))
        self.max_caption_len = int(max_caption_len) if max_caption_len is not None else None
        self.strict_loading = str(config.get('strict_video_loading', 'True')).strip().lower() in ('1', 'true', 'yes', 'on')
        
        # User-friendly real-time parsing
        fps = float(config.get('fps', 8.0))
        duration = float(config.get('duration', 2.0))
        self.num_frames = int(fps * duration)
        
        # Augmentation parameters
        self.flip_prob = float(config.get('flip_prob', 0.5))
        self.crop_pad_percent = float(config.get('crop_pad_percent', 0.1))
        
        self.tokenizer = CharTokenizer()
        if not self.tokenizer.load(tokenizer_path):
            raise ValueError(f"Could not load tokenizer from {tokenizer_path}")
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(f"Video data directory not found: {self.data_path}")

        # Scan for video files, strictly enforcing associated caption files
        all_videos = sorted(f for f in os.listdir(self.data_path) if f.lower().endswith('.mp4'))
        self.video_files = []
        for f in all_videos:
            if os.path.exists(os.path.join(self.data_path, self._caption_name(f))):
                self.video_files.append(f)
            else:
                print(f"WARNING: Skipping {f} - Missing associated .txt caption.")
        self.caption_cache = {}
                
        print(f"Dataset initialized with {len(self.video_files)} validated video-text pairs found in {self.data_path}")

    def __len__(self):
        return len(self.video_files)

    def video_path(self, video_name):
        return os.path.join(self.data_path, video_name)

    def caption_path(self, video_name):
        return os.path.join(self.data_path, self._caption_name(video_name))

    def validate_files(self):
        problems = []
        for video_name in self.video_files:
            path = self.video_path(video_name)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                problems.append(f"{video_name}: OpenCV could not open video.")
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count <= 0:
                problems.append(f"{video_name}: no readable frames.")
            if frame_count < self.num_frames:
                problems.append(f"{video_name}: only {frame_count} frames, needs {self.num_frames}; will pad with last frame.")

            caption_path = self.caption_path(video_name)
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    if not f.read().strip():
                        problems.append(f"{video_name}: caption is empty.")
            except OSError as e:
                problems.append(f"{video_name}: caption read failed: {e}")

        return problems

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video file: {path}")
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine sampling stride to get exactly num_frames
        stride = max(1, total_frames // self.num_frames)
        
        # Calculate consistent spatial augmentations for this entire sequence
        flip = np.random.rand() < self.flip_prob
        pad_w = int(self.width * self.crop_pad_percent)
        pad_h = int(self.height * self.crop_pad_percent)
        target_w = self.width + pad_w
        target_h = self.height + pad_h
        
        start_x = np.random.randint(0, max(1, pad_w + 1))
        start_y = np.random.randint(0, max(1, pad_h + 1))
        
        count = 0
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % stride == 0:
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize slightly larger
                frame = cv2.resize(frame, (target_w, target_h))
                # Crop consistently down to target dims
                frame = frame[start_y:start_y+self.height, start_x:start_x+self.width]
                # Flip consistently
                if flip:
                    frame = cv2.flip(frame, 1)
                    
                frames.append(frame)
            count += 1
            
        cap.release()
        
        # If too few frames, pad with last frame
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.height, self.width, 3), dtype=np.uint8))
            
        # Stack and normalize to [-1, 1]
        video = np.stack(frames, axis=0) # (T, H, W, 3)
        video = (video.astype(np.float32) / 127.5) - 1.0
        return torch.from_numpy(video).permute(3, 0, 1, 2) # (C, T, H, W)

    def _caption_name(self, video_name):
        return os.path.splitext(video_name)[0] + '.txt'

    def _load_caption(self, video_name):
        if video_name in self.caption_cache:
            return self.caption_cache[video_name]

        caption_path = self.caption_path(video_name)
        if not os.path.exists(caption_path):
            raise FileNotFoundError(f"Missing caption: {caption_path}")
            
        with open(caption_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            
        tokens = self.tokenizer.encode(text)
        if self.max_caption_len is not None and self.max_caption_len > 0:
            tokens = tokens[:self.max_caption_len]
        if not tokens:
            tokens = [self.tokenizer.pad_id]
        caption_tokens = torch.tensor(tokens, dtype=torch.long)
        self.caption_cache[video_name] = caption_tokens
        return caption_tokens

    def __getitem__(self, idx):
        v_name = self.video_files[idx]
        try:
            video = self._load_video(self.video_path(v_name))
            caption_tokens = self._load_caption(v_name)
        except Exception as e:
            if self.strict_loading:
                raise
            print(f"Error loading {v_name}: {e}")
            # Return a blank sample on error to avoid crashing the training loop
            video = torch.zeros(3, self.num_frames, self.height, self.width)
            caption_tokens = torch.tensor([self.tokenizer.pad_id], dtype=torch.long)
            
        return video, caption_tokens
