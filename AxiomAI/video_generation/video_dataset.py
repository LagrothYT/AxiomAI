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
        if self.num_frames < 1:
            raise ValueError(f"Video fps * duration must produce at least 1 frame. Current: {fps} * {duration} = {fps * duration}.")
        
        # Augmentation parameters
        self.flip_prob = float(config.get('flip_prob', 0.5))
        self.crop_pad_percent = float(config.get('crop_pad_percent', 0.1))
        if self.width < 1 or self.height < 1:
            raise ValueError("Video width and height must both be positive.")
        if self.flip_prob < 0.0 or self.flip_prob > 1.0:
            raise ValueError("Video flip_prob must be between 0.0 and 1.0.")
        if self.crop_pad_percent < 0.0:
            raise ValueError("Video crop_pad_percent must be >= 0.0.")
        
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

    def _readable_frame_count(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None

        count = 0
        try:
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                count += 1
        finally:
            cap.release()
        return count

    def validate_files(self):
        problems = []
        for video_name in self.video_files:
            path = self.video_path(video_name)
            readable_frames = self._readable_frame_count(path)
            if readable_frames is None:
                problems.append(f"{video_name}: OpenCV could not open video.")
                continue

            if readable_frames <= 0:
                problems.append(f"{video_name}: no readable frames.")
            elif readable_frames < self.num_frames:
                problems.append(f"{video_name}: only {readable_frames} readable frames, needs {self.num_frames}; will pad with last frame.")

            caption_path = self.caption_path(video_name)
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    if not f.read().strip():
                        problems.append(f"{video_name}: caption is empty.")
            except OSError as e:
                problems.append(f"{video_name}: caption read failed: {e}")

        return problems

    def _sample_frames_sequentially(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video file: {path}")

        all_frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
        finally:
            cap.release()

        if not all_frames:
            raise RuntimeError(f"OpenCV could not decode any frames from: {path}")

        sample_positions = np.linspace(0, len(all_frames) - 1, self.num_frames).round().astype(np.int64).tolist()
        return [all_frames[pos] for pos in sample_positions]

    def _augment_frame(self, frame, target_w, target_h, start_x, start_y, flip):
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize slightly larger
        frame = cv2.resize(frame, (target_w, target_h))
        # Crop consistently down to target dims
        frame = frame[start_y:start_y+self.height, start_x:start_x+self.width]
        # Flip consistently
        if flip:
            frame = cv2.flip(frame, 1)
        return frame

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video file: {path}")
        raw_frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            sample_positions = np.linspace(0, max(0, total_frames - 1), self.num_frames).round().astype(np.int64).tolist()
        else:
            sample_positions = []
        
        # Calculate consistent spatial augmentations for this entire sequence
        flip = np.random.rand() < self.flip_prob
        pad_w = int(self.width * self.crop_pad_percent)
        pad_h = int(self.height * self.crop_pad_percent)
        target_w = self.width + pad_w
        target_h = self.height + pad_h
        
        start_x = np.random.randint(0, max(1, pad_w + 1))
        start_y = np.random.randint(0, max(1, pad_h + 1))
        
        for frame_idx in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            raw_frames.append(frame)
            
        cap.release()

        if len(raw_frames) < self.num_frames:
            raw_frames = self._sample_frames_sequentially(path)

        frames = [
            self._augment_frame(frame, target_w, target_h, start_x, start_y, flip)
            for frame in raw_frames
        ]
        
        # If too few frames, pad with last frame
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.height, self.width, 3), dtype=np.uint8))
        frames = frames[:self.num_frames]
            
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
