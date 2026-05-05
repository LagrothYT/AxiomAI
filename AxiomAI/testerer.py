import cv2
import glob
import os

for path in glob.glob("data/Videos/*.mp4"):
    cap = cv2.VideoCapture(path)
    print("\n", path)
    print("exists:", os.path.exists(path))
    print("opened:", cap.isOpened())
    print("frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps:", cap.get(cv2.CAP_PROP_FPS))
    ok, frame = cap.read()
    print("first frame ok:", ok)
    print("shape:", None if frame is None else frame.shape)
    cap.release()