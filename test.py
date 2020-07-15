import easyocr
from tqdm.auto import tqdm
from skimage import io
import numpy as np
import cv2
import os
import time
from easyocr.datautils import MyDataset
import torch


reader = easyocr.Reader(['ko', 'en'])
folder = 'images'
canvas_size=2560
exec_time = 1
num_of_img = len(os.listdir(f'./{folder}'))

start = time.time()

for i in range(exec_time):
    # res = reader.get_textbox_batch_test(f'./{folder}', batch_size=1, workers=0)

    for i, file in enumerate(sorted(os.listdir(f'./{folder}'))):
        if 'DS' in file:
            continue
        cur = time.time()
        result = reader.readtext(f'./{folder}/{file}', batch_size=32, detail=0)
        print(f"{file}\t time: {time.time() - cur}\t text: {result}")
        # text = reader.get_textbox_test(f'./{folder}/{file}')

fin = time.time() - start
print("time:", fin, "\n1 steps: ", fin/exec_time, "\n1 images: ", fin / (num_of_img * exec_time))