import os
import json

import cv2
import lmdb
import numpy as np
from tqdm import tqdm


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(rootPath, annoPath, outputPath, checkValid=True):
    """Create LMDB dataset for training and evaluation.

    Args:
        rootPath (_type_): path to images
        annoPath (_type_): path to annotations
        outputPath (_type_): LMDB output path
        checkValid (bool, optional): if true, check the validity of
            every image.Defaults to True.

    E.g.
    for text recognition task, the file structure is as follow:
      data
         |_image
               |_ img1.jpg
               |_ img2.jpg
         |_annotation.jsonl
            annotation: {'filename':image/img1.jpg, 'text': 'hello world'}
    """

    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=109951162776)
    cache = {}
    cnt = 1
    with open(annoPath, 'r') as f:
        anno_list = [json.loads(line.strip()) for line in f.readlines()]
    nSamples = len(anno_list)
    for anno in tqdm(anno_list):
        image_name, label = anno['filename'], anno['text']
        image_path = os.path.join(imagePath, image_name)
        if not os.path.exists(image_path):
            print('%s does not exist' % image_path)
            continue
        with open(image_path, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except Exception:
                print('error occured', image_name)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('{image_name} occurred error\n')
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    imagePath = 'Union14M-L/full_images'
    annoPath = 'Union14M-L/train_annos/mmocr-0.x/train_challenging.jsonl'
    outputPath = 'Union14M-L/lmdb_format/training/challenging'
    createDataset(imagePath, annoPath, outputPath)
