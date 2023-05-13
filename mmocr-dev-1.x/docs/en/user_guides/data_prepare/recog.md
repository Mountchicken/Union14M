# Text Recognition

```{note}
This page is a manual preparation guide for datasets not yet supported by [Dataset Preparer](./dataset_preparer.md), which all these scripts will be eventually migrated into.
```

## Overview

|     Dataset      |                         images                          |                         annotation file                          |                          annotation file                          |
| :--------------: | :-----------------------------------------------------: | :--------------------------------------------------------------: | :---------------------------------------------------------------: |
|                  |                                                         |                             training                             |                               test                                |
|    coco_text     | [homepage](https://rrc.cvc.uab.es/?ch=5&com=downloads)  |                    [train_labels.json](#TODO)                    |                                 -                                 |
|    ICDAR2011     |        [homepage](https://rrc.cvc.uab.es/?ch=1)         |                                -                                 |                                 -                                 |
|     SynthAdd     | [SynthText_Add.zip](https://pan.baidu.com/s/1uV0LtoNmcxbO-0YA7Ch4dg)  (code:627x) | [train_labels.json](https://download.openmmlab.com/mmocr/data/1.x/recog/synthtext_add/train_labels.json) |                                 -                                 |
|     OpenVINO     | [Open Images](https://github.com/cvdfoundation/open-images-dataset) | [annotations](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/datasets/open_images_v5_text) | [annotations](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/datasets/open_images_v5_text) |
|      DeText      |        [homepage](https://rrc.cvc.uab.es/?ch=9)         |                                -                                 |                                 -                                 |
| Lecture Video DB | [homepage](https://cvit.iiit.ac.in/research/projects/cvit-projects/lecturevideodb) |                                -                                 |                                 -                                 |
|       LSVT       |        [homepage](https://rrc.cvc.uab.es/?ch=16)        |                                -                                 |                                 -                                 |
|      IMGUR       | [homepage](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset) |                                -                                 |                                 -                                 |
|      KAIST       | [homepage](http://www.iapr-tc11.org/mediawiki/index.php/KAIST_Scene_Text_Database) |                                -                                 |                                 -                                 |
|       MTWI       | [homepage](https://tianchi.aliyun.com/competition/entrance/231685/information?lang=en-us) |                                -                                 |                                 -                                 |
|      ReCTS       |        [homepage](https://rrc.cvc.uab.es/?ch=12)        |                                -                                 |                                 -                                 |
|    IIIT-ILST     | [homepage](http://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-ilst) |                                -                                 |                                 -                                 |
|     VinText      | [homepage](https://github.com/VinAIResearch/dict-guided) |                                -                                 |                                 -                                 |
|       BID        | [homepage](https://github.com/ricardobnjunior/Brazilian-Identity-Document-Dataset) |                                -                                 |                                 -                                 |
|       RCTW       |     [homepage](https://rctw.vlrlab.net/index.html)      |                                -                                 |                                 -                                 |
|     HierText     | [homepage](https://github.com/google-research-datasets/hiertext) |                                -                                 |                                 -                                 |
|       ArT        |        [homepage](https://rrc.cvc.uab.es/?ch=14)        |                                -                                 |                                 -                                 |

(\*) Since the official homepage is unavailable now, we provide an alternative for quick reference. However, we do not guarantee the correctness of the dataset.

### Install AWS CLI (optional)

- Since there are some datasets that require the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) to be installed in advance, we provide a quick installation guide here:

  ```bash
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    ./aws/install -i /usr/local/aws-cli -b /usr/local/bin
    !aws configure
    # this command will require you to input keys, you can skip them except
    # for the Default region name
    # AWS Access Key ID [None]:
    # AWS Secret Access Key [None]:
    # Default region name [None]: us-east-1
    # Default output format [None]
  ```

For users in China, these datasets can also be downloaded from [OpenDataLab](https://opendatalab.com/) with high speed:

- [icdar_2013](https://opendatalab.com/ICDAR_2013?source=OpenMMLab%20GitHub)
- [icdar_2015](https://opendatalab.com/ICDAR2015?source=OpenMMLab%20GitHub)
- [IIIT5K](https://opendatalab.com/IIIT_5K?source=OpenMMLab%20GitHub)
- [ct80](https://opendatalab.com/CUTE_80?source=OpenMMLab%20GitHub)
- [svt](https://opendatalab.com/SVT?source=OpenMMLab%20GitHub)
- [Totaltext](https://opendatalab.com/TotalText?source=OpenMMLab%20GitHub)
- [IAM](https://opendatalab.com/IAM_Handwriting?source=OpenMMLab%20GitHub)

## ICDAR 2011 (Born-Digital Images)

- Step1: Download `Challenge1_Training_Task3_Images_GT.zip`, `Challenge1_Test_Task3_Images.zip`, and `Challenge1_Test_Task3_GT.txt` from [homepage](https://rrc.cvc.uab.es/?ch=1&com=downloads) `Task 1.3: Word Recognition (2013 edition)`.

  ```bash
  mkdir icdar2011 && cd icdar2011
  mkdir annotations

  # Download ICDAR 2011
  wget https://rrc.cvc.uab.es/downloads/Challenge1_Training_Task3_Images_GT.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/Challenge1_Test_Task3_Images.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/Challenge1_Test_Task3_GT.txt --no-check-certificate

  # For images
  mkdir crops
  unzip -q Challenge1_Training_Task3_Images_GT.zip -d crops/train
  unzip -q Challenge1_Test_Task3_Images.zip -d crops/test

  # For annotations
  mv Challenge1_Test_Task3_GT.txt annotations && mv crops/train/gt.txt annotations/Challenge1_Train_Task3_GT.txt
  ```

- Step2: Convert original annotations to `train_labels.json` and `test_labels.json` with the following command:

  ```bash
  python tools/dataset_converters/textrecog/ic11_converter.py PATH/TO/icdar2011
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── icdar2011
  │   ├── crops
  │   ├── train_labels.json
  │   └── test_labels.json
  ```

## coco_text

- Step1: Download from [homepage](https://rrc.cvc.uab.es/?ch=5&com=downloads)

- Step2: Download [train_labels.json](https://download.openmmlab.com/mmocr/data/mixture/coco_text/train_labels.json)

- After running the above codes, the directory structure
  should be as follows:

  ```text
  ├── coco_text
  │   ├── train_labels.json
  │   └── train_words
  ```

## SynthAdd

- Step1: Download `SynthText_Add.zip` from [SynthAdd](https://pan.baidu.com/s/1uV0LtoNmcxbO-0YA7Ch4dg) (code:627x))

- Step2: Download [train_labels.json](https://download.openmmlab.com/mmocr/data/1.x/recog/synthtext_add/train_labels.json)

- Step3:

  ```bash
  mkdir SynthAdd && cd SynthAdd

  mv /path/to/SynthText_Add.zip .

  unzip SynthText_Add.zip

  mv /path/to/train_labels.json .

  # create soft link
  cd /path/to/mmocr/data/recog

  ln -s /path/to/SynthAdd SynthAdd

  ```

- After running the above codes, the directory structure
  should be as follows:

  ```text
  ├── SynthAdd
  │   ├── train_labels.json
  │   └── SynthText_Add
  ```

## OpenVINO

- Step1 (optional): Install [AWS CLI](https://mmocr.readthedocs.io/en/latest/datasets/recog.html#install-aws-cli-optional).

- Step2: Download [Open Images](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations) subsets `train_1`, `train_2`, `train_5`, `train_f`, and `validation` to `openvino/`.

  ```bash
  mkdir openvino && cd openvino

  # Download Open Images subsets
  for s in 1 2 5 f; do
    aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_${s}.tar.gz .
  done
  aws s3 --no-sign-request cp s3://open-images-dataset/tar/validation.tar.gz .

  # Download annotations
  for s in 1 2 5 f; do
    wget https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/datasets/open_images_v5_text/text_spotting_openimages_v5_train_${s}.json
  done
  wget https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/datasets/open_images_v5_text/text_spotting_openimages_v5_validation.json

  # Extract images
  mkdir -p openimages_v5/val
  for s in 1 2 5 f; do
    tar zxf train_${s}.tar.gz -C openimages_v5
  done
  tar zxf validation.tar.gz -C openimages_v5/val
  ```

- Step3: Generate `train_{1,2,5,f}_labels.json`, `val_labels.json` and crop images using 4 processes with the following command:

  ```bash
  python tools/dataset_converters/textrecog/openvino_converter.py /path/to/openvino 4
  ```

- After running the above codes, the directory structure
  should be as follows:

  ```text
  ├── OpenVINO
  │   ├── image_1
  │   ├── image_2
  │   ├── image_5
  │   ├── image_f
  │   ├── image_val
  │   ├── train_1_labels.json
  │   ├── train_2_labels.json
  │   ├── train_5_labels.json
  │   ├── train_f_labels.json
  │   └── val_labels.json
  ```

## DeText

- Step1: Download `ch9_training_images.zip`, `ch9_training_localization_transcription_gt.zip`, `ch9_validation_images.zip`, and `ch9_validation_localization_transcription_gt.zip` from **Task 3: End to End** on the [homepage](https://rrc.cvc.uab.es/?ch=9).

  ```bash
  mkdir detext && cd detext
  mkdir imgs && mkdir annotations && mkdir imgs/training && mkdir imgs/val && mkdir annotations/training && mkdir annotations/val

  # Download DeText
  wget https://rrc.cvc.uab.es/downloads/ch9_training_images.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/ch9_training_localization_transcription_gt.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/ch9_validation_images.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/ch9_validation_localization_transcription_gt.zip --no-check-certificate

  # Extract images and annotations
  unzip -q ch9_training_images.zip -d imgs/training && unzip -q ch9_training_localization_transcription_gt.zip -d annotations/training && unzip -q ch9_validation_images.zip -d imgs/val && unzip -q ch9_validation_localization_transcription_gt.zip -d annotations/val

  # Remove zips
  rm ch9_training_images.zip && rm ch9_training_localization_transcription_gt.zip && rm ch9_validation_images.zip && rm ch9_validation_localization_transcription_gt.zip
  ```

- Step2: Generate `train_labels.json` and `test_labels.json` with following command:

  ```bash
  # Add --preserve-vertical to preserve vertical texts for training, otherwise
  # vertical images will be filtered and stored in PATH/TO/detext/ignores
  python tools/dataset_converters/textrecog/detext_converter.py PATH/TO/detext --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── detext
  │   ├── crops
  │   ├── ignores
  │   ├── train_labels.json
  │   └── test_labels.json
  ```

## NAF

- Step1: Download [labeled_images.tar.gz](https://github.com/herobd/NAF_dataset/releases/tag/v1.0) to `naf/`.

  ```bash
  mkdir naf && cd naf

  # Download NAF dataset
  wget https://github.com/herobd/NAF_dataset/releases/download/v1.0/labeled_images.tar.gz
  tar -zxf labeled_images.tar.gz

  # For images
  mkdir annotations && mv labeled_images imgs

  # For annotations
  git clone https://github.com/herobd/NAF_dataset.git
  mv NAF_dataset/train_valid_test_split.json annotations/ && mv NAF_dataset/groups annotations/

  rm -rf NAF_dataset && rm labeled_images.tar.gz
  ```

- Step2: Generate `train_labels.json`, `val_labels.json`, and `test_labels.json` with following command:

  ```bash
  # Add --preserve-vertical to preserve vertical texts for training, otherwise
  # vertical images will be filtered and stored in PATH/TO/naf/ignores
  python tools/dataset_converters/textrecog/naf_converter.py PATH/TO/naf --nproc 4

  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── naf
  │   ├── crops
  │   ├── train_labels.json
  │   ├── val_labels.json
  │   └── test_labels.json
  ```

## Lecture Video DB

```{warning}
This section is not fully tested yet.
```

```{note}
The LV dataset has already provided cropped images and the corresponding annotations
```

- Step1: Download [IIIT-CVid.zip](http://cdn.iiit.ac.in/cdn/preon.iiit.ac.in/~kartik/IIIT-CVid.zip) to `lv/`.

  ```bash
  mkdir lv && cd lv

  # Download LV dataset
  wget http://cdn.iiit.ac.in/cdn/preon.iiit.ac.in/~kartik/IIIT-CVid.zip
  unzip -q IIIT-CVid.zip

  # For image
  mv IIIT-CVid/Crops ./

  # For annotation
  mv IIIT-CVid/train.txt train_labels.json && mv IIIT-CVid/val.txt val_label.txt && mv IIIT-CVid/test.txt test_labels.json

  rm IIIT-CVid.zip
  ```

- Step2: Generate `train_labels.json`, `val.json`, and `test.json` with following command:

  ```bash
  python tools/dataset_converters/textdreog/lv_converter.py PATH/TO/lv
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── lv
  │   ├── Crops
  │   ├── train_labels.json
  │   └── test_labels.json
  ```

## LSVT

```{warning}
This section is not fully tested yet.
```

- Step1: Download [train_full_images_0.tar.gz](https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_0.tar.gz), [train_full_images_1.tar.gz](https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_1.tar.gz), and [train_full_labels.json](https://dataset-bj.cdn.bcebos.com/lsvt/train_full_labels.json) to `lsvt/`.

  ```bash
  mkdir lsvt && cd lsvt

  # Download LSVT dataset
  wget https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_0.tar.gz
  wget https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_1.tar.gz
  wget https://dataset-bj.cdn.bcebos.com/lsvt/train_full_labels.json

  mkdir annotations
  tar -xf train_full_images_0.tar.gz && tar -xf train_full_images_1.tar.gz
  mv train_full_labels.json annotations/ && mv train_full_images_1/*.jpg train_full_images_0/
  mv train_full_images_0 imgs

  rm train_full_images_0.tar.gz && rm train_full_images_1.tar.gz && rm -rf train_full_images_1
  ```

- Step2: Generate `train_labels.json` and `val_label.json` (optional) with the following command:

  ```bash
  # Annotations of LSVT test split is not publicly available, split a validation
  # set by adding --val-ratio 0.2
  # Add --preserve-vertical to preserve vertical texts for training, otherwise
  # vertical images will be filtered and stored in PATH/TO/lsvt/ignores
  python tools/dataset_converters/textdrecog/lsvt_converter.py PATH/TO/lsvt --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── lsvt
  │   ├── crops
  │   ├── ignores
  │   ├── train_labels.json
  │   └── val_label.json (optional)
  ```

## IMGUR

```{warning}
This section is not fully tested yet.
```

- Step1: Run `download_imgur5k.py` to download images. You can merge [PR#5](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset/pull/5) in your local repository to enable a **much faster** parallel execution of image download.

  ```bash
  mkdir imgur && cd imgur

  git clone https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset.git

  # Download images from imgur.com. This may take SEVERAL HOURS!
  python ./IMGUR5K-Handwriting-Dataset/download_imgur5k.py --dataset_info_dir ./IMGUR5K-Handwriting-Dataset/dataset_info/ --output_dir ./imgs

  # For annotations
  mkdir annotations
  mv ./IMGUR5K-Handwriting-Dataset/dataset_info/*.json annotations

  rm -rf IMGUR5K-Handwriting-Dataset
  ```

- Step2: Generate `train_labels.json`, `val_label.txt` and `test_labels.json` and crop images with the following command:

  ```bash
  python tools/dataset_converters/textrecog/imgur_converter.py PATH/TO/imgur
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── imgur
  │   ├── crops
  │   ├── train_labels.json
  │   ├── test_labels.json
  │   └── val_label.json
  ```

## KAIST

```{warning}
This section is not fully tested yet.
```

- Step1: Download [KAIST_all.zip](http://www.iapr-tc11.org/mediawiki/index.php/KAIST_Scene_Text_Database) to `kaist/`.

  ```bash
  mkdir kaist && cd kaist
  mkdir imgs && mkdir annotations

  # Download KAIST dataset
  wget http://www.iapr-tc11.org/dataset/KAIST_SceneText/KAIST_all.zip
  unzip -q KAIST_all.zip && rm KAIST_all.zip
  ```

- Step2: Extract zips:

  ```bash
  python tools/dataset_converters/common/extract_kaist.py PATH/TO/kaist
  ```

- Step3: Generate `train_labels.json` and `val_label.json` (optional) with following command:

  ```bash
  # Since KAIST does not provide an official split, you can split the dataset by adding --val-ratio 0.2
  # Add --preserve-vertical to preserve vertical texts for training, otherwise
  # vertical images will be filtered and stored in PATH/TO/kaist/ignores
  python tools/dataset_converters/textrecog/kaist_converter.py PATH/TO/kaist --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── kaist
  │   ├── crops
  │   ├── ignores
  │   ├── train_labels.json
  │   └── val_label.json (optional)
  ```

## MTWI

```{warning}
This section is not fully tested yet.
```

- Step1: Download `mtwi_2018_train.zip` from [homepage](https://tianchi.aliyun.com/competition/entrance/231685/information?lang=en-us).

  ```bash
  mkdir mtwi && cd mtwi

  unzip -q mtwi_2018_train.zip
  mv image_train imgs && mv txt_train annotations

  rm mtwi_2018_train.zip
  ```

- Step2: Generate `train_labels.json` and `val_label.json` (optional) with the following command:

  ```bash
  # Annotations of MTWI test split is not publicly available, split a validation
  # set by adding --val-ratio 0.2
  # Add --preserve-vertical to preserve vertical texts for training, otherwise
  # vertical images will be filtered and stored in PATH/TO/mtwi/ignores
  python tools/dataset_converters/textrecog/mtwi_converter.py PATH/TO/mtwi --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── mtwi
  │   ├── crops
  │   ├── train_labels.json
  │   └── val_label.json (optional)
  ```

## ReCTS

```{warning}
This section is not fully tested yet.
```

- Step1: Download [ReCTS.zip](https://datasets.cvc.uab.es/rrc/ReCTS.zip) to `rects/` from the [homepage](https://rrc.cvc.uab.es/?ch=12&com=downloads).

  ```bash
  mkdir rects && cd rects

  # Download ReCTS dataset
  # You can also find Google Drive link on the dataset homepage
  wget https://datasets.cvc.uab.es/rrc/ReCTS.zip --no-check-certificate
  unzip -q ReCTS.zip

  mv img imgs && mv gt_unicode annotations

  rm ReCTS.zip -f && rm -rf gt
  ```

- Step2: Generate `train_labels.json` and `val_label.json` (optional) with the following command:

  ```bash
  # Annotations of ReCTS test split is not publicly available, split a validation
  # set by adding --val-ratio 0.2
  # Add --preserve-vertical to preserve vertical texts for training, otherwise
  # vertical images will be filtered and stored in PATH/TO/rects/ignores
  python tools/dataset_converters/textrecog/rects_converter.py PATH/TO/rects --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── rects
  │   ├── crops
  │   ├── ignores
  │   ├── train_labels.json
  │   └── val_label.json (optional)
  ```

## ILST

```{warning}
This section is not fully tested yet.
```

- Step1: Download `IIIT-ILST.zip` from [onedrive link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/minesh_mathew_research_iiit_ac_in/EtLvCozBgaBIoqglF4M-lHABMgNcCDW9rJYKKWpeSQEElQ?e=zToXZP)

- Step2: Run the following commands

  ```bash
  unzip -q IIIT-ILST.zip && rm IIIT-ILST.zip
  cd IIIT-ILST

  # rename files
  cd Devanagari && for i in `ls`; do mv -f $i `echo "devanagari_"$i`; done && cd ..
  cd Malayalam && for i in `ls`; do mv -f $i `echo "malayalam_"$i`; done && cd ..
  cd Telugu && for i in `ls`; do mv -f $i `echo "telugu_"$i`; done && cd ..

  # transfer image path
  mkdir imgs && mkdir annotations
  mv Malayalam/{*jpg,*jpeg} imgs/ && mv Malayalam/*xml annotations/
  mv Devanagari/*jpg imgs/ && mv Devanagari/*xml annotations/
  mv Telugu/*jpeg imgs/ && mv Telugu/*xml annotations/

  # remove unnecessary files
  rm -rf Devanagari && rm -rf Malayalam && rm -rf Telugu && rm -rf README.txt
  ```

- Step3: Generate `train_labels.json` and `val_label.json` (optional) and crop images using 4 processes with the following command (add `--preserve-vertical` if you wish to preserve the images containing vertical texts). Since the original dataset doesn't have a validation set, you may specify `--val-ratio` to split the dataset. E.g., if val-ratio is 0.2, then 20% of the data are left out as the validation set in this example.

  ```bash
  python tools/dataset_converters/textrecog/ilst_converter.py PATH/TO/IIIT-ILST --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── IIIT-ILST
  │   ├── crops
  │   ├── ignores
  │   ├── train_labels.json
  │   └── val_label.json (optional)
  ```

## VinText

```{warning}
This section is not fully tested yet.
```

- Step1: Download [vintext.zip](https://drive.google.com/drive/my-drive) to `vintext`

  ```bash
  mkdir vintext && cd vintext

  # Download dataset from google drive
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml" -O vintext.zip && rm -rf /tmp/cookies.txt

  # Extract images and annotations
  unzip -q vintext.zip && rm vintext.zip
  mv vietnamese/labels ./ && mv vietnamese/test_image ./ && mv vietnamese/train_images ./ && mv vietnamese/unseen_test_images ./
  rm -rf vietnamese

  # Rename files
  mv labels annotations && mv test_image test && mv train_images  training && mv unseen_test_images  unseen_test
  mkdir imgs
  mv training imgs/ && mv test imgs/ && mv unseen_test imgs/
  ```

- Step2: Generate `train_labels.json`, `test_labels.json`, `unseen_test_labels.json`,  and crop images using 4 processes with the following command (add `--preserve-vertical` if you wish to preserve the images containing vertical texts).

  ```bash
  python tools/dataset_converters/textrecog/vintext_converter.py PATH/TO/vietnamese --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── vintext
  │   ├── crops
  │   ├── ignores
  │   ├── train_labels.json
  │   ├── test_labels.json
  │   └── unseen_test_labels.json
  ```

## BID

```{warning}
This section is not fully tested yet.
```

- Step1: Download [BID Dataset.zip](https://drive.google.com/file/d/1Oi88TRcpdjZmJ79WDLb9qFlBNG8q2De6/view)

- Step2: Run the following commands to preprocess the dataset

  ```bash
  # Rename
  mv BID\ Dataset.zip BID_Dataset.zip

  # Unzip and Rename
  unzip -q BID_Dataset.zip && rm BID_Dataset.zip
  mv BID\ Dataset BID

  # The BID dataset has a problem of permission, and you may
  # add permission for this file
  chmod -R 777 BID
  cd BID
  mkdir imgs && mkdir annotations

  # For images and annotations
  mv CNH_Aberta/*in.jpg imgs && mv CNH_Aberta/*txt annotations && rm -rf CNH_Aberta
  mv CNH_Frente/*in.jpg imgs && mv CNH_Frente/*txt annotations && rm -rf CNH_Frente
  mv CNH_Verso/*in.jpg imgs && mv CNH_Verso/*txt annotations && rm -rf CNH_Verso
  mv CPF_Frente/*in.jpg imgs && mv CPF_Frente/*txt annotations && rm -rf CPF_Frente
  mv CPF_Verso/*in.jpg imgs && mv CPF_Verso/*txt annotations && rm -rf CPF_Verso
  mv RG_Aberto/*in.jpg imgs && mv RG_Aberto/*txt annotations && rm -rf RG_Aberto
  mv RG_Frente/*in.jpg imgs && mv RG_Frente/*txt annotations && rm -rf RG_Frente
  mv RG_Verso/*in.jpg imgs && mv RG_Verso/*txt annotations && rm -rf RG_Verso

  # Remove unnecessary files
  rm -rf desktop.ini
  ```

- Step3: Generate `train_labels.json` and `val_label.json` (optional) and crop images using 4 processes with the following command (add `--preserve-vertical` if you wish to preserve the images containing vertical texts). Since the original dataset doesn't have a validation set, you may specify `--val-ratio` to split the dataset. E.g., if test-ratio is 0.2, then 20% of the data are left out as the validation set in this example.

  ```bash
  python tools/dataset_converters/textrecog/bid_converter.py PATH/TO/BID --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── BID
  │   ├── crops
  │   ├── ignores
  │   ├── train_labels.json
  │   └── val_label.json (optional)
  ```

## RCTW

```{warning}
This section is not fully tested yet.
```

- Step1: Download `train_images.zip.001`, `train_images.zip.002`, and `train_gts.zip` from the [homepage](https://rctw.vlrlab.net/dataset.html), extract the zips to `rctw/imgs` and `rctw/annotations`, respectively.

- Step2: Generate `train_labels.json` and `val_label.json` (optional). Since the original dataset doesn't have a validation set, you may specify `--val-ratio` to split the dataset. E.g., if val-ratio is 0.2, then 20% of the data are left out as the validation set in this example.

  ```bash
  # Annotations of RCTW test split is not publicly available, split a validation set by adding --val-ratio 0.2
  # Add --preserve-vertical to preserve vertical texts for training, otherwise vertical images will be filtered and stored in PATH/TO/rctw/ignores
  python tools/dataset_converters/textrecog/rctw_converter.py PATH/TO/rctw --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  │── rctw
  │   ├── crops
  │   ├── ignores
  │   ├── train_labels.json
  │   └── val_label.json (optional)
  ```

## HierText

```{warning}
This section is not fully tested yet.
```

- Step1 (optional): Install [AWS CLI](https://mmocr.readthedocs.io/en/latest/datasets/recog.html#install-aws-cli-optional).

- Step2: Clone [HierText](https://github.com/google-research-datasets/hiertext) repo to get annotations

  ```bash
  mkdir HierText
  git clone https://github.com/google-research-datasets/hiertext.git
  ```

- Step3: Download `train.tgz`, `validation.tgz` from aws

  ```bash
  aws s3 --no-sign-request cp s3://open-images-dataset/ocr/train.tgz .
  aws s3 --no-sign-request cp s3://open-images-dataset/ocr/validation.tgz .
  ```

- Step4: Process raw data

  ```bash
  # process annotations
  mv hiertext/gt ./
  rm -rf hiertext
  mv gt annotations
  gzip -d annotations/train.json.gz
  gzip -d annotations/validation.json.gz
  # process images
  mkdir imgs
  mv train.tgz imgs/
  mv validation.tgz imgs/
  tar -xzvf imgs/train.tgz
  tar -xzvf imgs/validation.tgz
  ```

- Step5: Generate `train_labels.json` and `val_label.json`. HierText includes different levels of annotation, including `paragraph`, `line`, and `word`. Check the original [paper](https://arxiv.org/pdf/2203.15143.pdf) for details. E.g. set `--level paragraph` to get paragraph-level annotation. Set `--level line` to get line-level annotation. set `--level word` to get word-level annotation.

  ```bash
  # Collect word annotation from HierText  --level word
  # Add --preserve-vertical to preserve vertical texts for training, otherwise vertical images will be filtered and stored in PATH/TO/HierText/ignores
  python tools/dataset_converters/textrecog/hiertext_converter.py PATH/TO/HierText --level word --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  │── HierText
  │   ├── crops
  │   ├── ignores
  │   ├── train_labels.json
  │   └── val_label.json
  ```

## ArT

```{warning}
This section is not fully tested yet.
```

- Step1: Download `train_images.tar.gz`, and `train_labels.json` from the [homepage](https://rrc.cvc.uab.es/?ch=14&com=downloads) to `art/`

  ```bash
  mkdir art && cd art
  mkdir annotations

  # Download ArT dataset
  wget https://dataset-bj.cdn.bcebos.com/art/train_task2_images.tar.gz
  wget https://dataset-bj.cdn.bcebos.com/art/train_task2_labels.json

  # Extract
  tar -xf train_task2_images.tar.gz
  mv train_task2_images crops
  mv train_task2_labels.json annotations/

  # Remove unnecessary files
  rm train_images.tar.gz
  ```

- Step2: Generate `train_labels.json` and `val_label.json` (optional). Since the test annotations are not publicly available, you may specify `--val-ratio` to split the dataset. E.g., if val-ratio is 0.2, then 20% of the data are left out as the validation set in this example.

  ```bash
  # Annotations of ArT test split is not publicly available, split a validation set by adding --val-ratio 0.2
  python tools/dataset_converters/textrecog/art_converter.py PATH/TO/art
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  │── art
  │   ├── crops
  │   ├── train_labels.json
  │   └── val_label.json (optional)
  ```
