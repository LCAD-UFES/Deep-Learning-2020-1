# Using Generative Adversarial Network techniques to upscale images for Automatic License Plate Recognition

Requirement for the discipline of Deep Learning from PPGI/UFES 2021/01.

## Requirements

- `Google Collab or JupyterNotebook` (Collab prefered for GPU training)
- `Google Drive`

## TLDR

Use demonstration.ipynb to run all codes in this page.

## Modifications of Original Code

- SRGAN: Image of butterfly and code were modified in order to observe the training (by a image of a plate). Tensorboard were included in the code.
- ESRGAN: Training were modified to train with same conditions of SRGAN (without image augmentation). Result were modified to try the resolution enhancement with original image.
- Other Algorithms: Algorithm and requirements were modified to run with newest versions of tensorflow.

## Initializing the project

Put the folder 'FOLDER' in your Google Drive

```bash
1)from google.colab import drive
drive.mount('/content/drive/', force_remount=True) # Mount Google Drive folders.

2)cd /content/drive/MyDrive/FOLDER # Install requirements
  !pip install -r requirements.txt
# Run every time you enter Colab.
```

## SRGAN

- `!python3 /content/drive/MyDrive/FOLDER/train.py -a srgan --gpu 0 --gan-epochs 200 --psnr-epoch 200 /content/drive/MyDrive/FOLDER/DatasetFinalPb`: Training with 200 epochs
- In train.py `base_image = transforms.ToTensor()(Image.open(os.path.join("assets", "baseimage.png")))`: Change this line to the image that you want to upscale
- `!python3 /content/drive/MyDrive/FOLDER/test_image.py -a srgan --gpu 0 --lr /content/drive/MyDrive/FOLDER/PLACADETESTE.png --model-path/content/drive/MyDrive/PASTA/weights/GAN-best.pth`: Testing using the last/best GAN values.

## Dependencies

`requirements.txt`

```python
opencv-python>=4.5.2.52
torchvision>=0.9.1+cu111
Pillow>=8.2.0
numpy>=1.19.5
torch>=1.8.1+cu111
tqdm>=4.60.0
scipy>=1.6.3
prettytable>=2.1.0
thop>=0.0.31.post2005241907
setuptools>=56.2.0
tensorboardX>=2.2
lpips>=0.1.3
albumentations
easyocr
pytesseract
imutils
```

## Dependencies for ALPN and OCR

```
tensorboardX
albumentations
easyocr
pytesseract
imutils
torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## DLSS

- In `Load Model and Analyze Results.ipynb`, change `model_path` (Path to saved .h5 model), `dataset_path` (Path to folder containing images to super sample), `save_path` (Folder where you want to save to model as well as generated samples)
- Run `Load Model and Analyze Results.ipynb` (Collab prefered for GPU training)

## ESRGAN

- Training: In `Training.ipynb`, change the main folder of ESRGAN algorithm (`cd /content/drive/MyDrive/License-super-resolution/`) and the `PATHTRAIN` / `PATHTEST` (Path to folder containing images to train algorithm). Use only images of 192 x 96 for training. Change the number of epochs (`epochs=1`).
- Generating High Resolution Image: change the main folder of ESRGAN algorithm (`cd /content/drive/MyDrive/License-super-resolution/`) and the `DATA_PATH` (Path to folder containing images to super sample). Change the `model.load_weights Choose` to the desired weights. Choose between `original` (original image) or`downSample` (downsampled image) to run the plate enhancement.

## Other Algorithms (SRFEAT, EDSR, ERCA, ...)

Collab with GPU is required in these algorithms.
Use `KerasImageSuperResolution.ipynb` to run the algorithm.

- Generating High Resolution images: change the main folder of algorithms (`cd /content/drive/MyDrive/SuperResolution/Keras-Image-Super-Resolution/`)
  - !python demo.py --arc=esrgan --lr_dir=/content/drive/MyDrive/UFPRCROPPEDPB/ --ext=.png --save_dir=/content/drive/MyDrive/SuperResolution/dataset/UFPROUTPUT/ESRGAN --model_path=/content/drive/MyDrive/SuperResolution/Keras-Image-Super-Resolution/exp/esrgan-gan-06-11-14:35/gan-cp-01.h5 --cuda=0
  - Change `arc` with the choosen algorithm (SRFEAT, EDSR, ERCA, ESRGAN, SRGAN).
  - Change `lr_dir` with folder containing images to super sample.
  - Change `ext` with the extension of images (png, jpg, ...).
  - Change `save_dir` with the output directory.
  - Change `model_path` with the .h5 model (inside checkpoints)

## Utilities

- Labelbox_Processing: a tool to receive data from labelbox and crop images of datasets.
- Processing_Resize: a tool written in Processing to resize and pre-process images.

## REFERENCES

-[SRGAN](https://github.com/Lornatang/SRGAN-PyTorch)

-[DLSS](https://github.com/vee-upatising/DLSS)

-[ESRGAN](https://github.com/zzxvictor/License-super-resolution)

-[Other Algorithms](https://github.com/hieubkset/Keras-Image-Super-Resolution)

-[ALPN and OCR](https://github.com/PhelaPoscam/ALPN-with-OpenCV-and-EasyOCR)

-[OCR New Version](https://platerecognizer.com/)
