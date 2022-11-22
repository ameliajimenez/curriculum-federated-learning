import os
import torchvision.transforms as transforms
import torch


# model hyperparameters
pretrained = True  # use pretrained weights for feature extractor

# federated learning
nsteps = 120  # 60
pace = 40  # 20
noise_type = 'G'
noise = 0.001
n_epochs_adversarial = 5  # start propagating adversarial loss for domain adaptation after "X" epochs
torch_seed = 0

# optimization hyperparameters
n_epochs = 51  # number of epochs
batch_size = 4  # batch size
learning_rate = 1E-5  # learning rate
weight_decay = 1E-4  # weight decay
optimizer = 'adam'   # optimizer

# data parameters
preprocess = True  # apply preprocessing to images
data_seed = 42  # seed for train/val split
num_workers = 0
ignore_label = None   # 'benign'   # train normal / cancer
n_classes = 2  # number of classes
input_size = 2048  # resize images to input_size pixels

# transformations to apply to the data
data_transform = transforms.Compose([
        #transforms.CenterCrop(100),
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.5], [0.5])
])

data_path = os.path.join(os.getcwd(), 'data')
dataset_mean = (0.5, 0.5, 0.5)
dataset_std = (0.5, 0.5, 0.5)

dpath = dict()
dpath['hologic'] = dict()
dpath['hologic']['train'] = os.path.join(data_path, 'hologic-img2048x-20200319-ico-hol/train')
dpath['hologic']['test'] = os.path.join(data_path, 'hologic-img2048x-20200319-ico-hol/test')

dpath['inbreast'] = dict()
dpath['inbreast']['train'] = os.path.join(data_path, 'inbreast-tmi-exps-20200513-s0-full/train')
dpath['inbreast']['test'] = os.path.join(data_path, 'inbreast-tmi-exps-20200513-s0-full/test')

dpath['ge'] = dict()
dpath['ge']['train'] = os.path.join(data_path, 'ge-img2048x-20200319-ico-ge/train')
dpath['ge']['test'] = os.path.join(data_path, 'ge-img2048x-20200319-ico-ge/test')
