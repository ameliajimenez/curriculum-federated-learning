import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from networks import Classifier
from dataset import CustomDataSet
import numpy as np
import os
import params
import warnings
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
warnings.filterwarnings("ignore")


def combine_cam_on_image(img, mask, image_weight=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)  # reverse
    img = 0.5 * (img + 1.0)
    img = np.uint8(255 * img)
    cam_img = (1 - image_weight) * heatmap + image_weight * np.repeat(img[:, :, np.newaxis], 3, axis=2)
    cam_img = cam_img / np.max(cam_img)
    return cam_img


def get_indices_preds(model, test_loader, mode_preds):
    probabilities = []
    predictions = []
    targets = []
    for inputs, labels, domain, idx in test_loader:
        inputs = inputs.to(device)
        targets.append(labels.detach().cpu().numpy())
        probs, logits = model(inputs)
        probabilities.append(probs.detach().cpu().numpy())
        preds = torch.argmax(probs, 1)
        predictions.append(preds.detach().cpu().numpy())

    # calculate precision and recall for each threshold
    y_true = np.asarray([val for sublist in targets for val in sublist])
    y_pred = np.asarray([np.argmax(val) for sublist in probabilities for val in sublist])

    if mode_preds == 'correct':
        indices_preds = np.arange(len(y_true))[y_true == y_pred]  # get correct predictions
    elif mode_preds == 'wrong':
        indices_preds = np.arange(len(y_true))[y_true != y_pred]  # get misclassified samples

    return indices_preds


# Dataset
dataset_name = 'ge'  # ['hologic', 'inbreast', 'ge']
test_dirs = [params.dpath[dataset_name]['test']]

testset = CustomDataSet(test_dirs, params.data_transform, params.preprocess, ignore_label=params.ignore_label)
test_loader = torch.utils.data.DataLoader(testset, batch_size=params.batch_size,
                                          shuffle=False, num_workers=params.num_workers)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch_seed = 27

preds_list = []

for model_type in ['fed', 'fed-cl', 'fed-align', 'fed-align-cl']:
    print(model_type)

    model_path = os.path.join('models', model_type, '0.001/120-40/torch-seed-'+str(torch_seed))
    if model_type == 'fed-align-cl':
        mode_preds = 'correct'  # get indices of correct predictions
    else:
        mode_preds = 'wrong'  # get indices of wrong predictions

    # setup models
    model = Classifier()
    checkpoint = torch.load(os.path.join(model_path, 'model.pt'))
    model.load_state_dict(checkpoint['global_model'])
    model = model.to(device)
    model.eval()

    preds = get_indices_preds(model, test_loader, mode_preds=mode_preds)
    preds_list.append(preds)

# get intersection of images: misclassified by federated models, correctly classified by fed-align-cl
common_list = set(preds_list[0])
for next_list in preds_list[1:]:
    common_list.intersection_update(next_list)
print('common list')
print(common_list)

test_dirs = [params.dpath[dataset_name]['test']]
testset = CustomDataSet(test_dirs, params.data_transform, params.preprocess, ignore_label=params.ignore_label)

test_filenames = [[testset.total_imgs[i]] for i in range(len(testset.total_imgs))]
common_filenames = [test_filenames[idx] for idx in common_list]

# our method vs. all federated methods
# model paths
model_path_fed = os.path.join('models', 'fed', '0.001/120-40/torch-seed-' + str(torch_seed))
model_path_fed_cl = os.path.join('models', 'fed-cl', '0.001/120-40/torch-seed-' + str(torch_seed))
model_path_fed_align = os.path.join('models', 'fed-align', '0.001/120-40/torch-seed-' + str(torch_seed))
model_path_fed_align_cl = os.path.join('models', 'fed-align-cl', '0.001/120-40/torch-seed-' + str(torch_seed))

# setup models
model_fed = Classifier()
model_fed_cl = Classifier()
model_fed_align = Classifier()
model_fed_align_cl = Classifier()

# model fed
checkpoint_fed = torch.load(os.path.join(model_path_fed, 'model.pt'))
model_fed.load_state_dict(checkpoint_fed['global_model'])
model_fed = model_fed.to(device)
model_fed.eval()

# model fed cl
checkpoint_fed_cl = torch.load(os.path.join(model_path_fed_cl, 'model.pt'))
model_fed_cl.load_state_dict(checkpoint_fed_cl['global_model'])
model_fed_cl = model_fed_cl.to(device)
model_fed_cl.eval()

# model fed align
checkpoint_fed_align = torch.load(os.path.join(model_path_fed_align, 'model.pt'))
model_fed_align.load_state_dict(checkpoint_fed_align['global_model'])
model_fed_align = model_fed_align.to(device)
model_fed_align.eval()

# model fed align cl
checkpoint_fed_align_cl = torch.load(os.path.join(model_path_fed_align_cl, 'model.pt'))
model_fed_align_cl.load_state_dict(checkpoint_fed_align_cl['global_model'])
model_fed_align_cl = model_fed_align_cl.to(device)
model_fed_align_cl.eval()

# GRAD CAM
target_layers_fed = model_fed.encoder.view_resnet.layer_list[-1]
target_layers_fed_cl = model_fed_cl.encoder.view_resnet.layer_list[-1]
target_layers_fed_align = model_fed_align.encoder.view_resnet.layer_list[-1]
target_layers_fed_align_cl = model_fed_align_cl.encoder.view_resnet.layer_list[-1]

# Construct the CAM object once, and then re-use it on many images:
cam_fed = GradCAM(model=model_fed, target_layers=target_layers_fed, use_cuda=True)
cam_fed_cl = GradCAM(model=model_fed_cl, target_layers=target_layers_fed_cl, use_cuda=True)
cam_fed_align = GradCAM(model=model_fed_align, target_layers=target_layers_fed_align, use_cuda=True)
cam_fed_align_cl = GradCAM(model=model_fed_align_cl, target_layers=target_layers_fed_align_cl, use_cuda=True)

# class: 0, 1
categories = ['Normal-Benign', 'Malignant']
output_targets = [ClassifierOutputTarget(1)]

subset = torch.utils.data.Subset(testset, list(common_list))
test_loader_subset = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

for inputs, labels, domain, idx in test_loader_subset:
    inputs = inputs.to(device)
    probs, logits = model(inputs)

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam_fed = cam_fed(input_tensor=inputs, targets=output_targets)
    grayscale_cam_fed_cl = cam_fed_cl(input_tensor=inputs, targets=output_targets)
    grayscale_cam_fed_align = cam_fed_align(input_tensor=inputs, targets=output_targets)
    grayscale_cam_fed_align_cl = cam_fed_align_cl(input_tensor=inputs, targets=output_targets)

    # In this example grayscale_cam has only one image in the batch:
    img = torch.squeeze(inputs).cpu().detach().numpy()
    the_class = int(labels.cpu().detach().numpy())

    cam_img_fed = combine_cam_on_image(img, grayscale_cam_fed[0, :])
    cam_img_fed_cl = combine_cam_on_image(img, grayscale_cam_fed_cl[0, :])
    cam_img_fed_align = combine_cam_on_image(img, grayscale_cam_fed_align[0, :])
    cam_img_fed_align_cl = combine_cam_on_image(img, grayscale_cam_fed_align_cl[0, :])

    plt.subplot(151)
    plt.imshow(img, 'gray')
    plt.axis('off')
    plt.title(categories[the_class])

    plt.subplot(152)
    plt.imshow(cam_img_fed)
    plt.axis('off')
    plt.title('Fed')

    plt.subplot(153)
    plt.imshow(cam_img_fed_cl)
    plt.axis('off')
    plt.title('Fed-CL')

    plt.subplot(154)
    plt.imshow(cam_img_fed_align)
    plt.axis('off')
    plt.title('Fed-Align')

    plt.subplot(155)
    plt.imshow(cam_img_fed_align_cl)
    plt.axis('off')
    plt.title('Fed-Align-CL')

    plt.tight_layout()
    idx_ = int(idx.cpu().detach().numpy())
    save_path = 'grad-cam/' + dataset_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'image_id_'+str(idx_) + '_' + categories[the_class] + '.png'),
                bbox_inches='tight')
    plt.show()
