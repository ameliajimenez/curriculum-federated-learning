import torch
from torch.utils.data import TensorDataset
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix
from networks import Classifier
from dataset import CustomDataSet
import numpy as np
import os
import csv
import params
import warnings
warnings.filterwarnings("ignore")

torch_seed = 0
dataset_name = 'hologic'  # ['hologic', 'inbreast', 'ge']
test_dirs = [params.dpath[dataset_name]['test']]

testset = CustomDataSet(test_dirs, params.data_transform, params.preprocess, ignore_label=params.ignore_label)
test_loader = torch.utils.data.DataLoader(testset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)

# experiment - save model
model_path = 'models/fed-align-cl/0.001/120-40/torch-seed-'+str(torch_seed)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# setup models
model = Classifier()

checkpoint = torch.load(os.path.join(model_path, 'model.pt'))
epoch = checkpoint['epoch']
print('Epoch model converged: '+str(epoch))
model.load_state_dict(checkpoint['global_model'])
model = model.to(device)
model.eval()

test_loss = 0
correct = 0
probabilities = []
predictions = []
targets = []
for inputs, labels, domain, idx in test_loader:
    inputs = inputs.to(device)
    targets.append(labels.detach().cpu().numpy())
    labels = labels.to(device)
    probs, logits = model(inputs)
    probabilities.append(probs.detach().cpu().numpy())
    preds = torch.argmax(probs, 1)
    predictions.append(preds.detach().cpu().numpy())
    correct += preds.eq(labels.view(-1)).sum().item()

correct /= len(test_loader.dataset)

# calculate precision and recall for each threshold
y_true = np.asarray([val for sublist in targets for val in sublist])
y_prob = np.asarray([np.exp(val[1]) for sublist in probabilities for val in sublist])
y_pred = np.asarray([np.argmax(val) for sublist in probabilities for val in sublist])
lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_prob)
pr_auc = auc(lr_recall, lr_precision)
roc_auc = roc_auc_score(y_true, y_prob)

print('ROC AUC: {:.4f}'.format(roc_auc))
print('PR AUC: {:.4f}'.format(pr_auc))
print(confusion_matrix(y_true, y_pred))

# get filenames
test_dirs = [params.dpath[dataset_name]['test']]
testset = CustomDataSet(test_dirs, params.data_transform, params.preprocess, ignore_label=params.ignore_label)

new_list = [[testset.total_imgs[i]] for i in range(len(testset.total_imgs))]
with open(dataset_name+'_test_files.csv', 'w') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(new_list)
