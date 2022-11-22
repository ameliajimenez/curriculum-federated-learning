import torch
from torch import nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader
from networks import Classifier
import params
from dataset import load_data
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

torch_seed = 2
torch.manual_seed(torch_seed)

sites = ['hologic', 'inbreast', 'ge']
source = 'inbreast'

# model path
PATH = './models/single/' + source + '/torch-seed-' + str(torch_seed)
# PATH = './models/mix/torch-seed-' + str(torch_seed)
print(PATH)

# setup models
model = Classifier()

# init criterions
class_criterion = nn.CrossEntropyLoss()

# dataset paths
train_dir = [params.dpath[source]['train']]

# load train / validation data
trainset, valset = load_data(train_dir, preprocess=params.preprocess, data_seed=params.data_seed,
                             ignore_label=params.ignore_label, val_split=0.15)

# define dataloaders
trainloader = DataLoader(trainset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
valloader = DataLoader(valset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)

optimizer = optim.Adam([{'params': model.parameters()}], lr=params.learning_rate)  #, weight_decay=params.weight_decay)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load pretrained weights from Wu et al model
if params.pretrained:
    print('loading pretrained weights')
    image_only_parameters = dict()
    image_only_parameters["model_path"] = "models/pretrained/sample_image_model.p"
    image_only_parameters["view"] = "L-CC"
    image_only_parameters["use_heatmaps"] = False

    model.encoder.load_state_from_shared_weights(
        state_dict=torch.load(image_only_parameters["model_path"])["model"],
        view=image_only_parameters["view"],
    )

model.to(device)

# Summary writers
writer_train = SummaryWriter(os.path.join(PATH, 'train'))
writer_val = SummaryWriter(os.path.join(PATH, 'val'))

best_val_loss = np.inf
n_train = len(trainset.indices)
sample_weights = np.ones(n_train)

print('Start optimization')
for epoch in range(params.n_epochs):
    train_running_loss = 0.0
    val_running_loss = 0.0

    model.train()

    # TRAINING ROUND
    for i, data in enumerate(trainloader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # get the inputs
        inputs, labels, domains, idx = data
        # show_images(inputs, fig_title=exp_name + str(i))

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        probs, preds = model(inputs)
        loss = class_criterion(preds, labels)

        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly

        train_running_loss += loss.detach().item()

    model.eval()
    n_batches_train = np.copy(i)

    # VALIDATION ROUND
    for i, data in enumerate(valloader):
        # get the inputs
        inputs, labels, domains, idx = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        probs, preds = model(inputs)

        loss = class_criterion(preds, labels)

        val_running_loss += loss.detach().item()

    n_batches_val = np.copy(i)

    # print progress
    print('Epoch:  {:3d} | Train Loss: {:.4f} | Val Loss: {:.4f} '.format(epoch,
                                                                          train_running_loss / n_batches_train,
                                                                          val_running_loss / n_batches_val))
    # write summaries
    writer_train.add_scalar('loss', train_running_loss / n_batches_train, epoch)
    writer_val.add_scalar('loss', val_running_loss / n_batches_val, epoch)

    # save model at minimum validation loss
    if (val_running_loss / i) < best_val_loss:
        print('saving model')
        best_val_loss = val_running_loss / n_batches_val
        # save model
        if not os.path.exists(PATH):
            os.mkdir(PATH)

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, os.path.join(PATH, 'model.pt'))

print('Optimization finished!')
