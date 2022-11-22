import torch
from torch import nn
import torch.optim as optim
import torch.distributions as tdist
import numpy as np
import os
from torch.utils.data import DataLoader
from networks import Classifier
import params
from dataset import load_data
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")


def test(federated_model, dataloader, train=False):
    federated_model.eval()
    val_running_loss = 0
    correct = 0
    probabilities = []
    predictions = []
    targets = []
    for n_batches, (inputs, labels, domain, idx) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        probs, logits = federated_model(inputs)
        preds = torch.argmax(probs, 1)
        loss = class_criterion(logits, labels)  # compute loss
        targets.append(labels.detach().cpu().numpy())
        probabilities.append(probs.detach().cpu().numpy())
        predictions.append(preds.detach().cpu().numpy())
        correct += preds.eq(labels.view(-1)).sum().item()
        val_running_loss += loss.item()

    correct /= len(dataloader.dataset)
    val_running_loss /= n_batches

    if train:
        print('Train set local: Average loss: {:.4f}, Average acc: {:.4f}'.format(val_running_loss, correct))
    else:
        print('Test set local: Average loss: {:.4f}, Average acc: {:.4f}'.format(val_running_loss, correct))
    return val_running_loss, correct, targets, probabilities, predictions


def get_predictions(model, dataloader, n_train_val):
    model.eval()
    correct_predictions = np.zeros(n_train_val)
    train_indices = dataloader.dataset.indices
    for n_batches, (inputs, labels, domain, idx) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        probs, logits = model(inputs)
        correct_preds = torch.eq(labels, torch.argmax(probs, dim=1)).int()
        correct_predictions[idx] = correct_preds.detach().cpu().numpy()
    correct_predictions = correct_predictions[train_indices]
    return correct_predictions


torch.manual_seed(params.torch_seed)

sites = ['hologic', 'inbreast', 'ge']
n_sites = len(sites)

# model path
PATH = './models/fed/'+str(params.noise)+'/'+str(params.nsteps)+'-'+str(params.pace)+'/torch-seed-' + str(params.torch_seed)
print(PATH)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# setup models & optimizers
global_model = Classifier().to(device)
local_model0 = Classifier().to(device)
local_model1 = Classifier().to(device)
local_model2 = Classifier().to(device)
local_models = [local_model0, local_model1, local_model2]

optimizer0 = optim.Adam([{'params': local_model0.parameters()}], lr=params.learning_rate)  #, weight_decay=params.weight_decay)
optimizer1 = optim.Adam([{'params': local_model1.parameters()}], lr=params.learning_rate)  #, weight_decay=params.weight_decay)
optimizer2 = optim.Adam([{'params': local_model2.parameters()}], lr=params.learning_rate)  #, weight_decay=params.weight_decay)
optimizers = [optimizer0, optimizer1, optimizer2]

# init criterions
class_criterion = nn.CrossEntropyLoss()

# dataset paths
train_dir0 = [params.dpath['hologic']['train']]
train_dir1 = [params.dpath['inbreast']['train']]
train_dir2 = [params.dpath['ge']['train']]

# load train / validation data
trainset0, valset0 = load_data(train_dir0, preprocess=params.preprocess, data_seed=params.data_seed,
                               ignore_label=params.ignore_label, val_split=0.15)
trainset1, valset1 = load_data(train_dir1, preprocess=params.preprocess, data_seed=params.data_seed,
                               ignore_label=params.ignore_label, val_split=0.15)
trainset2, valset2 = load_data(train_dir2, preprocess=params.preprocess, data_seed=params.data_seed,
                               ignore_label=params.ignore_label, val_split=0.15)
trainsets = [trainset0, trainset1, trainset2]

train_loader0 = DataLoader(trainset0, batch_size=len(trainset0)//params.nsteps, shuffle=True)  # len(trainset0)//params.nsteps
train_loader1 = DataLoader(trainset1, batch_size=len(trainset1)//params.nsteps, shuffle=True)
train_loader2 = DataLoader(trainset2, batch_size=len(trainset2)//params.nsteps, shuffle=True)

val_loader0 = DataLoader(valset0, batch_size=params.batch_size, shuffle=False)
val_loader1 = DataLoader(valset1, batch_size=params.batch_size, shuffle=False)
val_loader2 = DataLoader(valset2, batch_size=params.batch_size, shuffle=False)

# load pretrained weights from Wu et al model
if params.pretrained:
    print('loading pretrained weights')
    image_only_parameters = dict()
    image_only_parameters["model_path"] = "models/pretrained/sample_image_model.p"
    image_only_parameters["view"] = "L-CC"
    image_only_parameters["use_heatmaps"] = False

    local_model0.encoder.load_state_from_shared_weights(
        state_dict=torch.load(image_only_parameters["model_path"])["model"],
        view=image_only_parameters["view"],
    )
    local_model1.encoder.load_state_from_shared_weights(
        state_dict=torch.load(image_only_parameters["model_path"])["model"],
        view=image_only_parameters["view"],
    )
    local_model2.encoder.load_state_from_shared_weights(
        state_dict=torch.load(image_only_parameters["model_path"])["model"],
        view=image_only_parameters["view"],
    )

# define weights to combine local models
w = dict()
for i in range(n_sites):
    w[i] = 1.0 / n_sites

# Summary writers
writer_train = SummaryWriter(os.path.join(PATH, 'train'))
writer_val = SummaryWriter(os.path.join(PATH, 'val'))

best_val_loss = np.inf
n_train_val = [np.asarray(train_loader0.dataset.indices).max() + 1,
               np.asarray(train_loader1.dataset.indices).max() + 1,
               np.asarray(train_loader2.dataset.indices).max() + 1]

print('Start optimization')
for epoch in range(params.n_epochs):

    data_inters = [iter(train_loader0), iter(train_loader1), iter(train_loader2)]

    for i in range(n_sites):
        train_loader = DataLoader(trainsets[i], batch_size=len(trainsets[i]) // params.nsteps, shuffle=False,
                                  num_workers=params.num_workers)
        correct_preds_local = get_predictions(local_models[i], train_loader, n_train_val[i])
        local_models[i].train()

    loss_all = dict()
    num_data = dict()
    for i in range(n_sites):
        local_models[i].train()
        loss_all[i] = 0
        num_data[i] = 0

    count = 0
    for t in range(params.nsteps):
        for i in range(n_sites):
            optimizers[i].zero_grad()
            inputs, labels, domain, idx = next(data_inters[i])  # get mini-batch for site i
            num_data[i] += labels.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            probs, logits = local_models[i](inputs)  # get output of model i
            loss = class_criterion(logits, labels)  # compute loss

            loss.backward()
            loss_all[i] += loss.item() * labels.size(0)
            optimizers[i].step()  # step for optimizer i

        count += 1

        if (count % params.pace == 0) or t == params.nsteps - 1:
            print('communication - weights update')
            with torch.no_grad():
                for key in global_model.state_dict().keys():
                    # num_batches_tracked is a non trainable LongTensor and
                    # num_batches_tracked are the same for all clients for the given datasets
                    if local_models[0].state_dict()[key].dtype == torch.int64:
                        global_model.state_dict()[key].data.copy_(local_models[0].state_dict()[key])
                    else:
                        temp = torch.zeros_like(global_model.state_dict()[key])
                        # add noise
                        for s in range(n_sites):
                            if params.noise_type == 'G':
                                nn = tdist.Normal(torch.tensor([0.0]),  # 0 mean & std
                                                  params.noise * torch.std(local_models[s].state_dict()[key].detach().cpu()))
                            else:
                                nn = tdist.Laplace(torch.tensor([0.0]),
                                                   params.noise * torch.std(local_models[s].state_dict()[key].detach().cpu()))
                            noise = nn.sample(local_models[s].state_dict()[key].size()).squeeze(-1)
                            noise = noise.to(device)
                            temp += w[s] * (local_models[s].state_dict()[key] + noise)
                        # update global model
                        global_model.state_dict()[key].data.copy_(temp)
                        # update local model
                        for s in range(n_sites):
                            local_models[s].state_dict()[key].data.copy_(global_model.state_dict()[key])

    print('Epoch: {:d} Train: L1 loss: {:.4f}, L2 loss: {:.4f}, L3 loss: {:.4f}'.format(epoch,
                                                                                        loss_all[0]/num_data[0],
                                                                                        loss_all[1]/num_data[1],
                                                                                        loss_all[2]/num_data[2]))

    average_train_loss = 1.0*(loss_all[0]/num_data[0]+loss_all[1]/num_data[1]+loss_all[2]/num_data[2])/n_sites

    print('===HOLOGIC===')
    val_loss0, acc1, targets1, outputs1, preds1 = test(global_model, val_loader0, train=False)
    print('===INBREAST===')
    val_loss1, acc2, targets2, outputs2, preds2 = test(global_model, val_loader1, train=False)
    print('===GE===')
    val_loss2, acc3, targets3, outputs3, preds3 = test(global_model, val_loader2, train=False)

    average_val_loss = 1.0*(val_loss0+val_loss1+val_loss2)/n_sites

    # write summaries
    writer_train.add_scalar('loss', average_train_loss, epoch)
    writer_train.add_scalar('hologic', loss_all[0]/num_data[0], epoch)
    writer_train.add_scalar('inbreast', loss_all[1]/num_data[1], epoch)
    writer_train.add_scalar('ge', loss_all[2]/num_data[2], epoch)

    writer_val.add_scalar('loss', average_val_loss, epoch)
    writer_val.add_scalar('hologic', val_loss0, epoch)
    writer_val.add_scalar('inbreast', val_loss1, epoch)
    writer_val.add_scalar('ge', val_loss2, epoch)

    # save model at minimum validation loss
    if average_val_loss < best_val_loss:
        print('saving model')
        best_val_loss = average_val_loss
        # save model
        if not os.path.exists(PATH):
            os.mkdir(PATH)

        torch.save({
            'epoch': epoch,
            'global_model': global_model.state_dict(),
            'loss': best_val_loss,
        }, os.path.join(PATH, 'model.pt'))

print('Optimization finished!')
