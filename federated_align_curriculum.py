import torch
from torch import nn
import torch.optim as optim
import torch.distributions as tdist
import numpy as np
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
from networks import Classifier, Discriminator
import params
from dataset import load_data
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

EPS = 1e-15


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


def get_curriculum_weights(preds_back, preds_recent):
    comparison = preds_back > preds_recent
    weights = comparison.astype(np.float) + 1.0
    return weights


sites = ['hologic', 'inbreast', 'ge']
n_sites = len(sites)

# model path
PATH = './models/fed-align-cl/'+str(params.noise)+'/'+str(params.nsteps)+'-'+str(params.pace)+'/torch-seed-'+str(params.torch_seed)
print(PATH)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# setup models & optimizers
global_model = Classifier().to(device)

# local models, discriminators, and optimizers
local_models = dict()
discriminators = dict()
optimizers = dict()
optimizerGs = dict()
optimizerDs = dict()
for i in range(n_sites):
    local_models[i] = Classifier().to(device)
    discriminators[i] = Discriminator().to(device)

    optimizers[i] = optim.Adam(local_models[i].parameters(), lr=params.learning_rate)
    optimizerGs[i] = optim.Adam(local_models[i].encoder.parameters(), lr=params.learning_rate)
    optimizerDs[i] = optim.Adam(discriminators[i].parameters(), lr=params.learning_rate)  # weight_decay=1e-3

# loss functions
class_criterion = nn.CrossEntropyLoss()
def advDloss(d1,d2):
    res = -torch.log(d1).mean()-torch.log(1-d2).mean()
    return res
def advGloss(d1,d2):
    res = -torch.log(d1).mean()-torch.log(d2).mean()
    return res.mean()

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

train_loader0 = DataLoader(trainset0, batch_size=len(trainset0)//params.nsteps, shuffle=True)
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

    for i in range(n_sites):
        local_models[i].encoder.load_state_from_shared_weights(
        state_dict=torch.load(image_only_parameters["model_path"])["model"],
        view=image_only_parameters["view"],
    )

    global_model.encoder.load_state_from_shared_weights(
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
track_preds = dict()
n_train_val = [np.asarray(train_loader0.dataset.indices).max() + 1,
               np.asarray(train_loader1.dataset.indices).max() + 1,
               np.asarray(train_loader2.dataset.indices).max() + 1]

print('Start optimization')
for epoch in range(params.n_epochs):
    track_preds[epoch] = dict()

    data_inters = [iter(train_loader0), iter(train_loader1), iter(train_loader2)]

    for i in range(n_sites):
        train_loader = DataLoader(trainsets[i], batch_size=len(trainsets[i]) // params.nsteps, shuffle=False,
                                  num_workers=params.num_workers)
        correct_preds_local = get_predictions(local_models[i], train_loader, n_train_val[i])
        correct_preds_global = get_predictions(global_model, train_loader, n_train_val[i])
        track_preds[epoch][i] = correct_preds_local
        local_models[i].train()

    if epoch > params.n_epochs_adversarial:
        curriculum_weights = []
        for i in range(n_sites):
            weights = get_curriculum_weights(track_preds[epoch - 1][i], track_preds[epoch][i])
            curriculum_weights.append(weights)

        train_sampler0 = WeightedRandomSampler(weights=curriculum_weights[0], num_samples=len(trainset0))
        train_loader0 = DataLoader(trainset0, batch_size=len(trainset0) // params.nsteps, shuffle=False,
                                   num_workers=params.num_workers, sampler=train_sampler0)

        train_sampler1 = WeightedRandomSampler(weights=curriculum_weights[1], num_samples=len(trainset1))
        train_loader1 = DataLoader(trainset1, batch_size=len(trainset1) // params.nsteps, shuffle=False,
                                   num_workers=params.num_workers, sampler=train_sampler1)

        train_sampler2 = WeightedRandomSampler(weights=curriculum_weights[2], num_samples=len(trainset2))
        train_loader2 = DataLoader(trainset2, batch_size=len(trainset2) // params.nsteps, shuffle=False,
                                   num_workers=params.num_workers, sampler=train_sampler2)

        data_inters = [iter(train_loader0), iter(train_loader1), iter(train_loader2)]

    loss_all = dict()
    lossD_all = dict()
    lossG_all = dict()
    num_data = dict()
    num_dataG = dict()
    num_dataD = dict()
    for i in range(n_sites):
        local_models[i].train()
        discriminators[i].train()

        loss_all[i] = 0
        num_data[i] = EPS
        num_dataG[i] = EPS
        lossG_all[i] = 0
        lossD_all[i] = 0
        num_dataD[i] = EPS

    count = 0
    for t in range(params.nsteps):
        # feature space
        fs = []

        # optimize classifier
        for i in range(n_sites):
            optimizers[i].zero_grad()
            inputs, labels, domain, idx = next(data_inters[i])  # get mini-batch for site i
            num_data[i] += labels.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            probs, logits = local_models[i](inputs)  # get output of model i
            loss = class_criterion(logits, labels)  # compute loss

            loss_all[i] += loss.item() * labels.size(0)
            loss.backward(retain_graph=True)
            optimizers[i].step()

            fs.append(local_models[i].encoder(inputs))  # feature space  # .detach() ??

        # optimize alignment
        nn = []
        noises = []
        for i in range(n_sites):
            nn = tdist.Normal(torch.tensor([0.0]), 0.001 * torch.std(fs[i].detach().cpu()))
            noises.append(nn.sample(fs[i].size()).squeeze().to(device))

        for i in range(n_sites):
            for j in range(n_sites):
                if i != j:
                    optimizerDs[i].zero_grad()
                    optimizerGs[i].zero_grad()
                    optimizerGs[j].zero_grad()

                    d1 = discriminators[i](fs[i].detach() + noises[i])
                    d2 = discriminators[i](fs[j].detach() + noises[j])
                    num_dataG[i] += d1.size(0)
                    num_dataD[i] += d1.size(0)
                    lossD = advDloss(d1, d2)
                    lossG = advGloss(d1, d2)

                    lossD_all[i] += lossD.item() * d1.size(0)
                    lossG_all[i] += lossG.item() * d1.size(0)
                    lossG_all[j] += lossG.item() * d2.size(0)
                    lossD = 0.1 * lossD

                    if epoch >= params.n_epochs_adversarial:
                        lossG.backward(retain_graph=True)
                        optimizerGs[i].step()
                        optimizerGs[j].step()

                        lossD.backward(retain_graph=True)  # retain_graph=True works too
                        optimizerDs[i].step()

                    writer_train.add_histogram('Hist/hist_' + sites[i] + '2' + sites[j] + '_source', d1,
                                               epoch * params.nsteps + t)
                    writer_train.add_histogram('Hist/hist_' + sites[i] + '2' + sites[j] + '_target', d2,
                                               epoch * params.nsteps + t)

        count += 1

        if (count % params.pace == 0) or t == params.nsteps - 1:
            # print('communication - weights update')
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
                                nn = tdist.Normal(torch.tensor([0.0]),
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

    print(f"Epoch Number {epoch + 1}")
    print("===========================")
    print("L1: {:.7f}, L2: {:.7f}, L3: {:.7f}".format(loss_all[0] / num_data[0], loss_all[1] / num_data[1], loss_all[2] / num_data[2]))
    print("G1: {:.7f}, G2: {:.7f}, G3: {:.7f}".format(lossG_all[0] / num_dataG[0], lossG_all[1] / num_dataG[1], lossG_all[2] / num_dataG[2]))
    print("D1: {:.7f}, D2: {:.7f}, D3: {:.7f}".format(lossD_all[0] / num_dataD[0], lossD_all[1] / num_dataD[1], lossD_all[2] / num_dataD[2]))
    writer_train.add_scalars('CEloss', {'l1': loss_all[0] / num_data[0], 'l2': loss_all[1] / num_data[1], 'l3': loss_all[2] / num_data[2]}, epoch)
    writer_train.add_scalars('Gloss', {'gl1': lossG_all[0] / num_dataG[0], 'gl2': lossG_all[1] / num_dataG[1], 'gl3': lossG_all[2] / num_dataG[2]}, epoch)
    writer_train.add_scalars('Dloss', {'dl1': lossD_all[0] / num_dataD[0], 'dl2': lossD_all[1] / num_dataD[1], 'dl3': lossD_all[2] / num_dataD[2]}, epoch)

    del fs, inputs

    print('===HOLOGIC===')
    val_loss0, acc1, targets1, outputs1, preds1 = test(global_model, val_loader0, train=False)
    print('===INBREAST===')
    val_loss1, acc2, targets2, outputs2, preds2 = test(global_model, val_loader1, train=False)
    print('===GE===')
    val_loss2, acc3, targets3, outputs3, preds3 = test(global_model, val_loader2, train=False)

    average_train_loss = 1.0*(loss_all[0]/num_data[0]+loss_all[1]/num_data[1]+loss_all[2]/num_data[2])/n_sites
    average_val_loss = 1.0*(val_loss0+val_loss1+val_loss2)/n_sites

    # write summaries
    writer_train.add_scalar('loss', average_train_loss, epoch)

    writer_val.add_scalar('loss', average_val_loss, epoch)

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
