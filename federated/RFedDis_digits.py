"""
federated learning with different aggregation strategy on benchmark exp.
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
from torch import nn, optim
import time
import copy
from nets.models import DigitModelDis
import argparse
import numpy as np
import torchvision.transforms as transforms
from utils import data_utils
import torch.nn.functional as F

def prepare_data(args):
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # MNIST
    mnist_trainset     = data_utils.DigitsDataset(data_path="./Dataset/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist)
    mnist_testset      = data_utils.DigitsDataset(data_path="./Dataset/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist)

    # SVHN
    svhn_trainset      = data_utils.DigitsDataset(data_path='./Dataset/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
    svhn_testset       = data_utils.DigitsDataset(data_path='./Dataset/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn)

    # USPS
    usps_trainset      = data_utils.DigitsDataset(data_path='./Dataset/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps)
    usps_testset       = data_utils.DigitsDataset(data_path='./Dataset/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps)

    # Synth Digits
    synth_trainset     = data_utils.DigitsDataset(data_path='./Dataset/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=transform_synth)
    synth_testset      = data_utils.DigitsDataset(data_path='./Dataset/SynthDigits/', channels=3, percent=args.percent,  train=False, transform=transform_synth)

    # MNIST-M
    mnistm_trainset     = data_utils.DigitsDataset(data_path='./Dataset/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm)
    mnistm_testset      = data_utils.DigitsDataset(data_path='./Dataset/MNIST_M/', channels=3, percent=args.percent,  train=False, transform=transform_mnistm)

    mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
    mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
    svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
    usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
    usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
    synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True)
    synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False)
    mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True)
    mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False)

    train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
    test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]

    return train_loaders, test_loaders


# loss function
def KL(alpha, c, device):
    beta = torch.ones((1, c)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step, device):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c, device)

    return (A + B)





def DS_Combin(alpha,classes):

        def DS_Combin_two(alpha1, alpha2):

            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = classes/S[v]
            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))

            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            S_a = classes / u_a
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

class KL_Loss(nn.Module):
    def __init__(self):
        super(KL_Loss, self).__init__()

    def forward(self, features_t, features_s):
        kl_loss = F.kl_div(F.log_softmax(features_s, dim=1), F.softmax(features_t, dim=1), reduction='mean')
        return kl_loss


def train_UncertaintyDis(model, data_loader, optimizer, loss_fun, loss_kl, device, global_step, args):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    annealing_coef = min(1, global_step / args.iters)
    for data, target in data_loader:
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output_t, output_s, t_h, s_h = model(data)

        loss_ce = loss_fun(F.log_softmax(output_t, dim=1), target) + loss_fun(F.log_softmax(output_s, dim=1), target)
        evidences = [F.softplus(output_t), F.softplus(output_s)]

        loss_un = 0
        alpha = dict()
        for v_num in range(2):
            # step two
            alpha[v_num] = evidences[v_num] + 1
            # step three
            loss_un += ce_loss(target, alpha[v_num], args.num_classes, global_step, args.iters, device)
        # step four
        alpha_a = DS_Combin(alpha, args.num_classes)
        evidence_a = alpha_a - 1
        loss_un += ce_loss(target, alpha_a, args.num_classes, global_step, args.iters, device)
        loss_un = torch.mean(loss_un)
        kl_loss = annealing_coef * torch.exp(-(loss_kl(t_h, s_h) + loss_kl(s_h, t_h)))

        loss = loss_ce + loss_un + kl_loss
        loss_all += loss.item()
        total += target.size(0)
        pred = evidence_a.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total



def test(model, server_model, data_loader, loss_fun, loss_kl, device, global_step, args):
    model.eval()
    server_model.eval()
    loss_all = 0
    total = 0
    correct = 0
    annealing_coef = min(1, (global_step) / (args.iters/1))
    with torch.no_grad():
        for data, target in data_loader:

            data = data.to(device)
            target = target.to(device)
            output_t, output_s, t_h, s_h = model(data)

            loss_ce = loss_fun(F.log_softmax(output_t,dim=1), target) + loss_fun(F.log_softmax(output_s,dim=1), target)
            evidences = [F.softplus(output_t), F.softplus(output_s)]

            loss_un = 0
            alpha = dict()
            for v_num in range(2):
                # step two
                alpha[v_num] = evidences[v_num] + 1
                # step three
                loss_un += ce_loss(target, alpha[v_num], args.num_classes, global_step, args.iters, device)

            # step four
            alpha_a = DS_Combin(alpha, args.num_classes)
            evidence_a = alpha_a - 1
            loss_un += ce_loss(target, alpha_a, args.num_classes, global_step, args.iters, device)
            loss_un = torch.mean(loss_un)

            kl_loss = annealing_coef * torch.exp(-(loss_kl(t_h, s_h) + loss_kl(s_h, t_h)))

            loss = loss_ce + loss_un + kl_loss
            loss_all += loss.item()
            total += target.size(0)
            pred = evidence_a.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

        return loss_all / len(data_loader), correct/total
################# Key Function ########################
def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        for key in server_model.state_dict().keys():
            if 'bn' not in key:
                if 'teacher' in key or 'features' in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 

    print('Device:', device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=True, help ='whether to make a log')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size')
    parser.add_argument('--gpu', type = int, default= 1, help ='batch size')
    parser.add_argument('--num_classes', type = int, default= 10, help ='batch size')
    parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='fedbn', help='fedavg | fedprox | fedbn')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    exp_folder = 'RFedDis'

    args.save_path = os.path.join(args.save_path, exp_folder)
    
    log = args.log
    if log:
        log_path = os.path.join('../logs/RFedDis/', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'{}.log'.format(args.mode)), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    iters: {}\n'.format(args.iters))
        logfile.write('    wk_iters: {}\n'.format(args.wk_iters))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))
   
   
    server_model = DigitModelDis().to(device)
    loss_fun = nn.CrossEntropyLoss()
    loss_kl = KL_Loss().to(device)

    # prepare the data
    train_loaders, test_loaders = prepare_data(args)

    # name of each client dataset
    datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
    
    # federated setting
    client_num = len(datasets)
    client_weights = [1/client_num for i in range(client_num)]
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]


    resume_iter = 0

    # start training
    for a_iter in range(resume_iter, args.iters):
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log: logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters)) 
            
            for client_idx in range(client_num):
                train_loss, train_acc = train_UncertaintyDis(models[client_idx], train_loaders[client_idx], optimizers[client_idx], loss_fun, loss_kl, device, a_iter, args)
         
        # aggregation
        server_model, models = communication(args, server_model, models, client_weights)
        
        # report after aggregation
        for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                train_loss, train_acc = test(model,server_model, train_loader, loss_fun, loss_kl, device, a_iter, args)

                print(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx] ,train_loss, train_acc))
                if args.log:
                    logfile.write(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx] ,train_loss, train_acc))\

        # start testing
        for test_idx, test_loader in enumerate(test_loaders):
            test_loss, test_acc = test(models[test_idx], server_model, test_loader, loss_fun, loss_kl, device, a_iter, args)
            print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss, test_acc))
            if args.log:
                logfile.write(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_loss, test_acc))

    # Save checkpoint
    print(' Saving checkpoints to {}...'.format(SAVE_PATH))
    torch.save({
        'model_0': models[0].state_dict(),
        'model_1': models[1].state_dict(),
        'model_2': models[2].state_dict(),
        'model_3': models[3].state_dict(),
        'model_4': models[4].state_dict(),
        'server_model': server_model.state_dict(),
    }, SAVE_PATH)

    if log:
        logfile.flush()
        logfile.close()


