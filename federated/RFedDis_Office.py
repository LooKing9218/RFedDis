"""
federated learning with different aggregation strategy on domainnet dataset
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torch
import torch.nn as nn
import torch.optim as optim

from utils.data_utils import OfficeDataset
from nets.models import AlexNetDis
import argparse
import time
import copy
import torchvision.transforms as transforms
import random
import numpy as np
import torch.nn.functional as F


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

            # calculate new S
            S_a = classes / u_a
            # calculate new e_k
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
    # annealing_coef = min(5, (global_step) / (args.iters/5))
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

        kl_loss = annealing_coef*torch.exp(-(loss_kl(t_h, s_h)+loss_kl(s_h, t_h)))


        loss = loss_ce + loss_un + kl_loss
        loss_all += loss.item()
        total += target.size(0)
        pred = evidence_a.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total



def test(model,server_model, data_loader, loss_fun, loss_kl, device, global_step, args):
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

            kl_loss = annealing_coef*torch.exp(-(loss_kl(t_h, s_h) + loss_kl(s_h, t_h)))


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
    
def prepare_data(args):
    data_base_path = './data'
    transform_office = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
    ])

    # amazon
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=transform_office)
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)
    # caltech
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=transform_office)
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)
    # dslr
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=transform_office)
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)
    # webcam
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=transform_office)
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    min_data_len = min(len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset))
    val_len = int(min_data_len * 0.4)
    min_data_len = int(min_data_len * 0.5)

    amazon_valset = torch.utils.data.Subset(amazon_trainset, list(range(len(amazon_trainset)))[-val_len:])
    amazon_trainset = torch.utils.data.Subset(amazon_trainset, list(range(min_data_len)))

    caltech_valset = torch.utils.data.Subset(caltech_trainset, list(range(len(caltech_trainset)))[-val_len:])
    caltech_trainset = torch.utils.data.Subset(caltech_trainset, list(range(min_data_len)))

    dslr_valset = torch.utils.data.Subset(dslr_trainset, list(range(len(dslr_trainset)))[-val_len:])
    dslr_trainset = torch.utils.data.Subset(dslr_trainset, list(range(min_data_len)))

    webcam_valset = torch.utils.data.Subset(webcam_trainset, list(range(len(webcam_trainset)))[-val_len:])
    webcam_trainset = torch.utils.data.Subset(webcam_trainset, list(range(min_data_len)))

    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=args.batch, shuffle=True)
    amazon_val_loader = torch.utils.data.DataLoader(amazon_valset, batch_size=args.batch, shuffle=False)
    amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=args.batch, shuffle=False)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=args.batch, shuffle=True)
    caltech_val_loader = torch.utils.data.DataLoader(caltech_valset, batch_size=args.batch, shuffle=False)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=args.batch, shuffle=False)

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=args.batch, shuffle=True)
    dslr_val_loader = torch.utils.data.DataLoader(dslr_valset, batch_size=args.batch, shuffle=False)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=args.batch, shuffle=False)

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=args.batch, shuffle=True)
    webcam_val_loader = torch.utils.data.DataLoader(webcam_valset, batch_size=args.batch, shuffle=False)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=args.batch, shuffle=False)

    train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
    val_loaders = [amazon_val_loader, caltech_val_loader, dslr_val_loader, webcam_val_loader]
    test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]
    return train_loaders, val_loaders, test_loaders

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 12
    np.random.seed(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=True, help='whether to log')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size')
    parser.add_argument('--gpu', type = int, default= 1, help ='batch size')
    parser.add_argument('--num_classes', type = int, default= 10, help ='batch size')
    parser.add_argument('--iters', type = int, default=300, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='fedbn', help='[FedBN | FedAvg | FedProx]')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    exp_folder = 'TrFedDis_annealing_coef_seed_{}_lr_{}'.format(seed,str(args.lr))

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))

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

    train_loaders, val_loaders, test_loaders = prepare_data(args)

    
    # setup model
    server_model = AlexNetDis().to(device)
    loss_fun = nn.CrossEntropyLoss()
    loss_kl = KL_Loss().to(device)

    # name of each datasets
    datasets = ['Amazon', 'Caltech', 'DSLR', 'Webcam']
    # federated client number
    client_num = len(datasets)
    client_weights = [1/client_num for i in range(client_num)]
    
    # each local client model
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    best_changed = False


    best_epoch = 0
    best_acc = [0. for j in range(client_num)]
    start_iter = 0

    # Start training
    for a_iter in range(start_iter, args.iters):
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log:
                logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters)) 

            for client_idx, model in enumerate(models):
                train_loss, train_acc = train_UncertaintyDis(model, train_loaders[client_idx],
                                                        optimizers[client_idx], loss_fun, loss_kl, device, a_iter, args)

        with torch.no_grad():
            # Aggregation
            server_model, models = communication(args, server_model, models, client_weights)

            # Report loss after aggregation
            for client_idx, model in enumerate(models):
                train_loss, train_acc = test(model,server_model, train_loaders[client_idx], loss_fun, loss_kl, device, a_iter, args)
                print(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx] ,train_loss, train_acc))
                if args.log:
                    logfile.write(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx] ,train_loss, train_acc))

            # Validation
            val_acc_list = [None for j in range(client_num)]
            for client_idx, model in enumerate(models):
                val_loss, val_acc = test(model, server_model, val_loaders[client_idx], loss_fun, loss_kl, device, a_iter, args)
                val_acc_list[client_idx] = val_acc
                print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[client_idx], val_loss, val_acc))
                if args.log:
                    logfile.write(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[client_idx], val_loss, val_acc))

            # Record best
            if np.mean(val_acc_list) > np.mean(best_acc):
                for client_idx in range(client_num):
                    best_acc[client_idx] = val_acc_list[client_idx]
                    best_epoch = a_iter
                    best_changed=True
                    print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(datasets[client_idx], best_epoch, best_acc[client_idx]))
                    if args.log:
                        logfile.write(' Best site-{:<10s} | Epoch:{} | Val Acc: {:.4f}\n'.format(datasets[client_idx], best_epoch, best_acc[client_idx]))
        
            if best_changed:     
                print(' Saving the local and server checkpoint to {}...'.format(SAVE_PATH))
                logfile.write(' Saving the local and server checkpoint to {}...\n'.format(SAVE_PATH))
                if args.mode.lower() == 'fedbn':
                    torch.save({
                        'model_0': models[0].state_dict(),
                        'model_1': models[1].state_dict(),
                        'model_2': models[2].state_dict(),
                        'model_3': models[3].state_dict(),
                        'server_model': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter
                    }, SAVE_PATH)
                    best_changed = False
                    for client_idx, datasite in enumerate(datasets):
                        _, test_acc = test(models[client_idx], server_model, test_loaders[client_idx], loss_fun, loss_kl, device,
                                           a_iter, args)
                        print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                        if args.log:
                            logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
                else:
                    torch.save({
                        'server_model': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter
                    }, SAVE_PATH)
                    best_changed = False
                    for client_idx, datasite in enumerate(datasets):
                        _, test_acc = test(server_model, test_loaders[client_idx], loss_fun, device)
                        print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                        if args.log:
                            logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))

        if log:
            logfile.flush()

    if log:
        logfile.flush()
        logfile.close()