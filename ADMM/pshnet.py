import time

import torch
import torch.nn as nn
import torch.optim as optim
from kmeans_pytorch import kmeans
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.nn.functional as F

from models.model_loader import load_model
from models.model_loader import load_model_mo
from utils.evaluate import mean_average_precision, pr_curve


def train(train_dataloader, query_dataloader, retrieval_dataloader, arch, code_length, device, lr,
          max_iter, topk, evaluate_interval, anchor_num, proportion
          ):
    #print("using device")
    #print(torch.cuda.current_device())
    #print(torch.cuda.get_device_name(torch.cuda.current_device()))
    # Load model
    model = load_model(arch, code_length).to(device)
    model_mo = load_model_mo(arch).to(device)

    # Create criterion, optimizer, scheduler
    criterion = PrototypicalLoss()
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        max_iter,
        lr / 100,
    )

    # Initialization
    running_loss = 0.
    best_map = 0.
    training_time = 0.

    # Training
    for it in range(max_iter):
        # timer
        tic = time.time()

        # harvest prototypes/anchors#some times killed, try another way
        with torch.no_grad():
            output_mo = torch.tensor([]).to(device)
            for data, _, _ in train_dataloader:
                data = data.to(device)
                output_mo_temp = model_mo(data)
                output_mo = torch.cat((output_mo, output_mo_temp), 0)
                torch.cuda.empty_cache()

            anchor = get_anchor(output_mo, anchor_num, device)  # compute anchor

        # self-supervised deep learning
        model.train()
        for data, targets, index in train_dataloader:
            data, targets, index = data.to(device), targets.to(device), index.to(device)
            optimizer.zero_grad()

            # output
            output_B = model(data)
            output_mo_batch = model_mo(data)

            # prototypes/anchors based similarity

            #sample_anchor_distance = torch.sqrt(torch.sum((output_mo_batch[:, None, :] - anchor) ** 2, dim=2)).to(device)
            #sample_anchor_dist_normalize = F.normalize(sample_anchor_distance, p=2, dim=1).to(device)
            #S = sample_anchor_dist_normalize @ sample_anchor_dist_normalize.t()

            # loss
            #loss = criterion(output_B, S)
            #running_loss = running_loss + loss.item()
            #loss.backward(retain_graph=True)
            with torch.no_grad():
                dist = torch.sum((output_mo_batch[:, None, :] - anchor.to(device)) ** 2, dim=2)
                k = dist.size(1)
                dist = torch.exp(-1 * dist / torch.max(dist)).to(device)
                Z_su = torch.ones(k, 1).to(device)
                Z_sum = torch.sqrt(dist.mm(Z_su)) + 1e-12
                Z_simi = torch.div(dist, Z_sum).to(device)
                S = (Z_simi.mm(Z_simi.t()))
                S=(2/(torch.max(S)-torch.min(S)))*S-1


            loss = criterion(output_B, S)

            running_loss += loss.item()
            loss.backward()

            optimizer.step()
        with torch.no_grad():
            # momentum update:
            for param_q, param_k in zip(model.parameters(), model_mo.parameters()):
                param_k.data = param_k.data * proportion + param_q.data * (1. - proportion)  # proportion = 0.999 for update

        scheduler.step()
        training_time += time.time() - tic

        # Evaluate
        if it % evaluate_interval == evaluate_interval - 1:
            # Generate hash code
            query_code = generate_code(model, query_dataloader, code_length, device)
            retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)

            query_targets = query_dataloader.dataset.get_onehot_targets()
            retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets()

            # Compute map
            mAP = mean_average_precision(
                query_code.to(device),
                retrieval_code.to(device),
                query_targets.to(device),
                retrieval_targets.to(device),
                device,
                topk,
            )

            # Compute pr curve
            P, R = pr_curve(
                query_code.to(device),
                retrieval_code.to(device),
                query_targets.to(device),
                retrieval_targets.to(device),
                device,
            )

            # Log
            logger.info('[iter:{}/{}][loss:{:.2f}][map:{:.4f}][time:{:.2f}]'.format(
                it + 1,
                max_iter,
                running_loss / evaluate_interval,
                mAP,
                training_time,
            ))
            running_loss = 0.

            # Checkpoint
            if best_map < mAP:
                best_map = mAP

                checkpoint = {
                    'model': model.state_dict(),
                    'qB': query_code.cpu(),
                    'rB': retrieval_code.cpu(),
                    'qL': query_targets.cpu(),
                    'rL': retrieval_targets.cpu(),
                    'P': P,
                    'R': R,
                    'map': best_map,
                }

    return checkpoint


def get_anchor(all_data, anchor_num, device):
    cluster_ids_x, cluster_centers = kmeans(
        X=all_data, num_clusters=anchor_num, distance='euclidean', device=device
    )
    return cluster_centers


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code


class PrototypicalLoss(nn.Module):
    """
    Prototypical semantic loss function.

    Args
        alpha(float): Sigmoid hyper-parameter.
    """

    def __init__(self):
        super(PrototypicalLoss, self).__init__()
        #print('PrototypicalLoss works!')

    def forward(self, H, S):
        # H: batchsize \times code_length; S: similarity matrix [0,1 ]^{batchsize \times batchsize}
        code_length = H.size(1)

        '''
        k = Z.size(1)
        Z_su=torch.ones(k, 1)
        Z_sum=Z.mm(Z_su)+1e-12
        Z_nor=torch.div(Z, Z_sum)
        Z_simi=torch.exp(-1*Z_nor/torch.max(Z_nor))
        S = Z_simi.mm(Z_simi.t()) - 1
        '''
        #loss = torch.mean(H @ H.t() / code_length - S) # ||1/q H*H' - S ||_F^2##old version , but we can use MSE directly

        loss_fn = torch.nn.MSELoss(reduction='mean')# mean of the square of each element
        loss = loss_fn(H @ H.t() / code_length, S)#H @ H.t() / code_length - S

        return loss
