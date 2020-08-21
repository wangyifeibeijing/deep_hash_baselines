import time
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from kmeans_pytorch import kmeans
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.model_loader import load_model
from models.model_loader import load_model_mo
from utils.evaluate import mean_average_precision, pr_curve
import numpy as np

global_rho1 = 1e-2
#ρ1
global_rho2 = 1e-2
#ρ2
global_rho3 = 1e-3
#µ1
global_rho4 = 1e-3
#µ2
global_gamma = 1e-3
global_Z1=np.ones((1,1))
global_Z2=np.ones((1,1))
global_Y1=np.ones((1,1))
global_Y2=np.ones((1,1))
global_A=np.ones((1,1))

def train(train_dataloader, query_dataloader, retrieval_dataloader, arch, code_length, device, lr,
          max_iter, topk, evaluate_interval, anchor_num, proportion
          ):
    rho1 = 1e-2
    #ρ1
    rho2 = 1e-2
    #ρ2
    rho3 = 1e-3
    #µ1
    rho4 = 1e-3
    #µ2
    gamma = 1e-3
    #γ
    with torch.no_grad():
        data_mo = torch.tensor([]).to(device)
        for data, _, _ in train_dataloader:
            data = data.to(device)
            data_mo = torch.cat((data_mo, data), 0)
            torch.cuda.empty_cache()
        n = data_mo.size(1)
        Y1 = torch.rand(n, code_length).to(device)
        Y2 = torch.rand(n, code_length).to(device)
        B=torch.rand(n,code_length).to(device)
    # Load model
    model = load_model(arch, code_length).to(device)
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

        # ADMM use anchors in first step but drop them later
        '''
        with torch.no_grad():
            output_mo = torch.tensor([]).to(device)
            for data, _, _ in train_dataloader:
                data = data.to(device)
                output_mo_temp = model_mo(data)
                output_mo = torch.cat((output_mo, output_mo_temp), 0)
                torch.cuda.empty_cache()
            anchor = get_anchor(output_mo, anchor_num, device)  # compute anchor
        '''
        with torch.no_grad():
            output_mo = torch.tensor([]).to(device)
            data_mo = torch.tensor([]).to(device)
            for data, _, _ in train_dataloader:
                output_B, output_A = model(data)
                output_mo = torch.cat((output_mo, output_A), 0)
                torch.cuda.empty_cache()

            dist = euclidean_dist(output_mo, output_mo)
            dist = torch.exp(-1 * dist / torch.max(dist)).to(device)
            A = (2 / (torch.max(dist) - torch.min(dist))) * dist - 1
            global_A=A.numpy()
            Z1 = B + 1 / rho1 * Y1
            Z1[Z1 > 1] = 1
            Z1[Z1 > -1] = -1
            Z2 = B + 1 / rho2 * Y2
            norm_B = torch.norm(Z2)
            Z2 = torch.sqrt(n * code_length) * Z2 / norm_B
            Y1 = Y1 + gamma * rho1 * (B - Z1)
            Y2 = Y2 + gamma * rho2 * (B - Z2)
            global_Z1=Z1.numpy()
            global_Z2=Z2.numpy()
            global_Y1 = Y1.numpy()
            global_Y2 = Y2.numpy()
            B0 = B.numpy()
            B= torch.from_numpy(scipy.optimize.fmin_l_bfgs_b(Baim_func, B0)).to(device)
        # self-supervised deep learning
        model.train()
        for data, targets, index in train_dataloader:
            data, targets, index = data.to(device), targets.to(device), index.to(device)
            optimizer.zero_grad()

            # output_B for hash code .output_A for result without hash layer
            output_B, output_A= model(data)

            loss = criterion(output_B, B)

            running_loss += loss.item()
            loss.backward()

            optimizer.step()

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

def Baim_func(b):
    [r,n]=b.shape
    ones=np.ones((n,1))
    IR=np.identity(r)
    L=np.diag(global_A.dot(ones))-global_A
    G=global_Y1+global_Y2-global_rho1*global_Z1-global_rho2*global_Z2
    aim =np.trace(b.dot(L.dot(b.T)))+global_rho3*0.25*np.norm(b.dot(b.T)-IR)+global_rho4*0.5*np.norm(b.dot(ones))\
         +(global_rho1+global_rho2)*0.5*np.norm(b)+np.trace(b.dot(G.T))
    return aim

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

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


    def forward(self, H, B):



        loss_fn = torch.nn.MSELoss(reduction='mean')# mean of the square of each element
        loss = loss_fn(H , B)#H @ H.t() / code_length - S

        return loss
