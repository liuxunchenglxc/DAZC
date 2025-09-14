import numpy as np
import torch
from torch import nn
from . import measure

def get_ntk_n(networks, inputs, train_mode=False, num_batch=1, gpu=None):
    if gpu is not None:
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')

    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()

    grads = [[] for _ in range(len(networks))]

    for i in range(num_batch):
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            if gpu is not None:
                inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            else:
                inputs_ = inputs.clone()

            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                grad = []
                for layer in network.modules():
                    if layer._get_name() == 'PatchembedSuper':
                        if layer.sampled_weight.grad is not None:
                            grad.append(layer.sampled_weight.grad.view(-1).detach())
                        # else:
                        #     return torch.zeros_like(layer.sampled_weight)
                    elif isinstance(layer, nn.Linear) and layer.out_features != 1000 and layer.samples:
                        if layer.samples['weight'].grad is not None:
                            grad.append(layer.samples['weight'].grad.view(-1).detach())
                        # else:
                        #     return torch.zeros_like(layer.samples['weight'])
                    elif isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:
                        if layer.samples['weight'].grad is not None:
                            grad.append(layer.samples['weight'].grad.view(-1).detach())
                #for name, W in network.named_parameters():
                #    if 'weight' in name and W.grad is not None:
                #        grad.append(W.grad.view(-1).detach())
                grads[net_idx].append(torch.cat(grad, -1))
                network.zero_grad()
                if gpu is not None:
                    torch.cuda.empty_cache()

    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues, _ = torch.linalg.eigh(ntk)  # ascending
        # conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True))
    return conds


@measure('te_nas', bn=False, copy_net=True)
def compute_NTK_score(model, inputs, targets, loss_fn=None, split_data=1):
    # import time
    # t = time.time()
    gpu=0
    ntk_score = get_ntk_n([model], inputs, train_mode=True, gpu=gpu)[0]
    # print('te_nas time: {:.9f}'.format(time.time() - t)) # debug
    return -1 * ntk_score

