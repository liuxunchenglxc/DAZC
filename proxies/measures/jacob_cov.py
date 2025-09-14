import torch
import numpy as np

from . import measure

def get_batch_jacobian(net, x, target, device, split_data):
    x.requires_grad_(True)

    N = x.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data
        y = net(x[st:en])
        y.backward(torch.ones_like(y))

    jacob = x.grad.detach()
    x.requires_grad_(False)
    return jacob, target.detach()

def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))

@measure('jacob_cov', bn=True, copy_net=False)
def compute_jacob_cov(net, inputs, targets, loss_fn=None, split_data=1):
    # import time 
    # t = time.time() # debug
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()

    jacobs, labels = get_batch_jacobian(net, inputs, targets, device, split_data=split_data)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    try:
        jc = eval_score(jacobs, labels)
    except Exception as e:
        print(e)
        jc = np.nan
    # print('jacob_cov time: {:.9f} s'.format(time.time() - t)) # debug
    return jc