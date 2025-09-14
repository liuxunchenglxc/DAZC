import torch
from torch import nn
import torch.nn.functional as F

from . import measure

@torch.no_grad()
@measure('kd_kl_divergence', bn=True, copy_net=False)
def get_kd_kl_divergence(model, inputs, targets, loss_fn, split_data):
    # import time
    # t = time.time()
    model.eval()
    # calculate the output
    output = model(inputs)
    # calculate the distillation loss with KL divergence
    temperature = 5.0
    soft_targets = F.softmax(targets / temperature, dim=1)
    log_probs = F.log_softmax(output / temperature, dim=1)
    distillation_loss = nn.KLDivLoss()(log_probs, soft_targets.detach())
    score = distillation_loss.item()
    # print('kd_kl_divergence time: {:.9f} s'.format(time.time() - t)) # debug
    return score

