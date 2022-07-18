import torch
import numpy as np
import torch.nn.functional as F

def pgdm(net, x, y, loss_criterion=torch.nn.CrossEntropyLoss(), alpha=0.001, eps=0.2, steps=10, radius=0.2, norm=2):
    # perturbations 
    pgd = x.new_zeros(x.shape)
    # create the adversarial input
    adv_x = x + pgd
    for step in range(steps):
        pgd = pgd.detach()
        x = x.detach()
        adv_x = adv_x.clone().detach()
        adv_x.requires_grad = True 
        preds = net(adv_x)
        net.zero_grad()
        loss = loss_criterion(preds, y)
        loss.backward(create_graph=False, retain_graph=False)
        adv_x_grad = adv_x.grad
        if norm == 'inf':
            scaled_adv_x_grad = adv_x_grad.sign()
        else:
            scaled_adv_x_grad = adv_x_grad/adv_x_grad.view(adv_x.shape[0], -1)\
                                .norm(norm, dim=-1).view(-1, 1, 1, 1)
        
        pgd = pgd + (alpha*scaled_adv_x_grad)# create the adversarial input
        if norm == 'inf':
            pgd = torch.clamp(pgd, -radius, radius)
        else:
            mask = pgd.view(pgd.shape[0], -1).norm(norm, dim=1) <= eps
            scaling_factor = pgd.view(pgd.shape[0], -1).norm(norm, dim=1)
            scaling_factor[mask] = eps
            pgd *= eps / scaling_factor.view(-1, 1, 1, 1)
        adv_x = x + pgd 
    return adv_x, pgd

def fgsm(model, original_images, labels, epsilon=0.2, criterion=torch.nn.CrossEntropyLoss(), min_val=0, max_val=1):
    x = original_images.clone()
    x.requires_grad = True 

    model.eval()

    with torch.enable_grad():# this is used because somewhere in the code, calculation of the grad is disabled
        outputs = model(x)# make sure its in eval_mode
        #loss = F.cross_entropy(outputs, labels)
        loss = criterion(outputs, labels)
        
        grads = torch.autograd.grad(loss, x, only_inputs=True)[0]
        reduction_indices = list(range(1, len(grads.shape)))
        l2_norm = torch.sqrt(torch.sum(torch.mul(grads,grads),dim=reduction_indices,keepdim=True)) + epsilon
        signed_grads = grads / l2_norm
        x.data += epsilon * signed_grads
        x.clamp(min_val, max_val)

    return x, epsilon * signed_grads