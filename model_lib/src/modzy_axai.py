import torch
import cv2
import numpy as np
import torch.nn.functional as F
from skimage.segmentation import quickshift,slic,mark_boundaries
import copy
from skimage.transform import resize

def split_long_brle_lengths(lengths, dtype=np.int64):
    """Split lengths that exceed max dtype value.
    Lengths `l` are converted into [max_val, 0] * l // max_val + [l % max_val]
    e.g. for dtype=np.uint8 (max_value == 255)
    ```
    split_long_brle_lengths([600, 300, 2, 6], np.uint8) == \
       [255, 0, 255, 0, 90, 255, 0, 45, 2, 6]
    ```
    """
    lengths = np.asarray(lengths)
    max_val = np.iinfo(dtype).max
    bad_length_mask = lengths > max_val
    if np.any(bad_length_mask):
        # there are some bad lenghs
        nl = len(lengths)
        repeats = np.asarray(lengths) // max_val
        remainders = (lengths % max_val).astype(dtype)
        lengths = np.empty(shape=(np.sum(repeats) * 2 + nl,), dtype=dtype)
        np.concatenate(
            [
                np.array([max_val, 0] * repeat + [remainder], dtype=dtype)
                for repeat, remainder in zip(repeats, remainders)
            ],
            out=lengths,
        )
        return lengths
    elif lengths.dtype != dtype:
        return lengths.astype(dtype)
    else:
        return lengths


def dense_to_brle(dense_data, dtype=np.int64):
    """
    Get the binary run length encoding of `dense_data`.
    Args:
      dense_data: rank 1 bool array of data to encode.
      dtype: numpy int type.
    Returns:
      Binary run length encoded rank 1 array of dtype `dtype`.
    """
    if dense_data.dtype != np.bool:
        raise ValueError("`dense_data` must be bool")
    if len(dense_data.shape) != 1:
        raise ValueError("`dense_data` must be rank 1.")
    n = len(dense_data)
    starts = np.r_[0, np.flatnonzero(dense_data[1:] != dense_data[:-1]) + 1]
    lengths = np.diff(np.r_[starts, n])
    lengths = split_long_brle_lengths(lengths, dtype=dtype)
    return maybe_pad_brle(lengths, dense_data[0])


def maybe_pad_brle(lengths, start_value=False):
    """Get a potentially padded version of lengths.
    Args:
    lengths: rank 1 int array
    start_value: bool indicating value corresponding to the first value of
      lengths
    Returns:
    rank 1 array of same dtype as lengths, with an extra zero at the front
      if `start_value`, and an extra zero at the end if the resulting array
      would not have an even number of elements.
    """
    # TODO: doesn't seem like the initial implementation of rle is right padded
    # pad_left = int(start_value)
    # pad_right = (len(lengths) + pad_left) % 2
    # if pad_left + pad_right > 0:
    #     return np.pad(lengths, [pad_left, pad_right], mode='constant')

    pad_left = int(start_value)
    right_pad = 0
    if pad_left > 0:
        return np.pad(lengths, (pad_left, right_pad), mode="constant")
    else:
        return lengths


def rle_encode_mask(mask):
    fortran_ordered_image = mask.T.flatten().astype(bool)
    result = dense_to_brle(fortran_ordered_image).tolist()
    return result

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

class AXAI():
    def __init__(self, model):
        self.model = model
        self.attack_type = "FGSM"

    def explain(self,img,orig_shape):

        # gen adv image
        diff = self._gen_adv(img)

        # filter attacked pixels, keep only strongly-attacked
        thresholded = self._threshold(diff)

        # map to image segments
        explanations = self._mapping2(thresholded,img,kernel_size=8,max_dist=200, ratio=0.1)

        # resize mask back to original shape and rle encode
        explanations_tmp = copy.deepcopy(explanations)
        explanations_tmp[explanations==255.0]=0
        explanations_tmp[explanations!=255.0]=1
        mask_axai = explanations_tmp[:,:,0]
        resized_mask = resize(mask_axai,orig_shape)
        resized_mask[resized_mask!=1] = 0
        rle_mask = rle_encode_mask(resized_mask)      

        #remove noise
        kernel = np.ones((5,5),np.uint8)
        closed_mask = cv2.morphologyEx(resized_mask, cv2.MORPH_CLOSE, kernel)        

        return rle_mask
        
    def _tensor2cuda(self, tensor):
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor

    def _gen_adv(self,inputs):
        with torch.no_grad():
            adv_examples=[]
            
            data = self._tensor2cuda(inputs)
            output = self.model(data)
            pred = torch.max(output, dim=1)[1] # use predicted label as target label
            with torch.enable_grad():
                if self.attack_type == 'FGSM':
                    adv_data, diff = fgsm(self.model, data, pred)
                if self.attack_type == 'PGDM':
                    adv_data, diff= pgdm(self.model, data, pred)

            diff_ex = diff.squeeze().detach().cpu().numpy()

        return diff_ex

    def _mapping2(self,Mask, data_org, K=1,kernel_size=4,max_dist=200, ratio=0.2):
        data_org = data_org.squeeze().detach().cpu().numpy()
        image = np.transpose(data_org, (1, 2, 0))
        segments_orig = quickshift(image.astype(np.float), kernel_size=kernel_size,max_dist=max_dist, ratio=ratio)
        
        values, counts = np.unique(segments_orig, return_counts=True)
        attack_frequency=[]
        attack_intensity=[]

        for i in range(len(values)):
            segments_orig_loc=segments_orig==values[i]
            tmp = np.logical_and(segments_orig_loc,Mask)
            attack_frequency.append(np.sum(tmp))
            attack_intensity.append(np.sum(tmp)/counts[i])
            
        top_attack = np.sort(attack_intensity)[::-1][:K]
        zero_filter = np.zeros(np.array(attack_intensity).shape, dtype=bool)
        for i in range(len(top_attack)):
            intensity_filter = attack_intensity == top_attack[i]
            zero_filter = zero_filter+intensity_filter

        strongly_attacked_list = values[zero_filter]
        un_slightly_attacked_list = np.delete(values, strongly_attacked_list)
        strongly_attacked_image = copy.deepcopy(image)
        for x in un_slightly_attacked_list:
            strongly_attacked_image[segments_orig == x] = (255,255,255)
        
        return strongly_attacked_image
            
    def _threshold(self,diff, percentage=15):
        dif_1=copy.deepcopy(diff[0]) 
        dif_2=copy.deepcopy(diff[1])
        dif_3=copy.deepcopy(diff[2])
        dif_total_1 = copy.deepcopy(dif_1)
        dif_total_2 = copy.deepcopy(dif_2)
        dif_total_3 = copy.deepcopy(dif_3)
        thres_1_1=np.percentile(dif_1, 15)
        thres_1_2=np.percentile(dif_1, 85)
        mask_1_1 = dif_1 < thres_1_1
        mask_1_2 = (dif_1 >= thres_1_1) & (dif_1 < thres_1_2)
        mask_1_3 = dif_1 >= thres_1_2
        dif_total_1[mask_1_2] = 0
        
        thres_2_1=np.percentile(dif_2, 15)
        thres_2_2=np.percentile(dif_2, 85)
        mask_2_1 = dif_2 < thres_2_1
        mask_2_2 = (dif_2 >= thres_2_1) & (dif_2 < thres_2_2)
        mask_2_3 = dif_2 >= thres_2_2
        dif_total_2[mask_2_2] = 0
        
        thres_3_1=np.percentile(dif_3, 15)
        thres_3_2=np.percentile(dif_3, 85)
        mask_3_1 = dif_3 < thres_3_1
        mask_3_2 = (dif_3 >= thres_3_1) & (dif_3 < thres_3_2)
        mask_3_3 = dif_3 >= thres_3_2
        dif_total_3[mask_3_2] = 0

        dif_total = dif_total_1+dif_total_2+dif_total_3

        return dif_total