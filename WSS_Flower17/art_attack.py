import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import numpy as np
from scipy import stats


class AttackSaliency(nn.Module):
    def __init__(self,  config , normalize, device):
        super(AttackSaliency, self).__init__()
        self.step_size = config['step_size']
        self.eps = config['epsilon']
        self.n_class = config['n_classes']
        self.num_steps = config['num_steps']
        self.k_top = config['k_top']
        self.criterion = F.cross_entropy
        self.normalize_fn = normalize
        self.device = device
        self.im_size = config['img_size']
        self.target_map = None     
        if 'num_ig_steps' in config:
            self.num_ig_steps = config['num_ig_steps']
        else:
            self.num_ig_steps = 30 
        self.c = 3                   
        self.ranking_loss = nn.SoftMarginLoss()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-15)
        self.softmax_logit = nn.Softmax(dim=1)
        

    def topk_sal_attack(self, model, images, y, min_value, max_value, grad_fn):
        def exemplar_loss_fn(x, g1, g2):
            sim_ap = self.cos(x, g1)  # similarity of attribution positive
            sim_an = self.cos(x, g2)  # similarity of attribution negative
            y = sim_an.new().resize_as_(sim_an).fill_(1)
            loss = self.ranking_loss(sim_ap - sim_an, y)
            return loss
        
        batch_size = images.size(0)
        adv = images.clone()
        with torch.no_grad():
            random_0_to_1 = torch.rand(images.size()).cuda(self.device)
            adv = adv + 2 * self.eps * (random_0_to_1 - 0.5)
            adv = torch.clamp(adv, min_value, max_value)

        adv.requires_grad = True
        scores = model.forward({0: self.normalize_fn(adv), 1: 0})  # 1:0 denotes relu and 1:1 denotes softplus

        # https://pytorch.org/docs/stable/generated/torch.gather.html
        # torch.gather(input, dim, index, *, sparse_grad=False, out=None) -> Tensor.
        # Gathers values along an axis specified by dim.
        # For a 3-D tensor the output is specified by:
        #   given x = index[i][j][k]
        #   output[i][j][k] = input[x][j][k] # if dim == 0
        #   output[i][j][k] = input[i][x][k] # if dim == 1
        #   output[i][j][k] = input[i][j][x] # if dim == 2
        # The output has the same dimension as index;
        # For an element of output tensor, its value is from input, but special subscript
        top_scores = scores.gather(1, index=y.unsqueeze(1))

        # g1 shape:[bs, 3, 224, 224]. bs is batch_size
        # g1 = torch.autograd.grad(top_scores.mean(), adv, retain_graph=True)[0]

        g1 = grad_fn(top_scores.mean(), adv, y, model, self.device)

        # torch.abs(): Computes the absolute value of each element
        g1_adp = torch.mean(torch.abs(g1), 1).reshape(batch_size, -1)

        # https://pytorch.org/docs/stable/generated/torch.topk.html
        # Returns the k largest elements of the given input tensor along a given dimension.
        # A namedtuple of (values, indices) is returned, where the indices are the
        # indices of the elements in the original input tensor.
        _, top_idx_g1 = torch.topk(g1_adp, self.k_top, 1)

        for i in range(self.num_steps):
            adv.requires_grad = True
            # 1:0 denotes relu and 1:1 denotes softplus
            logits_with_softplus = model.forward({0: self.normalize_fn(adv), 1: 1})
            top_scores = logits_with_softplus.gather(1, index=y.unsqueeze(1))
            tmp_arr = [[k for k in range(self.n_class) if k != y[j]] for j in range(y.size(0))]
            other_indices = torch.from_numpy(np.array(tmp_arr)).cuda(self.device)  # non-target index
            other_scores = logits_with_softplus.gather(1, index=other_indices)     # non-target scores
            snd_scores = other_scores.max(dim=1)[0]

            # g1 = torch.autograd.grad(top_scores.mean(), adv, retain_graph=True, create_graph=True)[0]
            # g2 = torch.autograd.grad(snd_scores.mean(), adv, retain_graph=True, create_graph=True)[0]
            # g1.shape: [2, 3, 224, 224]

            g1 = grad_fn(top_scores.mean(), adv, y, model, self.device)
            g2 = grad_fn(snd_scores.mean(), adv, y, model, self.device)

            adv_f = torch.mean(torch.abs(self.normalize_fn(adv)), 1).reshape(batch_size, -1)
            g1_adp = torch.mean(torch.abs(g1), 1).reshape(batch_size, -1)
            g2_adp = torch.mean(torch.abs(g2), 1).reshape(batch_size, -1)
            # g1_adp.shape: [2, 50176]
            
            top_g1 = g1_adp.gather(1, index=top_idx_g1)
            top_g2 = g2_adp.gather(1, index=top_idx_g1)
            top_adv = adv_f.gather(1, index=top_idx_g1)
            # top_g1.shape: [2, 15000]

            exemplar_loss = exemplar_loss_fn(top_adv, top_g1, top_g2)

            # topK_direction = torch.autograd.grad(exemplar_loss, adv)[0]
            topK_direction = grad_fn(exemplar_loss, adv, y, model, self.device, retain_graph=False)

            with torch.no_grad():
                adv = adv + self.step_size * torch.sign(topK_direction)
                adv = torch.clamp(adv, min_value,  max_value)
        # for
        return adv
    

    def saliency_attackIG(self, model , images, y , min_value , max_value , target=None , target_y=None  ):
        
        if target is None:
            labels = y.repeat(self.num_ig_steps)
            images.requires_grad = True
            prefactors = images.new_tensor([k / self.num_ig_steps for k in range(1, self.num_ig_steps + 1)])
            pred_with_relu , _ = model({0:self.normalize_fn(prefactors.view(self.num_ig_steps, 1, 1, 1) * images),1:0})
            pred_with_relu = pred_with_relu.gather(1 , index = labels.unsqueeze(1)).squeeze()

            pred_with_relu = (1 / self.num_ig_steps) * torch.sum(pred_with_relu / prefactors, dim=0)
            orig_grad = torch.autograd.grad(pred_with_relu, images)[0]*images
            orig_grad = torch.sum(torch.abs(orig_grad), dim=1)
            sum_sals = orig_grad.reshape(images.size(0), -1).sum(1)
            orig_grad = self.im_size*self.im_size * torch.div(orig_grad.reshape(images.size(0) , -1), sum_sals)
            flatten_orig_grad = orig_grad.view(-1, self.im_size*self.im_size)
            top_vals , top_idx = torch.topk(flatten_orig_grad, self.k_top ,1)
            elements1 = torch.zeros( (images.size(0) , self.im_size*self.im_size) )
            elements1.scatter_(1 , top_idx.cpu() , 1)
        else:

            labels = target_y.repeat(self.num_ig_steps)
            target.requires_grad = True
            prefactors = target.new_tensor([k / self.num_ig_steps for k in range(1, self.num_ig_steps + 1)])
            pred_with_relu , _ = model({0:self.normalize_fn(prefactors.view(self.num_ig_steps, 1, 1, 1) * target),1:0})
            pred_with_relu = pred_with_relu.gather(1 , index = labels.unsqueeze(1)).squeeze()

            pred_with_relu = (1 / self.num_ig_steps) * torch.sum(pred_with_relu / prefactors, dim=0)
            orig_grad = torch.autograd.grad(pred_with_relu, target)[0]*target
            orig_grad = torch.sum(torch.abs(orig_grad), dim=1)
            sum_sals = orig_grad.reshape(target.size(0), -1).sum(1)
            orig_grad = self.im_size*self.im_size * torch.div(orig_grad.reshape(images.size(0) , -1), sum_sals)
            flatten_orig_grad = orig_grad.view(-1, self.im_size*self.im_size)
            top_vals , top_idx = torch.topk(flatten_orig_grad, self.k_top ,1)
            elements1 = torch.zeros( (target.size(0) , self.im_size*self.im_size) )
            elements1.scatter_(1 , top_idx.cpu() , 1)
        
        adv = images.clone()
        with torch.no_grad():
            adv = adv + 2* self.eps * (torch.rand(images.size()).cuda(self.device) - 0.5)
            adv = torch.clamp(adv , min_value , max_value)
        
        
        list_of_adv = []
        list_of_measure = []
        labels = y.repeat(self.num_ig_steps)
        for i in range(self.num_steps):
            adv.requires_grad = True            
            
            prefactors = images.new_tensor([k / self.num_ig_steps for k in range(1, self.num_ig_steps + 1)])
            inputs = prefactors.view(self.num_ig_steps, 1, 1, 1) * adv
            pred_with_softplus , _  = model.forward({0:self.normalize_fn(inputs),1:1})
            pred_with_softplus = pred_with_softplus.gather(1 , index = labels.unsqueeze(1)).squeeze()
            
            pred_with_softplus_avg = (1 / self.num_ig_steps) * torch.sum(pred_with_softplus / prefactors, dim=0)
            adv_grad = torch.autograd.grad(pred_with_softplus_avg, adv, retain_graph=True, create_graph=True)[0]
            adv_grad = adv_grad * (adv)
            adv_grad = torch.sum(torch.abs(adv_grad), dim=1)
            
            sum_sals = adv_grad.reshape(images.size(0), -1).sum(1)
            adv_grad = self.im_size*self.im_size * torch.div(adv_grad.reshape(images.size(0) , -1), sum_sals)
            flatten_adv_grad = adv_grad.view(-1, self.im_size*self.im_size)
            
            

            pred = model.forward({0:self.normalize_fn(adv) , 1: 0})[0].argmax(1)
            
            if (pred==y)[0].data.item() == 1 :
                list_of_measure.append([-1,i])
            
            list_of_adv.append(adv)                
            topK_loss = (flatten_adv_grad*elements1.cuda(self.device)).sum(1).mean()
            topK_direction = -torch.autograd.grad(topK_loss, adv)[0]            

            with torch.no_grad():
                #remove numpy computation and make it pytorch
                if target is None:
                    adv = adv + self.step_size * torch.sign(topK_direction)
                else:
                    adv = adv - self.step_size * torch.sign(topK_direction)
                adv = torch.min(adv, images+self.eps)
                adv = torch.max(adv, images-self.eps)
                adv = torch.clamp(adv, min_value,  max_value) 
                
        if len(list_of_measure) > 0:
            list_of_measure = sorted(list_of_measure , key = lambda x : x[1])
            index_adv =  list_of_measure[-1][1]        
            return list_of_adv[index_adv]
        else:
            return None
 