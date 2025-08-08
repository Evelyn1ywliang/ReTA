import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
import torch
import torch.nn.functional as F
import operator
import torch.nn as nn
from info_nce import InfoNCE
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
import clip
from utils import *
import open_clip


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings on specific dataset in yaml format.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='../data/', help='Path to the datasets directory. Default is ../data/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16', 'OpenCLIP'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--n-res', dest='n_res', type=int, default=3, help='number of adjacent class-specific text embedding')


    args = parser.parse_args()

    return args

def InfoNCELoss(A, B):
    loss = InfoNCE(temperature=0.01, reduction='mean')
    return loss(A, B)

def get_text_basis(text_features, n_components=256):

    U, S, Vh = torch.linalg.svd(text_features.float())
    text_basis = Vh[:n_components].to(dtype=text_features.dtype)
    
    return text_basis 

def calculate_stability_consistence_score(preds, ori_pred, gamma, ratio=1.0):

    unique_preds, counts = torch.unique(preds, return_counts=True)
    most_common_idx = torch.argmax(counts)
    most_common_pred = unique_preds[most_common_idx]
    most_common_count = counts[most_common_idx]

    sta_score = preds.shape[1] / most_common_count  
    
    con_score = gamma if most_common_pred.item() != ori_pred else 1.0

    w = sta_score * con_score

    return ratio*torch.log(w)
    

def update_cache(cache, pred, features_loss, shot_capacity, text_features, text_basis, ratio, include_prob_map=False):
    """Update cache with new features and loss using our entropy reweighting strategy, maintaining the maximum shot capacity."""
    with torch.no_grad():
        (img_feat, entropy, *rest) = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]

        # proj & SVD
        proj_img = img_feat @ text_basis.T @ text_basis
        sim_adjacent = torch.einsum("bd, dkn->bkn", proj_img, text_features)
        preds_adjacent = torch.max(sim_adjacent, dim=-1)[1]

        w = calculate_stability_consistence_score(preds_adjacent, pred, gamma=2.0, ratio=ratio)

        # reweighting
        new_entropy = entropy * (1 + w)

        item = [img_feat, new_entropy] if not include_prob_map else [img_feat, new_entropy, rest[-1]]

        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif item[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]
        return w
    
def visualize_cache(cache, iter):
    # t-SNE visualization of cache features
    with torch.no_grad():
        cache_features = []
        cache_labels = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_features.append(item[0].reshape(-1))
                cache_labels.append(class_index)
        cache_features = torch.stack(cache_features, dim=0)
        cache_labels = torch.Tensor(cache_labels).to(torch.int64)
        cache_features = F.normalize(cache_features, dim=1)
        cache_features = cache_features.cpu().numpy()
        cache_labels = cache_labels.cpu().numpy()
        tsne = TSNE(n_components=2)
        print(cache_features.shape)
        cache_features_fit = tsne.fit_transform(cache_features)
        
        # Assign different colors to different cache_labels
        colors = [
            '#00429d',  # Strong Blue
            '#93003a',  # Deep Red
            '#007d34',  # Vivid Green
            '#ff6800',  # Vivid Orange
            '#e30022',  # Bright Red
            '#a6bdd7',  # Light Periwinkle
            '#ffcc00',  # Vivid Yellow
            '#540d6e',  # Dark Violet
            '#7f180d',  # Dark Red
            '#00939c',  # Cyan Process
            '#5f3c99',  # Purplish Blue
            '#ff4a46',  # Bright Red-Orange
            '#8f0075',  # Strong Purple
            '#ff3c38',  # Bright Red
            '#83a697',  # Muted Cyan
            '#1e96be',  # Strong Cyan
            '#d9e021',  # Vivid Lime Green
            '#f18d05',  # Rich Orange
            '#f6e120',  # Bright Yellow
            '#8f2d56',  # Strong Rose
            '#006837',  # Dark Green
            '#e7298a',  # Bright Pink
            '#ce1256',  # Dark Pink
            '#01665e',  # Dark Teal
            '#dfc27d',  # Pale Gold
            '#35978f',  # Muted Teal
            '#bf812d',  # Mustard Brown
            '#543005',  # Dark Brown
            '#8c510a',  # Light Brown
            '#80cdc1',  # Soft Turquoise
        ]
        colors_others = 'gray'
        figure, ax = plt.subplots(1, 1, dpi=600, figsize=(5, 5))
        patch = ax.patch
        patch.set_color("#f5f5f5")
        ax.tick_params(axis='both',          # Changes apply to both x and y axes
               which='both',         # Apply changes to both major and minor ticks
               bottom=False,         # No ticks along the bottom edge
               top=False,            # No ticks along the top edge
               left=False,           # No ticks along the left edge
               right=False,          # No ticks along the right edge
               labelbottom=False,    # No labels along the bottom edge
               labelleft=False)      # No labels along the left edge
        plt.grid(color='w', zorder=0, linewidth=2)
        plt.gca().spines['bottom'].set_color('gray')
        plt.gca().spines['left'].set_color('gray')
        plt.gca().spines['top'].set_color('gray')
        plt.gca().spines['right'].set_color('gray')
        # In Food-101, we have 101 classes
        for i in range(101):
            if i < 30:
                plt.scatter(cache_features_fit[cache_labels == i, 0], cache_features_fit[cache_labels == i, 1], c=colors[i], s=15, marker='x', zorder=5)
            else:
                plt.scatter(cache_features_fit[cache_labels == i, 0], cache_features_fit[cache_labels == i, 1], c=colors_others, s=5, zorder=5)
        save_path = 'fig/cache_features_iter_{}.png'.format(iter)
        plt.savefig(save_path)
        plt.close()
        

def cache_key_value(image_features, cache, alpha, beta, clip_weights):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        all_classes = []
        for class_index in sorted(cache.keys()):
            num_items = len(cache[class_index])
            # Compute the prototype of the class
            image_prototype = torch.zeros_like(image_features)
            for item in cache[class_index]:
                image_prototype += item[0] / num_items
            cache_keys.append(image_prototype)
            cache_values.append(class_index)
            all_classes.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(2))).cuda().half()
            
        return cache_keys, cache_values, all_classes
    
def compute_cache_logits(image_features, cache_keys, cache_values, alpha, beta, clip_weights):
    affinity = image_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits

def compute_gaussian_logits(image_features_x, clip_weights_local, labels, logit_scale, test=False):

    n_res = clip_weights_local.shape[1]  

    clip_weights_local = clip_weights_local.permute(2, 1, 0)
    logit_scale = logit_scale.exp()
    
    text_mean = clip_weights_local.mean(dim=1)
    
    if test:
        logits_img_text = logit_scale * image_features_x @ text_mean.t()
        return None, logits_img_text
    
    
    centered_text = clip_weights_local - text_mean.unsqueeze(1)
    logits_img_text = logit_scale * image_features_x @ text_mean.t()
    
    # Covariance between classes for each feature dimension
    diag_cov = (centered_text.permute(2,0,1) @ centered_text.permute(2,1,0))
    diag_cov.div_(n_res + 1) # [D, C, C]
    
    # L_{b,i,k} = Σ_d x_{b,d}^2 · Σ_{d,i,k} 
    quad_scores = torch.einsum('bd, dck -> bck', image_features_x.pow(2), diag_cov)
    
    # Per-class diagonal term q_{b,c} = L_{b,c,c}
    diag_scores = torch.diagonal(quad_scores, dim1=1, dim2=2)

    # Target-class quadratic term
    target_quad = diag_scores[:, labels].unsqueeze(1) 
    
    # σ_{b,c} = q_{b,y} + q_{b,c} − 2L_{b,y,c}
    sigma = target_quad  + diag_scores - 2 * quad_scores[:, labels, :] # [B, C]
    
    # Gaussian refined logits
    gauss_logits = 0.5 * (logit_scale ** 2) * sigma  # [B, C]
    
    return gauss_logits, logits_img_text
    
class TextResidue(nn.Module):
    def __init__(self, clip_weights, n_res):
        super(TextResidue, self).__init__()
        self.feat_dim, self.n_res, self.cate_num = clip_weights.shape
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, n_res, self.cate_num]).half().cuda(), requires_grad=True)
        
    def forward(self, x):
        new_clip_weights = x.clone() + self.residual
        new_clip_weights = F.normalize(new_clip_weights, dim=0)
        return new_clip_weights
    
    def reset(self):
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cate_num]).half().cuda(), requires_grad=True)
        
class PositiveCacheResidue(nn.Module):
    def __init__(self, pos_cache_keys):
        super(PositiveCacheResidue, self).__init__()
        self.feat_dim, self.cache_size = pos_cache_keys.shape
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cache_size]).half().cuda(), requires_grad=True)
        
    def forward(self, x):
        new_pos_cache_keys = x.clone() + self.residual
        new_pos_cache_keys = F.normalize(new_pos_cache_keys, dim=0)
        return new_pos_cache_keys


def run_test_reta(pos_cfg, lr_cfg, loader, clip_model, clip_weights, dataset_name, n_res=3):   
    with torch.cuda.amp.autocast():
        pos_cache, accuracies = {}, []
        
        pos_enabled = pos_cfg['enabled']
        
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'ratio']}
        
        clip_weights_global = clip_weights.clone()
        num_avg = 0

        # Test-time adaptation
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            clip_weights_local = clip_weights_global.clone().detach()
            text_residue = TextResidue(clip_weights_local, n_res)
            new_clip_weights_all = text_residue(clip_weights_local)
            new_clip_weights = new_clip_weights_all[:, -1]
            text_basis = get_text_basis(new_clip_weights.t(), n_components=64)   # default: 64

            if dataset_name == 'A':
                image_features_x, clip_logits, entropy, prob_map, pred, _, all_img_feat_init = get_clip_logits(images, clip_model, new_clip_weights, get_all_feat=True)
            else:
                image_features_x, clip_logits, entropy, prob_map, pred, _, all_img_feat_init = get_clip_logits(images, clip_model, new_clip_weights, get_all_feat=True, get_first=True)
            target = target.cuda()
            
            if pos_enabled:
                entropy = get_entropy(entropy, clip_weights)
                w_ent = update_cache(pos_cache, pred, [image_features_x, entropy], pos_params['shot_capacity'], clip_weights_local, text_basis, pos_params['ratio'])
                 
                pos_cache_keys, pos_cache_values, all_classes = cache_key_value(image_features_x, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
                pos_cache_keys = pos_cache_keys / pos_cache_keys.norm(dim=0)

            steps = 1 # Update step, set to 1 in default
            for j in range(steps):
                final_logits = clip_logits.clone()
                
                if pos_enabled and pos_cache:
                    final_logits += compute_cache_logits(image_features_x, pos_cache_keys, pos_cache_values, pos_params['alpha'], pos_params['beta'], clip_weights)

                    # text gaussian logits (refined boundary)
                    gauss_logits, gauss_mean_logits = compute_gaussian_logits(all_img_feat_init, text_residue(clip_weights_local), pred, clip_model.logit_scale)  

                    loss = avg_entropy(final_logits)
                    
                    # alignment loss
                    text_proto = new_clip_weights[:, all_classes] if pos_params['ratio'] >=0.5 else new_clip_weights_all[:, :, all_classes].mean(dim=1)
                    image2text_loss = InfoNCELoss(pos_cache_keys.T, text_proto.T)
                    
                    loss += image2text_loss * lr_cfg['align']
                    
                    if w_ent == 0.0:
                        # use reliable samples for updating
                        refined_gaussian_logits = gauss_mean_logits + gauss_logits
                        # choose low entropy
                        gauss_batch_entropy = softmax_entropy(refined_gaussian_logits)
                        gauss_selected_idx = torch.argsort(gauss_batch_entropy, descending=False)[:int(gauss_batch_entropy.size()[0] * 0.1)]
                        filt_gauss_logits = refined_gaussian_logits[gauss_selected_idx]

                        gauss_target = torch.tensor([pred]*len(gauss_selected_idx), device=filt_gauss_logits.device)
                        gauss_loss = lr_cfg['train_w'] * F.cross_entropy(filt_gauss_logits, gauss_target)
                        loss += gauss_loss 
                else:
                    loss = avg_entropy(final_logits)
                
                lr_text = lr_cfg['text']
                if pos_enabled and pos_cache:
                    optimizer = torch.optim.AdamW([
                        {'params': text_residue.parameters(), 'lr': lr_text, 'eps': 1e-3, 'weight_decay': 1e-1},
                        ])
                else:
                    optimizer = torch.optim.AdamW([
                        {'params': text_residue.parameters(), 'lr': lr_text, 'eps': 1e-3, 'weight_decay': 1e-1}
                        ])

                optimizer.zero_grad()
                if j == steps - 1:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                optimizer.step() 

            # Actual inference
            text_residue.eval()
            with torch.no_grad():
                new_clip_weights_all = text_residue(clip_weights_local)
                new_clip_weights = new_clip_weights_all[:, -1]
                if dataset_name == 'A':
                    _, clip_logits, _, _, _, all_clip_logits, all_img_feat = get_multiview_clip_logits(images, clip_model, new_clip_weights, get_mean=True, all_img_feat=all_img_feat_init)
                else:
                    _, clip_logits, _, _, _, all_clip_logits, all_img_feat = get_multiview_clip_logits(images, clip_model, new_clip_weights, all_img_feat=all_img_feat_init)
                final_logits = clip_logits.clone()
                
                all_cache_logits = compute_cache_logits(all_img_feat, pos_cache_keys, pos_cache_values, pos_params['alpha'], pos_params['beta'], clip_weights)   
                all_clip_logits += all_cache_logits
                
                # gaussian refined inference score with cache logits
                _, gauss_infer_logits = compute_gaussian_logits(all_img_feat, text_residue(clip_weights_local), None, clip_model.logit_scale, test=True)   
                gauss_infer_logits += all_cache_logits
    
                # get gaussian low entropy
                batch_entropy = softmax_entropy(gauss_infer_logits)
                selected_idx = (torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]).tolist()
                
                if len(selected_idx) == 0:
                    final_logits = torch.cat([all_clip_logits[0].unsqueeze(0), gauss_infer_logits], dim=0)
                else:
                    final_logits = torch.cat([all_clip_logits[0].unsqueeze(0), gauss_infer_logits[selected_idx]], dim=0)

                acc = cls_acc(final_logits, target.cuda(), lam=pos_cfg['lam'])  
                accuracies.append(acc)
                
                loss = avg_entropy(final_logits[0][None, :])
                
                if get_entropy(loss, clip_weights) < 0.1:
                    # Cumalative Avg
                    num_avg += 1
                    clip_weights_global = clip_weights_global * (num_avg / (num_avg + 1)) + new_clip_weights_all * (1 / (num_avg + 1))
        
            if i % 1000 == 0:
                print("---- ReTA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
    # print("---- ReTA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))   
    

    return sum(accuracies)/len(accuracies)

def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    if args.backbone == 'RN50' or args.backbone == 'ViT-B/16' or args.backbone == 'ViT-L/14':
        clip_model, preprocess = clip.load(args.backbone)
    elif args.backbone == 'OpenCLIP':
        clip_model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
        clip_model = clip_model.to('cuda')
    
    # Run ReTA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        # Set random seed
        random.seed(1)
        torch.manual_seed(1)
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        print(args.backbone)
        
        test_loader, classnames, template, cupl_path = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, cupl_path, clip_model, args.backbone, args.n_res)

        acc = run_test_reta(cfg['positive'], cfg['learning_rate'], test_loader, clip_model, clip_weights, dataset_name)

        print(f"---- Final Acc. of {dataset_name}: {acc:.2f}. ----\n")

if __name__ == "__main__":
    main()