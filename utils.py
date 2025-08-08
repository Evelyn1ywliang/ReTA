import os
import yaml
import torch
import math
import numpy as np
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import json

import open_clip
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import torch
import torch.nn as nn

class TextEncoderWithPrompt(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1, lam=0.167):

    probs = F.softmax(output, dim=1)  
    gauss_mean = torch.mean(probs[1:], dim=0, keepdim=True)  
    w_prob = lam * probs[0] + (1 - lam) * gauss_mean  
        
    pred = w_prob.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    
    return acc


def clip_classifier(classnames, template, cupl_path, clip_model, backbone='RN50', nres=3):
    f = open(cupl_path)
    cupl = json.load(f)
    
    if backbone == 'OpenCLIP':
        tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts += cupl[classname]
            
            if backbone == 'RN50' or backbone == 'ViT-B/16' or backbone == 'ViT-L/14':
                texts = clip.tokenize(texts).cuda()
            elif backbone == 'OpenCLIP':
                texts = tokenizer(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # class_embedding = class_embeddings.mean(dim=0)
            # class_embedding /= class_embedding.norm()

            # compute relative similarity of prompts
            sim_mat = torch.matmul(class_embeddings, class_embeddings.T)
            sim_mat.fill_diagonal_(0)
            total_similarity = sim_mat.sum(dim=1)
            sorted_indices = total_similarity.argsort(descending=False)

            # Ascending progressive binning
            n_prompts = class_embeddings.shape[0]
            reps = []
            for i in range(1, nres + 1):
                end = i * (n_prompts // nres) if i != nres else n_prompts
                emb = class_embeddings[sorted_indices[:end]].mean(dim=0)
                reps.append(emb / emb.norm())       

            class_embedding = torch.stack(reps, dim=1)
            clip_weights.append(class_embedding)
 
        clip_weights = torch.stack(clip_weights, dim=2).cuda()          
    return clip_weights


def get_clip_logits(images, clip_model, clip_weights, get_views_logits=False, get_all_feat=False, get_first=False):
    # with torch.no_grad():
    if isinstance(images, list):
        images = torch.cat(images, dim=0).cuda()
    else:
        images = images.cuda()
    
    # Change 3D tensor to 4D tensor
    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    all_img_feat=None
    if get_all_feat:
        all_img_feat = image_features

    clip_logits = 100. * image_features @ clip_weights
    
    if get_first:
        image_features = image_features[0].unsqueeze(0)
        clip_logits = clip_logits[0].unsqueeze(0)

    if image_features.size(0) > 1:
        batch_entropy = softmax_entropy(clip_logits)
        selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
        output = clip_logits[selected_idx]
        select_feat = image_features[selected_idx]
        image_features = image_features[selected_idx].mean(0).unsqueeze(0)
        if get_views_logits:
            clip_logits = output
        else:
            clip_logits = output.mean(0).unsqueeze(0)

        loss = avg_entropy(output)
        prob_map = output.softmax(1).mean(0).unsqueeze(0)
        pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
    else:
        loss = softmax_entropy(clip_logits)
        prob_map = clip_logits.softmax(1)
        pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])
        select_feat = None

    return image_features, clip_logits, loss, prob_map, pred, select_feat, all_img_feat

def get_multiview_clip_logits(images, clip_model, clip_weights, get_mean=False, all_img_feat=None):
    # with torch.no_grad():
    if isinstance(images, list):
        images = torch.cat(images, dim=0).cuda()
    else:
        images = images.cuda()
    
    # Change 3D tensor to 4D tensor
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
        
    if all_img_feat is not None:
        image_features = all_img_feat
    else:
        image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    clip_logits = 100. * image_features @ clip_weights

    batch_entropy = softmax_entropy(clip_logits)
    selected_idx = (torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]).tolist()

    if get_mean:
        output = clip_logits[selected_idx].mean(0).unsqueeze(0)
        image_features = image_features[selected_idx].mean(0).unsqueeze(0)
        clip_logits = output
        select_feat = image_features

        loss = avg_entropy(output)
        prob_map = output.softmax(1).mean(0).unsqueeze(0)
        pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
    
    else:
        if 0 not in selected_idx:
            selected_idx.insert(0, 0)  
        else:
            zero_pos = selected_idx.index(0)  
            selected_idx.pop(zero_pos) 
            selected_idx.insert(0, 0) 
        output = clip_logits[selected_idx]
        select_feat = image_features[selected_idx]

        loss = avg_entropy(output)
        prob_map = output.softmax(1).mean(0).unsqueeze(0)
        pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())

    return select_feat, output, loss, prob_map, pred, clip_logits, image_features

def get_preprocess(is_ood=False):
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                             std=[0.5, 0.5, 0.5]) # For OpenCLIP
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    if is_ood:
        aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=False)
    else:
        aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    return aug_preprocess


def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "S"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"
    
    config_file = os.path.join(config_path, config_name)
    
    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess):
    if dataset_name == 'I':
        preprocess = get_preprocess(is_ood=True)
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=4, shuffle=True, pin_memory=True)
    
    elif dataset_name in ['A','V','R','S']:
        preprocess = get_preprocess(is_ood=True)
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ['caltech101','dtd','eurosat','fgvc','food101','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']:
        preprocess = get_preprocess()
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    else:
        raise "Dataset is not from the chosen list"
    
    return test_loader, dataset.classnames, dataset.template, dataset.cupl_path