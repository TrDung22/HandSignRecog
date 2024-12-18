from dataset.dataset import build_dataset
import torch

def vtn_pf_collate_fn_(batch):
    labels = torch.stack([s[2] for s in batch],dim = 0)
    clip = torch.stack([s[0] for s in batch],dim = 0)
    poseflow = torch.stack([s[1] for s in batch],dim = 0) 
    return {'clip':clip,'poseflow':poseflow},labels

def vtn_gcn_collate_fn_(batch):
    clip = torch.stack([s[0] for s in batch], dim = 0)
    poseflow = torch.stack([s[1] for s in batch], dim = 0)
    keypoints = torch.stack([s[2] for s in batch], dim = 0)
    labels = torch.stack([s[3] for s in batch], dim = 0)
    return {'clip':clip, 'poseflow':poseflow, 'keypoints':keypoints},labels

def crossVTN_collate_fn_(batch):
    heatmap = torch.stack([s[0] for s in batch], dim=0)
    clip = torch.stack([s[1] for s in batch],dim = 0)
    poseflow = torch.stack([s[2] for s in batch],dim = 0)
    labels = torch.stack([s[3] for s in batch], dim=0)
    return {'heatmap':heatmap,'clip':clip,'poseflow':poseflow},labels

def vtn_hc_pf_three_view_collate_fn_(batch):
    center_video = torch.stack([s[0] for s in batch],dim = 0)
    left_video = torch.stack([s[2] for s in batch],dim = 0)
    right_video = torch.stack([s[4] for s in batch],dim = 0)
    labels = torch.stack([s[6] for s in batch],dim = 0)

    center_pf = torch.stack([s[1] for s in batch],dim = 0)
    left_pf = torch.stack([s[3] for s in batch],dim = 0)
    right_pf = torch.stack([s[5] for s in batch],dim = 0)
    
    return {'left':left_video,'center':center_video,'right':right_video,'center_pf':center_pf,'left_pf':left_pf,'right_pf':right_pf},labels

def vtn_3_gcn_collate_fn_(batch):
    center_video = torch.stack([s[0] for s in batch],dim = 0)
    left_video = torch.stack([s[3] for s in batch],dim = 0)
    right_video = torch.stack([s[6] for s in batch],dim = 0)
    
    center_pf = torch.stack([s[1] for s in batch],dim = 0)
    left_pf = torch.stack([s[4] for s in batch],dim = 0)
    right_pf = torch.stack([s[7] for s in batch],dim = 0)

    center_kp = torch.stack([s[2] for s in batch],dim = 0)
    left_kp = torch.stack([s[5] for s in batch],dim = 0)
    right_kp = torch.stack([s[8] for s in batch],dim = 0)

    labels = torch.stack([s[9] for s in batch],dim = 0)

    return {'left':left_video,'center':center_video,'right':right_video,'center_pf':center_pf,'left_pf':left_pf,'right_pf':right_pf,
            'center_kp':center_kp,'left_kp':left_kp,'right_kp':right_kp},labels

def distilation_collate_fn_(batch):
    center_video = torch.stack([s[0] for s in batch],dim = 0)
    left_video = torch.stack([s[2] for s in batch],dim = 0)
    right_video = torch.stack([s[4] for s in batch],dim = 0)

    center_pf = torch.stack([s[1] for s in batch],dim = 0)
    left_pf = torch.stack([s[3] for s in batch],dim = 0)
    right_pf = torch.stack([s[5] for s in batch],dim = 0)

    center_clip_no_crop_hand = torch.stack([s[6] for s in batch],dim = 0)
    labels = torch.stack([s[7] for s in batch],dim = 0)
    
    return {'left':left_video,'center':center_video,'right':right_video,
            'center_pf':center_pf,'left_pf':left_pf,'right_pf':right_pf,
            'center_clip_no_crop_hand':center_clip_no_crop_hand
            },labels

def build_dataloader(cfg, split, is_train=True, model = None,labels = None):
    dataset = build_dataset(cfg['data'], split,model,train_labels = labels)

    if cfg['data']['model_name'] == '2s-CrossVTN':
        collate_func = crossVTN_collate_fn_
    if cfg['data']['model_name'] == 'VTNGCN':
        collate_func = vtn_gcn_collate_fn_
    if cfg['data']['model_name'] == 'VTN3GCN':
        collate_func = vtn_3_gcn_collate_fn_
    if cfg['data']['model_name'] == 'vtn_att_poseflow' or 'HandCrop' in cfg['data']['model_name'] or cfg['data']['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference':
        collate_func = vtn_pf_collate_fn_
    if cfg['data']['model_name']  == 'VTNHCPF_Three_view' or cfg['data']['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation':
        collate_func = vtn_hc_pf_three_view_collate_fn_

    dataloader = torch.utils.data.DataLoader(dataset,
                                            collate_fn = collate_func,
                                            batch_size = cfg['training']['batch_size'],
                                            num_workers = cfg['training'].get('num_workers',2),
                                            shuffle = is_train,
                                            prefetch_factor = cfg['training'].get('prefetch_factor',2),
                                            # pin_memory=True,
                                            persistent_workers =  True,
                                            # sampler = sampler
                                            )

    return dataloader
