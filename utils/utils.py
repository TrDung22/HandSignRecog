import torch.nn as nn
import torch.optim as optim
from modelling.vtn_att_poseflow_model import VTNHCPF,VTNHCPF_GCN,VTNHCPF_Three_View,VTN3GCN,VTNHCPF_OneView_Sim_Knowledge_Distilation,VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference
from modelling.crossVTN_model import TwoStreamCrossVTN, TwoStreamCrossViewVTN
import torch
from trainer.tools import MyCustomLoss,OLM_Loss
from torchvision import models
from torch.nn import functional as F
from collections import OrderedDict
from pytorch_lightning.utilities.migration import pl_legacy_patch

def load_criterion(train_cfg):
    criterion = None
    if train_cfg['criterion'] == "MyCustomLoss":
        criterion = MyCustomLoss(label_smoothing=train_cfg['label_smoothing'])
    if train_cfg['criterion'] == "OLM_Loss": 
        criterion = OLM_Loss(label_smoothing=train_cfg['label_smoothing'])
    assert criterion is not None
    return criterion

def load_optimizer(train_cfg,model):
    optimzer = None
    if train_cfg['optimzer'] == "SGD":
        optimzer = optim.SGD(model.parameters(), lr=train_cfg['learning_rate'],weight_decay=float(train_cfg['w_decay']),momentum=0.9,nesterov=True)
    if train_cfg['optimzer'] == "Adam":
        optimzer = optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'],weight_decay=float(train_cfg['w_decay']))
    assert optimzer is not None
    return optimzer

def load_lr_scheduler(train_cfg,optimizer):
    scheduler = None
    if train_cfg['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=train_cfg['scheduler_factor'], patience=train_cfg['scheduler_patience'])
    if train_cfg['lr_scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['lr_step_size'], gamma=train_cfg['gamma'])
    assert scheduler is not None
    return scheduler

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    try:
        if m.weight is not None:
            m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    except:
        pass


def load_model(cfg):
    if cfg['training']['pretrained']:
        print(f"load pretrained model: {cfg['training']['pretrained_model']}")
        if cfg['data']['model_name'] == 'vtn_att_poseflow':
            if ('.ckpt' in cfg['training']['pretrained_model']):
                model = VTNHCPF(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
                new_state_dict = {}
                with pl_legacy_patch():
                    checkpoint = torch.load(cfg['training']['pretrained_model'], map_location='cpu')['state_dict']
                    for key, value in checkpoint.items():
                        new_key = key.replace('model.', '')
                        if not new_key.startswith('feature_extractor'):
                            new_state_dict[new_key] = value
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model = VTNHCPF(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
                # Hot fix cause cannot find file .ckpt
                # Only need the line below in root repo:
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
                # Fix path:
                # new_state_dict = {}
                # for key, value in torch.load(cfg['training']['pretrained_model'],map_location='cpu').items():
                #         new_state_dict[key.replace('model.','')] = value
                # model.reset_head(226) # AUTSL
                # model.load_state_dict(new_state_dict)
                # model.reset_head(model.num_classes)
        elif cfg['data']['model_name'] == 'VTNGCN':
            model = VTNHCPF_GCN(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            if '.ckpt' in cfg['training']['pretrained_model']: #Temporary
                new_state_dict = {}
                with pl_legacy_patch():
                    checkpoint = torch.load(cfg['training']['pretrained_model'], map_location='cpu')['state_dict']
                    for key, value in checkpoint.items():
                        new_key = key.replace('model.', '')
                        # if not new_key.startswith('bottle_mm') and not new_key.startswith('self_attention_decoder') and not new_key.startswith('classifier'):
                        # if not new_key.startswith('bottle_mm') and not new_key.startswith('self_attention_decoder'):
                        if not new_key.startswith('bottle_mm') and not new_key.startswith('classifier'):
                        # if not new_key.startswith('classifier'):
                            new_state_dict[new_key] = value
                model.load_state_dict(new_state_dict, strict=False)
                model.reset_head(model.num_classes)
            else:
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))

        elif cfg['data']['model_name'] == '2s-CrossVTN':
            model = TwoStreamCrossVTN(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            if '.ckpt' in cfg['training']['pretrained_model']:
                new_state_dict = model.state_dict()
                with pl_legacy_patch():
                    state_dict = torch.load('VTN_HCPF.ckpt', map_location='cpu')['state_dict']

                pretrained_dict = {}
                for k, v in state_dict.items():
                    # Ánh xạ các trọng số của feature_extractor nếu cần
                    if k.startswith('feature_extractor'):
                        pretrained_dict[f'feature_extractor_rgb.{k[len("feature_extractor."):]}'] = v
                        pretrained_dict[f'feature_extractor_heatmap.{k[len("feature_extractor."):]}'] = v
                    # Ánh xạ các trọng số của self_attention_decoder
                    elif k.startswith('self_attention_decoder.layers.'):
                        parts = k.split('.')
                        layer_idx = parts[2]  # Ví dụ: '0', '1', ...
                        sub_module = parts[3]  # Ví dụ: 'slf_attn', 'pos_ffn', ...
                        param = '.'.join(parts[4:])  # Ví dụ: 'w_qs', 'layer_norm.a_2', ...

                        # 1. Ánh xạ các trọng số Self-Attention (w_qs, w_ks, w_vs)
                        if sub_module == 'slf_attn':
                            # Ánh xạ cho stream1.self_attn
                            new_key = f'cross_attention.layers.{layer_idx}.stream1.self_attn.{param}'
                            pretrained_dict[new_key] = v
                            # Ánh xạ cho stream1.cross_attn
                            new_key = f'cross_attention.layers.{layer_idx}.stream1.cross_attn.{param}'
                            pretrained_dict[new_key] = v
                            # Ánh xạ cho stream2.self_attn
                            new_key = f'cross_attention.layers.{layer_idx}.stream2.self_attn.{param}'
                            pretrained_dict[new_key] = v
                            # Ánh xạ cho stream2.cross_attn
                            new_key = f'cross_attention.layers.{layer_idx}.stream2.cross_attn.{param}'
                            pretrained_dict[new_key] = v

                        # 2. Ánh xạ các tham số Layer Normalization (a_2, b_2) cho Self-Attention
                        elif sub_module == 'slf_attn.layer_norm':
                            new_key1 = f'cross_attention.layers.{layer_idx}.stream1.self_attn.layer_norm.{param}'
                            pretrained_dict[new_key1] = v
                            new_key2 = f'cross_attention.layers.{layer_idx}.stream1.cross_attn.layer_norm.{param}'
                            pretrained_dict[new_key2] = v
                            new_key3 = f'cross_attention.layers.{layer_idx}.stream2.self_attn.layer_norm.{param}'
                            pretrained_dict[new_key3] = v
                            new_key4 = f'cross_attention.layers.{layer_idx}.stream2.cross_attn.layer_norm.{param}'
                            pretrained_dict[new_key4] = v

                        # 3. Ánh xạ các trọng số Feed Forward (w_1.weight, w_1.bias, ...)
                        elif sub_module == 'pos_ffn':
                            new_key1 = f'cross_attention.layers.{layer_idx}.stream1.feed_forward.{param}'
                            pretrained_dict[new_key1] = v
                            new_key2 = f'cross_attention.layers.{layer_idx}.stream2.feed_forward.{param}'
                            pretrained_dict[new_key2] = v

                        # 4. Ánh xạ các tham số Layer Normalization cho Feed Forward (a_2, b_2)
                        elif sub_module == 'pos_ffn.layer_norm':
                            new_key1 = f'cross_attention.layers.{layer_idx}.stream1.feed_forward.layer_norm.{param}'
                            pretrained_dict[new_key1] = v
                            new_key2 = f'cross_attention.layers.{layer_idx}.stream2.feed_forward.layer_norm.{param}'
                            pretrained_dict[new_key2] = v

                    # Ánh xạ trọng số position_encoding nếu cần
                    elif k.startswith('self_attention_decoder.position_encoding.enc.weight'):
                        pretrained_dict['cross_attention.position_encoding.enc.weight'] = v

                new_state_dict.update(pretrained_dict)
                model.load_state_dict(new_state_dict, strict=False)
                model.reset_head(model.num_classes)
            else:
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))

        elif cfg['data']['model_name'] == 'VTNHCPF_Three_view':
            model = VTNHCPF_Three_View(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            if '.ckpt' in cfg['training']['pretrained_model']:
                new_state_dict = {}
                with pl_legacy_patch():
                    for key, value in torch.load(cfg['training']['pretrained_model'],map_location='cpu')['state_dict'].items():
                        new_state_dict[key.replace('model.','')] = value
                model.center.reset_head(226) # AUTSL
                model.left.reset_head(226) # AUTSL
                model.right.reset_head(226) # AUTSL
                # load autsl ckpt
                model.center.load_state_dict(new_state_dict)
                model.right.load_state_dict(new_state_dict)
                model.left.load_state_dict(new_state_dict)
                # add backbone
                model.add_backbone()
                # remove center, left and right backbone
                model.remove_head_and_backbone()
                model.freeze(layers = 0)
                print("Load VTNHCPF Three View")
            elif "IMAGENET" == cfg['training']['pretrained_model']:
                model.add_backbone()
                model.remove_head_and_backbone()
                print("Load VTNHCPF Three View IMAGENET")
            else:
                model.add_backbone()
                model.remove_head_and_backbone()
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
        
            print("Load VTNHCPF Three View")

        elif cfg['data']['model_name'] == 'VTN3GCN':
            model = VTN3GCN(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            if '.ckpt' in cfg['training']['pretrained_model']:
                new_state_dict = {}
                with pl_legacy_patch():
                    for key, value in torch.load(cfg['training']['pretrained_model'],map_location='cpu')['state_dict'].items():
                        new_state_dict[key.replace('model.','')] = value
                model.center.reset_head(226) # AUTSL
                model.left.reset_head(226) # AUTSL
                model.right.reset_head(226) # AUTSL
                # load autsl ckpt
                model.center.load_state_dict(new_state_dict)
                model.right.load_state_dict(new_state_dict)
                model.left.load_state_dict(new_state_dict)
                # add backbone
                model.add_backbone()
                # remove center, left and right backbone
                model.remove_head_and_backbone()
                model.freeze(layers = 0)
                print("Load VTN3GCN")
            elif "IMAGENET" == cfg['training']['pretrained_model']:
                model.add_backbone()
                model.remove_head_and_backbone()
                print("Load VTN3GCN IMAGENET")
            else:
                model.add_backbone()
                model.remove_head_and_backbone()
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
        
            print("Load VTN3GCN")
    else:
        if cfg['data']['model_name'] == 'vtn_att_poseflow':
            model = VTNHCPF(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
        if cfg['data']['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation':
            model = VTNHCPF_OneView_Sim_Knowledge_Distilation(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
        if cfg['data']['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference':
            model = VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            state_dict = torch.load("checkpoints/VTNHCPF_OneView_Sim_Knowledge_Distilation/VTNHCPF_OneView_Sim_Knowledge_Distilation with testing labels/best_checkpoints.pth",map_location='cpu')
            new_state_dict = {}
            for key,value in state_dict.items():
                if key.startswith('teacher'): # omit teacher state dict
                    if key.split('.')[1].startswith('classifier'): # save the classifier of the teacher model
                        new_state_dict[key.replace('teacher.','')] = value
                    continue
                new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
        elif cfg['data']['model_name'] == 'VTNHCPF_Three_view':
            model = VTNHCPF_Three_View(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            state_dict = torch.load("checkpoints/vtn_att_poseflow/vtn_att_poseflow autsl to vsl for one view/best_checkpoints.pth",map_location='cpu')
            model.center.load_state_dict(state_dict,strict = True)
            model.right.load_state_dict(state_dict,strict = True)
            model.left.load_state_dict(state_dict,strict = True)
            model.add_backbone()
            model.remove_head_and_backbone()
            model.freeze(layers = 0)
            print("Load VTNHCPF Three View")
        elif cfg['data']['model_name'] == 'VTN3GCN':
            if cfg['data']['center_kp']:
                model = VTN3GCN(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
                ckpt_path = "checkpoints/VTNGCN/VTNGCN finetune autsl to vsl for one view/best_checkpoints.pth"
                state_dict = torch.load(ckpt_path,map_location='cpu')
                print("Load VTN3GCN initialized weights: ",ckpt_path)
                model.center.load_state_dict(state_dict,strict = True)
                model.right.load_state_dict(state_dict,strict = True)
                model.left.load_state_dict(state_dict,strict = True)
                model.add_backbone()
                model.remove_head_and_backbone()
                model.freeze(layers = 0)
                print("Load VTN3GCN")
            else:
                model = VTN3GCN(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
                vtngcn_ckpt_path = "checkpoints/VTNGCN/VTNGCN finetune autsl to vsl for one view/best_checkpoints.pth"
                vtn_ckpt_path = "checkpoints/vtn_att_poseflow/vtn_att_poseflow autsl to vsl for one view/best_checkpoints.pth"
                vtngcn_state_dict = torch.load(vtngcn_ckpt_path,map_location='cpu')
                vtn_state_dict = torch.load(vtn_ckpt_path,map_location='cpu')
                new_vtn_state_dict = {}
                for key, value in vtn_state_dict.items():
                    if not key.startswith('self_attention_decoder'):
                        new_vtn_state_dict[key] = value
                print("Load VTN3GCN initialized weights: ",vtngcn_ckpt_path)
                print("Load VTN3GCN initialized weights: ",vtn_ckpt_path)
                model.center.load_state_dict(new_vtn_state_dict,strict = False)
                model.right.load_state_dict(vtngcn_state_dict,strict = True)
                model.left.load_state_dict(vtngcn_state_dict,strict = True)
                model.add_backbone()
                model.remove_head_and_backbone()
                model.freeze(layers = 0)
                print("Load VTN3GCN")
    assert model is not None
    print("loaded model")
    return model
        