from utils.video_augmentation import *
from dataset.vtn_att_poseflow_model_dataset import VTN_ATT_PF_Dataset, VTN_GCN_Dataset, CrossVTN_Dataset
from dataset.vtn_hc_pf_three_view import VTNHCPF_ThreeViewsData,VTN3GCNData

def build_dataset(dataset_cfg, split,model = None,**kwargs):
    dataset = None

    if dataset_cfg['model_name'] == 'VTNGCN' or dataset_cfg['model_name'] == 'VTNGCN_Combine':
        dataset = VTN_GCN_Dataset(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)
    if dataset_cfg['model_name'] == '2s-CrossVTN':
        dataset = CrossVTN_Dataset(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)
    if dataset_cfg['model_name'] == 'VTN3GCN' or dataset_cfg['model_name'] == 'VTN3GCN_v2':
        dataset = VTN3GCNData(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    if dataset_cfg['model_name'] == "vtn_att_poseflow" or 'HandCrop' in dataset_cfg['model_name'] or dataset_cfg['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference':
        dataset = VTN_ATT_PF_Dataset(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    if dataset_cfg['model_name'] == 'VTNHCPF_Three_view' or dataset_cfg['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation':
        dataset = VTNHCPF_ThreeViewsData(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    assert dataset is not None
    return dataset



    