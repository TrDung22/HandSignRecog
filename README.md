# Introduction
<p> In my thesis, I researched a model for Vietnamese Sign Language (VSL) recognition.</p>
<p>My contributions are: </p>
<ul> 
  <li>Proposing a multi-view (center, left, right) dataset for VSL</li>
  <li>Proposing a three-view model based on my dataset. In my experiments, this model outperformed the current one-view model by at least 5%.</li>
  <li>Proposing a knowledge distillation mechanism to distill knowledge from the three-view model into a one-view model, making it practical to use with center-view input. The resulting one-view model with knowledge distillation still performs as well as the three-view model, outperforming the current one-view model by at least 4%.</li>
</ul>

# Results:
<p>The result of three-view model: </p>

![three-view](/images/three-view.png)
<p>Results of the one-view model with the proposed knowledge distillation mechanism: </p>

![KD](/images/KD.png)

# Dataset:
<p>Information about my dataset: </p>

![Data Information](/images/data.png)

<p>Link: Comming soon ...</p>

# Checkpoints:
<p>Comming soon ...</p>

# Training
  + Knowledge Distillation Mechanism (proposal): `python3 main.py --config configs/Knowledge_Distillation/vtn_hc_pf/vtn_hc_pf_sim_KD_oneview.yaml`
  + Three view (Initialized from one view): `python3 main.py --config configs/vtn_att_poseflow/vtn_att_poseflow_three_view_finetune_from_one_view.yaml`
# Testing
  + Knowledge Distillation Mechanism (proposal): `python3 main.py --config configs/Knowledge_Distillation/vtn_hc_pf/test_cfg/vtn_hc_pf_sim_KD_oneview.yaml`
  + Three view (Initialized from one view) (proposal): `python3 main.py --config configs/vtn_att_poseflow/test_cfg/vtn_att_poseflow_three_view_finetune_from_one_view.yaml`
