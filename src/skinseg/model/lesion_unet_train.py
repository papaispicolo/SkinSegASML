#!/usr/bin/env python
# coding: utf-8

hyper_params = {
    "input_size": 572,
    "num_layers": 2,
    "num_classes": 1,
    "batch_size": 8,
    "num_epochs": 150,
    "learning_rate": 0.00001
}

"""
hyper_params = {
    "input_size": 572,
    "num_layers": 2,
    "num_classes": 1,
    "batch_size": 1,
    "num_epochs": 1,
    "learning_rate": 0.00001
}
"""

experiment = Experiment(
    api_key="uespiI7sf0P5L5g2ja4vTFz25",
    project_name="unet-isic",
    workspace="omarkhaled99",
)

torch.cuda.empty_cache()
# experiment = Experiment(project_name="unet-isic-pytorch")
experiment.log_parameters(hyper_params)

lr = hyper_params["learning_rate"]
epochs = hyper_params["num_epochs"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet = Unet()
unet.to(device)
iou_loss = JaccardLoss(BINARY_MODE)
best_validation_iou = 0.0
optimizer = Adam(unet.parameters(), lr=lr)

loss_train = []
loss_valid = []

loaders = {"train": train_dataloader, "valid": valid_dataloader}
validation_iou = []
train_iou =  []
step = 0
#!nvidia-smi


