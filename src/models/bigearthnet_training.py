import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


import torchvision
import mlflow

import src.data.bigearthnet_datapipes as be_pipes
import src.data.general_datapipes as pipes
from src.globals import LABELS_TO_INDS
from src.infrastructure.aws_infrastructure import upload_file_to_s3, download_from_s3, spot_instance_terminating
import src.infrastructure.mlflow_logging as ml_logging
from src.models.training_utils import get_latest_weights


# ### 2. Create PyTorch data generators

# TODO: Set new test name
# mlflow_experiment = ml_logging.start_auto_logging("test5", "pytorch")

folders = pipes.get_s3_folder_content()

# TODO: Rewrite code so that it will run in AWS and remove limitation when running pipeline for training in AWS
datapipe = be_pipes.get_bigearth_pca_pipe(folders[:40])
train_pipe, test_pipe = pipes.split_pipe_to_train_test(datapipe, 0.2)
print(len(list(train_pipe)))
print("-------------------")
print(len(list(test_pipe)))
# TODO persist Traintestval-split
dataloaders = {
    "train": torch.utils.data.DataLoader(dataset=train_pipe, batch_size=2),
    "test": torch.utils.data.DataLoader(dataset=test_pipe, batch_size=2),
    # val
}

# ### 3. Create the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, 43)
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.fc.parameters())


# ### 4. Train the model
def train_model(model, criterion, optimizer, total_epochs=1, starting_epoch=0, model_path=None, mlflow_experiment=None):
    if model_path:
        try:
            model.load_state_dict(model_path)
        # TODO load latest model:
        except:
            pass
            # TODO change params to get model name
            #model = ml_logging.get_latest_model()
        print(f"Resuming training from earlier model.")
    for epoch in range(starting_epoch, total_epochs):
        print("Epoch {}/{}".format(epoch, total_epochs))
        print("-" * 10)

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for sample in dataloaders[phase]:
                # if spot_instance_terminating():
                    # logging.warning("Spot instances will be terminated soon. Saving weights to s3.")
                    # model_path = os.path.join(os.getcwd(), 'models/weights.h5')
                    # torch.save(model_trained.state_dict(), model_path)
                    # mlflow.pytorch.log_model(model, f"ben_res50_epoch_{epoch}", registered_model_name="ben_res50")
                    # TODO further cleanup: shut down?

                inputs = sample["data"]
                labels = sample["label"]
                inputs = inputs.to(device, dtype=torch.float)
                # labels = labels.type(torch.LongTensor)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # TODO running metric (eg accuracy)
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                # log loss to db
                mlflow.log_metric("running_loss", running_loss)
        # save model to s3 and register it (importtant to retrieve it later)
        # TODO IMPORTANT: change model name to something which is different for each experiment, but also memorable/ automatically callable by the process managing spot instance orchestration
        mlflow.pytorch.log_model(model, f"ben_res50_epoch_{epoch}", registered_model_name="ben_res50")
        print("Finished Epoch, logged model")
            # running_corrects += torch.sum(preds == labels.data)

            # epoch_loss = running_loss / len(image_datasets[phase])
            # epoch_acc = running_corrects.double() / len(image_datasets[phase])

            # print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
            #                                            epoch_loss,
            #                                            epoch_acc))
    # mlflow.pytorch.log_model(model)
    return model

# TODO: Potentially remove mlflow_experiment
model_trained = train_model(model, criterion, optimizer, total_epochs=3, mlflow_experiment=mlflow_experiment)
exit()

# ### 5. Save model locally
# model_path = os.path.join(os.getcwd(), 'models/weights.h5')
# torch.save(model_trained.state_dict(), model_path)

# ### 6.1 Either load a local model:
model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Sequential(
    nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, 43)
).to(device)
model.load_state_dict(torch.load(model_path))
# ### 6.2 Or get it from the bucket with mlflow:
model = ml_logging.get_latest_model("ben_res50", "latest")



# # ### 6. Make predictions on sample test images
for sample in dataloaders["test"]:
    inputs = sample["data"]
    inputs = inputs.to(device, dtype=torch.float)
    labels = sample["label"]
    pred_logits_tensor = model(inputs.to(device))
    print(pred_logits_tensor)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    print(pred_probs)


# TODOs after local todos:
# How/where to run online
# Test / ask abdellaziz if spot-instance callbacks are right like this
# save weights to s3 in callback & load latest weights
# parametrization
# modularize these steps
# save logs?
