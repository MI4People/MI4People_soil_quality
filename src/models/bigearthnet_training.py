# For the general context, see  also:
#
# * A deepsense.ai blog post [Keras vs. PyTorch - Alien vs. Predator recognition with transfer learning](https://deepsense.ai/keras-vs-pytorch-avp-transfer-learning) in which we compare and contrast Keras and PyTorch approaches.
# * Repo with code: [github.com/deepsense-ai/Keras-PyTorch-AvP-transfer-learning](https://github.com/deepsense-ai/Keras-PyTorch-AvP-transfer-learning).
# * Free event: [upcoming webinar (10 Oct 2018)](https://www.crowdcast.io/e/KerasVersusPyTorch/register), in which we walk trough the code (and you will be able to ask questions).
#
# ### 1. Import dependencies


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

import src.data.bigearthnet_datapipes as be_pipes
import src.data.general_datapipes as pipes
from src.globals import LABELS_TO_INDS
from src.infrastructure.aws_infrastructure import upload_file_to_s3, download_from_s3, spot_instance_terminating


# ### 2. Create PyTorch data generators

# TODO
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

# TODO Augmentations in dataloader?
# data_transforms = {
#     'train':
#     transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize
#     ]),
#     'validation':
#     transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#         normalize
#     ]),
# }

# image_datasets = {
#     'train':
#     datasets.ImageFolder(input_path + 'train', data_transforms['train']),
#     'validation':
#     datasets.ImageFolder(input_path + 'validation', data_transforms['validation'])
# }

# dataloaders = {
#     'train':
#     torch.utils.data.DataLoader(image_datasets['train'],
#                                 batch_size=32,
#                                 shuffle=True,
#                                 num_workers=0),  # for Kaggle
#     'validation':
#     torch.utils.data.DataLoader(image_datasets['validation'],
#                                 batch_size=32,
#                                 shuffle=False,
#                                 num_workers=0)  # for Kaggle
# }

folders = pipes.get_s3_folder_content()
folders = list(folders)
print(folders)
num_classes = len(LABELS_TO_INDS.keys())

datapipe = be_pipes.get_bigearth_pca_pipe(folders[:40])
train_pipe, test_pipe = pipes.split_pipe_to_train_test(datapipe, 0.2)
print(len(list(train_pipe)))
print("-------------------")
print(len(list(test_pipe)))
# TODO persist? Traintestval-split
dataloaders = {
    "train": torch.utils.data.DataLoader(dataset=train_pipe, batch_size=2),
    "test": torch.utils.data.DataLoader(dataset=test_pipe, batch_size=2),
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
def train_model(model, criterion, optimizer, num_epochs=1, starting_epoch=1, state_dict=None):
    if state_dict:
        model.load_state_dict(dl_model_path)
    # TODO set epoch to that of loaded dict
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
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
                    # upload_file_to_s3(bucket="mi4people-soil-project", local_path=model_path, remote_path=f"pytorch_models/ben_{epoch}.h5")
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

                # TODO running metric
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            # epoch_loss = running_loss / len(image_datasets[phase])
            # epoch_acc = running_corrects.double() / len(image_datasets[phase])

            # print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
            #                                            epoch_loss,
            #                                            epoch_acc))
    return model


# model_trained = train_model(model, criterion, optimizer, num_epochs=1)

# ### 5. Save and load the model
# model_path = os.path.join(os.getcwd(), 'models/weights.h5')
# torch.save(model_trained.state_dict(), model_path)
# upload_file_to_s3(bucket="mi4people-soil-project", local_path=model_path, remote_path="pytorch_models/weights.h5")

dl_model_path = os.path.join(os.getcwd(), 'models/dl_weights.h5')
# download_from_s3(bucket="mi4people-soil-project", local_path=dl_model_path, remote_path="pytorch_models/weights.h5")

model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Sequential(
    nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, 43)
).to(device)
model.load_state_dict(torch.load(dl_model_path))

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
# use mlflow callback
# How/where to run online
# spot-instance callbacks
# save weights to s3 in callback
# parametrization
