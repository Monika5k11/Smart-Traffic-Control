import torch
import torchvision.models as models


from PIL import Image,ImageFilter
import torch.nn as nn
import os

import cv2
import numpy as np

import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])


class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, targets = batch
        out = self(images)
        # _,out = torch.max(out,dim = 1)
        loss = F.binary_cross_entropy(torch.sigmoid(out), targets)
        return loss

    def validation_step(self, batch):
        images, targets = batch
        out = self(images)

        # Generate predictions
        loss = F.binary_cross_entropy(torch.sigmoid(out), targets)

        score = accuracy(out, targets)
        return {'val_loss': loss.detach(), 'val_score': score.detach()}

    # this 2 methods will not change .

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_score']))


class Densenet169(ImageClassificationBase):

    def __init__(self):
        super().__init__()
        self.pretrained_model = models.densenet169(pretrained=True)

        feature_in = self.pretrained_model.classifier.in_features
        self.pretrained_model.classifier = nn.Linear(feature_in, 2)

    def forward(self, x):
        return self.pretrained_model(x)

loaded_densenet169 = Densenet169()
loaded_densenet169.load_state_dict(torch.load('densenet169.pt', map_location=torch.device('cpu')))
loaded_densenet169.eval()
print('loaded')


preds = []

test_dataset = os.listdir("./test_images")

for image_file in test_dataset:
    image_path = "./test_images/" + image_file
    print(image_path)
    image = Image.open(image_path)
    image = image.resize((224,224))
    test_image= transform(image)
    test_image = test_image.view(1,3,224,224)
    print(test_image)
    pred  = loaded_densenet169.forward(test_image)
    _,idx = torch.max(pred,dim = 1)
    idx = idx.numpy()[0]
    preds.append(idx)
    print(idx)


print(preds)