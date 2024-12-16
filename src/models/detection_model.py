import torch
import torch.nn as nn

class CryoETDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.classifier = nn.Linear(64 * 8 * 8 * 8, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.box_loss = nn.SmoothL1Loss()
        
    def forward(self, predictions, targets):
        cls_loss = self.cls_loss(predictions['class_scores'], targets['labels'])
        box_loss = self.box_loss(predictions['boxes'], targets['boxes'])
        obj_loss = torch.zeros(1, device=predictions['class_scores'].device)
        
        total_loss = cls_loss + box_loss + obj_loss
        
        return total_loss, {
            'cls_loss': cls_loss.item(),
            'box_loss': box_loss.item(),
            'obj_loss': obj_loss.item()
        } 