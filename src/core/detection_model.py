import torch
import torch.nn as nn
from .model import CryoET3DCNN

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        
        # Classification head
        self.cls_head = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        # Box regression head (center_x, center_y, center_z, size)
        self.box_head = nn.Conv3d(in_channels, 4, kernel_size=1)
        # Objectness score
        self.obj_head = nn.Conv3d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        cls_scores = self.cls_head(x)
        box_pred = self.box_head(x)
        obj_scores = torch.sigmoid(self.obj_head(x))
        return cls_scores, box_pred, obj_scores

class CryoETDetector(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super().__init__()
        
        # Encoder with skip connections
        self.enc1 = Conv3DBlock(in_channels, 32)
        self.enc2 = Conv3DBlock(32, 64)
        self.enc3 = Conv3DBlock(64, 128)
        self.enc4 = Conv3DBlock(128, 256)
        
        # Feature Pyramid Network (FPN)
        self.fpn_conv1 = Conv3DBlock(256, 256)
        self.fpn_conv2 = Conv3DBlock(128, 256)
        self.fpn_conv3 = Conv3DBlock(64, 256)
        
        # Upsample layers
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        # Detection heads for different scales
        self.detect1 = DetectionHead(256, num_classes)
        self.detect2 = DetectionHead(256, num_classes)
        self.detect3 = DetectionHead(256, num_classes)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool3d(x1, 2))
        x3 = self.enc3(F.max_pool3d(x2, 2))
        x4 = self.enc4(F.max_pool3d(x3, 2))
        
        # FPN
        p4 = self.fpn_conv1(x4)
        p3 = self.fpn_conv2(x3 + self.upsample(p4))
        p2 = self.fpn_conv3(x2 + self.upsample(p3))
        
        # Detection heads
        d4 = self.detect1(p4)
        d3 = self.detect2(p3)
        d2 = self.detect3(p2)
        
        return [(d2, 4), (d3, 8), (d4, 16)]  # (predictions, stride)

class DetectionLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_box=1.0, lambda_obj=1.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        
    def forward(self, predictions, targets):
        total_loss = 0
        cls_loss = 0
        box_loss = 0
        obj_loss = 0
        
        for (cls_pred, box_pred, obj_pred), stride in predictions:
            # Match predictions with targets based on IoU
            matched_targets = self.match_targets(box_pred, targets, stride)
            
            # Classification loss (focal loss)
            cls_loss += self.focal_loss(cls_pred, matched_targets['cls'])
            
            # Box regression loss (GIoU loss)
            box_loss += self.giou_loss(box_pred, matched_targets['box'])
            
            # Objectness loss (BCE loss)
            obj_loss += F.binary_cross_entropy(obj_pred, matched_targets['obj'])
        
        total_loss = (self.lambda_cls * cls_loss + 
                     self.lambda_box * box_loss + 
                     self.lambda_obj * obj_loss)
        
        return total_loss, {
            'cls_loss': cls_loss,
            'box_loss': box_loss,
            'obj_loss': obj_loss
        }
    
    def focal_loss(self, pred, target, gamma=2.0, alpha=0.25):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = alpha * (1 - p_t) ** gamma * ce_loss
        return loss.mean()
    
    def giou_loss(self, pred_boxes, target_boxes):
        # Implement GIoU loss for 3D boxes
        # This is a placeholder - actual implementation needed
        return F.mse_loss(pred_boxes, target_boxes)
    
    def match_targets(self, pred_boxes, targets, stride):
        # Implement target matching based on IoU
        # This is a placeholder - actual implementation needed
        return {
            'cls': torch.zeros_like(pred_boxes[:, 0]),
            'box': torch.zeros_like(pred_boxes),
            'obj': torch.zeros_like(pred_boxes[:, 0])
        }

if __name__ == "__main__":
    # Test the model
    model = CryoETDetector()
    
    # Create a dummy batch
    batch_size = 2
    channels = 1
    size = 64
    x = torch.randn(batch_size, channels, size, size, size)
    
    # Forward pass
    outputs = model(x)
    
    print("Model output shapes:")
    for (cls_pred, box_pred, obj_pred), stride in outputs:
        print(f"Stride {stride}:")
        print(f"  Classification: {cls_pred.shape}")
        print(f"  Box regression: {box_pred.shape}")
        print(f"  Objectness: {obj_pred.shape}")
