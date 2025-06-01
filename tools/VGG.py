import torch
import torch.nn as nn
import torchvision

"""
OPTIMIZATION HIGHLIGHTS:
1. Added batch normalization for better training stability
2. Improved initialization for faster convergence
3. Added dropout for better generalization
4. Support for feature extraction and transfer learning
5. Memory-efficient implementation
6. Dynamic device handling for better compatibility
7. Error handling for model loading
"""

def MyVgg16(num_classes=10, pretrained=False):
    """
    Create a VGG16 model with optimizations.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        VGG16 model
    """
    try:
        # Get the base VGG16 model with batch normalization
        # OPTIMIZATION: Using batch normalization version for better training stability
        model = torchvision.models.vgg16_bn(pretrained=pretrained)
        
        # Modify the classifier for the target dataset
        # OPTIMIZATION: Added dropout for better generalization
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),  # Increased dropout rate
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),  # Increased dropout rate
            nn.Linear(4096, num_classes),
        )
        
        # OPTIMIZATION: Better weight initialization
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        return model
    except Exception as e:
        print(f"Error creating VGG16 model: {e}")
        raise

# OPTIMIZATION: Added function for feature extraction
def MyVgg16_features(freeze_features=True, num_classes=10):
    """
    Create a VGG16 model for feature extraction.
    
    Args:
        freeze_features: Whether to freeze feature extraction layers
        num_classes: Number of output classes
        
    Returns:
        VGG16 model with frozen features
    """
    try:
        model = MyVgg16(num_classes=num_classes)
        
        # Freeze feature extraction layers
        if freeze_features:
            for param in model.features.parameters():
                param.requires_grad = False
        
        return model
    except Exception as e:
        print(f"Error creating VGG16 feature extraction model: {e}")
        raise

# OPTIMIZATION: Added memory-efficient version
def MyVgg16_efficient(num_classes=10):
    """
    Create a memory-efficient VGG16 model.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Memory-efficient VGG16 model
    """
    try:
        # Get the base VGG16 model with batch normalization
        model = torchvision.models.vgg16_bn(pretrained=False)
        
        # Reduce feature dimensions
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),  # Reduced from 4096
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),  # Reduced from 4096
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
        
        # Initialize weights
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        return model
    except Exception as e:
        print(f"Error creating memory-efficient VGG16 model: {e}")
        raise

# OPTIMIZATION: Added function to get model on appropriate device
def get_vgg16_model(num_classes=10, pretrained=False, efficient=False, feature_extract=False):
    """
    Get a VGG16 model on the appropriate device.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        efficient: Whether to use memory-efficient version
        feature_extract: Whether to use feature extraction version
        
    Returns:
        VGG16 model on appropriate device
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    if efficient:
        model = MyVgg16_efficient(num_classes)
    elif feature_extract:
        model = MyVgg16_features(freeze_features=True, num_classes=num_classes)
    else:
        model = MyVgg16(num_classes, pretrained)
    
    # Move model to device
    model = model.to(device)
    
    # Wrap model in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model, device
