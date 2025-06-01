import torch
import torch.nn as nn
import numpy as np
import os
import copy
from tools.flops_params import get_flops_params

"""
OPTIMIZATION HIGHLIGHTS:
1. Enhanced pruning criteria (L1, L2, L1-L2 hybrid)
2. Global pruning strategy for better filter selection
3. Iterative pruning implementation
4. Layer sensitivity analysis
5. Quantization support for further model compression
6. Improved documentation and code organization
"""

def channels_index(weight_torch, criterion="l1"):
    """
    Calculate the importance of each filter based on specified criterion.
    
    Args:
        weight_torch: Weight tensor of convolutional layer
        criterion: Pruning criterion ('l1', 'l2', 'l1_l2')
        
    Returns:
        Sorted indices of filters by importance (least to most important)
    """
    # OPTIMIZATION: Support for multiple pruning criteria
    weight_torch = weight_torch.cpu().detach().numpy()
    weight_vec = weight_torch.reshape(weight_torch.shape[0], -1)
    
    if criterion == "l1":
        # Original L1-norm criterion
        norm = np.sum(np.abs(weight_vec), axis=1)
    elif criterion == "l2":
        # OPTIMIZATION: Added L2-norm criterion
        norm = np.sqrt(np.sum(np.power(weight_vec, 2), axis=1))
    elif criterion == "l1_l2":
        # OPTIMIZATION: Added hybrid L1-L2 criterion
        l1_norm = np.sum(np.abs(weight_vec), axis=1)
        l2_norm = np.sqrt(np.sum(np.power(weight_vec, 2), axis=1))
        # Normalize both norms to [0,1]
        l1_norm = l1_norm / np.max(l1_norm)
        l2_norm = l2_norm / np.max(l2_norm)
        # Combine with equal weights
        norm = 0.5 * l1_norm + 0.5 * l2_norm
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
        
    norm_argsort = np.argsort(norm)
    return norm_argsort

def select_channels(weight_torch, num_channels, criterion="l1"):
    """
    Select channels to prune based on importance criterion.
    
    Args:
        weight_torch: Weight tensor of convolutional layer
        num_channels: Number of channels to prune
        criterion: Pruning criterion ('l1', 'l2', 'l1_l2')
        
    Returns:
        Indices of channels to keep
    """
    norm_argsort = channels_index(weight_torch, criterion)
    # Select channels to keep (exclude the smallest num_channels)
    selected_channels = norm_argsort[num_channels:]
    return selected_channels

def prune_conv(conv, selected_channels, device):
    """
    Prune a convolutional layer by keeping only selected channels.
    
    Args:
        conv: Convolutional layer to prune
        selected_channels: Indices of channels to keep
        device: Device to use (CPU or GPU)
        
    Returns:
        Pruned convolutional layer
    """
    # Get weights and bias
    weight = conv.weight.data.cpu().numpy()
    bias = None
    if conv.bias is not None:
        bias = conv.bias.data.cpu().numpy()
        
    # Create new convolutional layer with selected channels
    new_conv = nn.Conv2d(in_channels=conv.in_channels,
                        out_channels=len(selected_channels),
                        kernel_size=conv.kernel_size,
                        stride=conv.stride,
                        padding=conv.padding,
                        dilation=conv.dilation,
                        groups=conv.groups,
                        bias=conv.bias is not None)
                        
    # Set weights and bias for selected channels
    new_weight = weight[selected_channels, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weight).to(device)
    
    if bias is not None:
        new_bias = bias[selected_channels]
        new_conv.bias.data = torch.from_numpy(new_bias).to(device)
        
    return new_conv

def prune_related_conv(conv, last_selected_channels, device):
    """
    Prune a convolutional layer related to a previously pruned layer.
    
    Args:
        conv: Convolutional layer to prune
        last_selected_channels: Indices of channels to keep from previous layer
        device: Device to use (CPU or GPU)
        
    Returns:
        Pruned convolutional layer
    """
    # Get weights and bias
    weight = conv.weight.data.cpu().numpy()
    bias = None
    if conv.bias is not None:
        bias = conv.bias.data.cpu().numpy()
        
    # Create new convolutional layer with input channels matching last_selected_channels
    new_conv = nn.Conv2d(in_channels=len(last_selected_channels),
                        out_channels=conv.out_channels,
                        kernel_size=conv.kernel_size,
                        stride=conv.stride,
                        padding=conv.padding,
                        dilation=conv.dilation,
                        groups=conv.groups,
                        bias=conv.bias is not None)
                        
    # Set weights and bias
    new_weight = weight[:, last_selected_channels, :, :]
    new_conv.weight.data = torch.from_numpy(new_weight).to(device)
    
    if bias is not None:
        new_conv.bias.data = torch.from_numpy(bias).to(device)
        
    return new_conv

def prune_batchnorm(bn, selected_channels, device):
    """
    Prune a batch normalization layer by keeping only selected channels.
    
    Args:
        bn: Batch normalization layer to prune
        selected_channels: Indices of channels to keep
        device: Device to use (CPU or GPU)
        
    Returns:
        Pruned batch normalization layer
    """
    # Create new batch normalization layer with selected channels
    new_bn = nn.BatchNorm2d(num_features=len(selected_channels))
    
    # Get weights and bias
    weight = bn.weight.data.cpu().numpy()
    bias = bn.bias.data.cpu().numpy()
    running_mean = bn.running_mean.data.cpu().numpy()
    running_var = bn.running_var.data.cpu().numpy()
    
    # Set weights and bias for selected channels
    new_bn.weight.data = torch.from_numpy(weight[selected_channels]).to(device)
    new_bn.bias.data = torch.from_numpy(bias[selected_channels]).to(device)
    new_bn.running_mean.data = torch.from_numpy(running_mean[selected_channels]).to(device)
    new_bn.running_var.data = torch.from_numpy(running_var[selected_channels]).to(device)
    
    return new_bn

def prune_vgg16_conv_layer(model, layer_index, num_channels, device, criterion="l1"):
    """
    Prune a convolutional layer in VGG16 model.
    
    Args:
        model: VGG16 model
        layer_index: Index of the layer to prune
        num_channels: Number of channels to prune
        device: Device to use (CPU or GPU)
        criterion: Pruning criterion ('l1', 'l2', 'l1_l2')
        
    Returns:
        Pruned model
    """
    # Get the convolutional layer to prune
    conv_to_prune = model.module.features[layer_index]
    
    # Select channels to keep
    selected_channels = select_channels(conv_to_prune.weight, num_channels, criterion)
    
    # Prune the convolutional layer
    new_conv = prune_conv(conv_to_prune, selected_channels, device)
    
    # Prune the batch normalization layer
    bn_to_prune = model.module.features[layer_index + 1]
    new_bn = prune_batchnorm(bn_to_prune, selected_channels, device)
    
    # Replace the pruned layers in the model
    model.module.features[layer_index] = new_conv
    model.module.features[layer_index + 1] = new_bn
    
    # If not the last convolutional layer, prune the next convolutional layer's input channels
    if layer_index + 3 < len(model.module.features):
        next_conv = model.module.features[layer_index + 3]
        if isinstance(next_conv, nn.Conv2d):
            new_next_conv = prune_related_conv(next_conv, selected_channels, device)
            model.module.features[layer_index + 3] = new_next_conv
    
    # If it's the last convolutional layer, prune the classifier's input features
    else:
        if hasattr(model.module, 'classifier') and isinstance(model.module.classifier[0], nn.Linear):
            in_features = model.module.classifier[0].in_features
            out_features = model.module.classifier[0].out_features
            
            # Calculate new in_features based on selected channels
            new_in_features = len(selected_channels) * (in_features // conv_to_prune.out_channels)
            
            # Create new linear layer
            new_linear = nn.Linear(new_in_features, out_features)
            
            # Copy weights for selected channels
            old_weights = model.module.classifier[0].weight.data
            new_weights = torch.zeros((out_features, new_in_features))
            
            for i, idx in enumerate(selected_channels):
                new_weights[:, i * (new_in_features // len(selected_channels)):(i + 1) * (new_in_features // len(selected_channels))] = \
                    old_weights[:, idx * (in_features // conv_to_prune.out_channels):(idx + 1) * (in_features // conv_to_prune.out_channels)]
            
            new_linear.weight.data = new_weights.to(device)
            new_linear.bias.data = model.module.classifier[0].bias.data.clone()
            
            # Replace the linear layer
            model.module.classifier[0] = new_linear
    
    return model

def prune_resnet_conv_layer(model, layer_name, num_channels, device, shortcutflag=False, criterion="l1"):
    """
    Prune a convolutional layer in ResNet model.
    
    Args:
        model: ResNet model
        layer_name: Name of the layer to prune
        num_channels: Number of channels to prune
        device: Device to use (CPU or GPU)
        shortcutflag: Whether to prune shortcut connections
        criterion: Pruning criterion ('l1', 'l2', 'l1_l2')
        
    Returns:
        Pruned model
    """
    # Parse layer name
    if shortcutflag:
        # Prune shortcut connection
        if layer_name == "downsample_1":
            conv_to_prune = model.module.layer1[0].downsample[0]
            bn_to_prune = model.module.layer1[0].downsample[1]
            
            # Select channels to keep
            selected_channels = select_channels(conv_to_prune.weight, num_channels, criterion)
            
            # Prune the convolutional layer
            new_conv = prune_conv(conv_to_prune, selected_channels, device)
            
            # Prune the batch normalization layer
            new_bn = prune_batchnorm(bn_to_prune, selected_channels, device)
            
            # Replace the pruned layers in the model
            model.module.layer1[0].downsample[0] = new_conv
            model.module.layer1[0].downsample[1] = new_bn
            
        elif layer_name == "downsample_2":
            conv_to_prune = model.module.layer2[0].downsample[0]
            bn_to_prune = model.module.layer2[0].downsample[1]
            
            # Select channels to keep
            selected_channels = select_channels(conv_to_prune.weight, num_channels, criterion)
            
            # Prune the convolutional layer
            new_conv = prune_conv(conv_to_prune, selected_channels, device)
            
            # Prune the batch normalization layer
            new_bn = prune_batchnorm(bn_to_prune, selected_channels, device)
            
            # Replace the pruned layers in the model
            model.module.layer2[0].downsample[0] = new_conv
            model.module.layer2[0].downsample[1] = new_bn
            
        elif layer_name == "downsample_3":
            conv_to_prune = model.module.layer3[0].downsample[0]
            bn_to_prune = model.module.layer3[0].downsample[1]
            
            # Select channels to keep
            selected_channels = select_channels(conv_to_prune.weight, num_channels, criterion)
            
            # Prune the convolutional layer
            new_conv = prune_conv(conv_to_prune, selected_channels, device)
            
            # Prune the batch normalization layer
            new_bn = prune_batchnorm(bn_to_prune, selected_channels, device)
            
            # Replace the pruned layers in the model
            model.module.layer3[0].downsample[0] = new_conv
            model.module.layer3[0].downsample[1] = new_bn
            
    else:
        # Prune regular convolutional layer
        layer_index = int(layer_name.split('_')[1])
        
        # Determine the layer and block indices
        if layer_index <= 6:
            block_index = (layer_index - 1) // 2
            conv_index = (layer_index - 1) % 2
            conv_to_prune = model.module.layer1[block_index].conv1 if conv_index == 0 else model.module.layer1[block_index].conv2
            bn_to_prune = model.module.layer1[block_index].bn1 if conv_index == 0 else model.module.layer1[block_index].bn2
            
        elif layer_index <= 14:
            block_index = (layer_index - 7) // 2
            conv_index = (layer_index - 7) % 2
            conv_to_prune = model.module.layer2[block_index].conv1 if conv_index == 0 else model.module.layer2[block_index].conv2
            bn_to_prune = model.module.layer2[block_index].bn1 if conv_index == 0 else model.module.layer2[block_index].bn2
            
        elif layer_index <= 26:
            block_index = (layer_index - 15) // 2
            conv_index = (layer_index - 15) % 2
            conv_to_prune = model.module.layer3[block_index].conv1 if conv_index == 0 else model.module.layer3[block_index].conv2
            bn_to_prune = model.module.layer3[block_index].bn1 if conv_index == 0 else model.module.layer3[block_index].bn2
            
        else:
            block_index = (layer_index - 27) // 2
            conv_index = (layer_index - 27) % 2
            conv_to_prune = model.module.layer4[block_index].conv1 if conv_index == 0 else model.module.layer4[block_index].conv2
            bn_to_prune = model.module.layer4[block_index].bn1 if conv_index == 0 else model.module.layer4[block_index].bn2
        
        # Select channels to keep
        selected_channels = select_channels(conv_to_prune.weight, num_channels, criterion)
        
        # Prune the convolutional layer
        new_conv = prune_conv(conv_to_prune, selected_channels, device)
        
        # Prune the batch normalization layer
        new_bn = prune_batchnorm(bn_to_prune, selected_channels, device)
        
        # Replace the pruned layers in the model
        if layer_index <= 6:
            if conv_index == 0:
                model.module.layer1[block_index].conv1 = new_conv
                model.module.layer1[block_index].bn1 = new_bn
                # Prune the next layer's input channels
                next_conv = model.module.layer1[block_index].conv2
                new_next_conv = prune_related_conv(next_conv, selected_channels, device)
                model.module.layer1[block_index].conv2 = new_next_conv
            else:
                model.module.layer1[block_index].conv2 = new_conv
                model.module.layer1[block_index].bn2 = new_bn
                
        elif layer_index <= 14:
            if conv_index == 0:
                model.module.layer2[block_index].conv1 = new_conv
                model.module.layer2[block_index].bn1 = new_bn
                # Prune the next layer's input channels
                next_conv = model.module.layer2[block_index].conv2
                new_next_conv = prune_related_conv(next_conv, selected_channels, device)
                model.module.layer2[block_index].conv2 = new_next_conv
            else:
                model.module.layer2[block_index].conv2 = new_conv
                model.module.layer2[block_index].bn2 = new_bn
                
        elif layer_index <= 26:
            if conv_index == 0:
                model.module.layer3[block_index].conv1 = new_conv
                model.module.layer3[block_index].bn1 = new_bn
                # Prune the next layer's input channels
                next_conv = model.module.layer3[block_index].conv2
                new_next_conv = prune_related_conv(next_conv, selected_channels, device)
                model.module.layer3[block_index].conv2 = new_next_conv
            else:
                model.module.layer3[block_index].conv2 = new_conv
                model.module.layer3[block_index].bn2 = new_bn
                
        else:
            if conv_index == 0:
                model.module.layer4[block_index].conv1 = new_conv
                model.module.layer4[block_index].bn1 = new_bn
                # Prune the next layer's input channels
                next_conv = model.module.layer4[block_index].conv2
                new_next_conv = prune_related_conv(next_conv, selected_channels, device)
                model.module.layer4[block_index].conv2 = new_next_conv
            else:
                model.module.layer4[block_index].conv2 = new_conv
                model.module.layer4[block_index].bn2 = new_bn
    
    return model

def prune_net(model, independentflag, prune_layers, prune_channels, net_name, shortcutflag=False, criterion="l1"):
    """
    Prune a neural network model.
    
    Args:
        model: Neural network model
        independentflag: Whether to prune layers independently
        prune_layers: List of layers to prune
        prune_channels: List of number of channels to prune for each layer
        net_name: Name of the network architecture
        shortcutflag: Whether to prune shortcut connections
        criterion: Pruning criterion ('l1', 'l2', 'l1_l2')
        
    Returns:
        Pruned model
    """
    device = next(model.parameters()).device
    
    # Create a copy of the model to avoid modifying the original
    model_copy = copy.deepcopy(model)
    
    # Prune each layer
    for i, (layer, num_channels) in enumerate(zip(prune_layers, prune_channels)):
        if net_name == 'vgg16':
            # Convert layer name to index
            layer_index = int(layer.split('_')[1]) * 2 - 2
            model_copy = prune_vgg16_conv_layer(model_copy, layer_index, num_channels, device, criterion)
        elif net_name == 'resnet34':
            model_copy = prune_resnet_conv_layer(model_copy, layer, num_channels, device, shortcutflag, criterion)
    
    return model_copy

# OPTIMIZATION: Added global pruning strategy
def global_pruning(model, prune_ratio, net_name, shortcutflag=False, criterion="l1"):
    """
    Prune a neural network model using global pruning strategy.
    
    Args:
        model: Neural network model
        prune_ratio: Ratio of filters to prune globally
        net_name: Name of the network architecture
        shortcutflag: Whether to prune shortcut connections
        criterion: Pruning criterion ('l1', 'l2', 'l1_l2')
        
    Returns:
        Pruned model
    """
    device = next(model.parameters()).device
    
    # Create a copy of the model to avoid modifying the original
    model_copy = copy.deepcopy(model)
    
    # Collect all convolutional layers and their importances
    conv_layers = []
    importances = []
    
    if net_name == 'vgg16':
        # Collect VGG16 convolutional layers
        for i, module in enumerate(model_copy.module.features):
            if isinstance(module, nn.Conv2d):
                # Skip first layer
                if i > 0:
                    conv_layers.append((i, module))
                    
                    # Calculate importance of each filter
                    weight = module.weight.data
                    if criterion == "l1":
                        importance = torch.sum(torch.abs(weight.view(weight.size(0), -1)), dim=1)
                    elif criterion == "l2":
                        importance = torch.sqrt(torch.sum(torch.pow(weight.view(weight.size(0), -1), 2), dim=1))
                    else:  # l1_l2
                        l1 = torch.sum(torch.abs(weight.view(weight.size(0), -1)), dim=1)
                        l2 = torch.sqrt(torch.sum(torch.pow(weight.view(weight.size(0), -1), 2), dim=1))
                        l1_norm = l1 / torch.max(l1)
                        l2_norm = l2 / torch.max(l2)
                        importance = 0.5 * l1_norm + 0.5 * l2_norm
                        
                    importances.append(importance)
    
    elif net_name == 'resnet34':
        # Collect ResNet34 convolutional layers
        for block_idx, block in enumerate(model_copy.module.layer1):
            if not shortcutflag:
                # Regular convolutional layers
                conv_layers.append((f"conv_{block_idx*2+1}", block.conv1))
                conv_layers.append((f"conv_{block_idx*2+2}", block.conv2))
                
                # Calculate importance of each filter
                for conv in [block.conv1, block.conv2]:
                    weight = conv.weight.data
                    if criterion == "l1":
                        importance = torch.sum(torch.abs(weight.view(weight.size(0), -1)), dim=1)
                    elif criterion == "l2":
                        importance = torch.sqrt(torch.sum(torch.pow(weight.view(weight.size(0), -1), 2), dim=1))
                    else:  # l1_l2
                        l1 = torch.sum(torch.abs(weight.view(weight.size(0), -1)), dim=1)
                        l2 = torch.sqrt(torch.sum(torch.pow(weight.view(weight.size(0), -1), 2), dim=1))
                        l1_norm = l1 / torch.max(l1)
                        l2_norm = l2 / torch.max(l2)
                        importance = 0.5 * l1_norm + 0.5 * l2_norm
                        
                    importances.append(importance)
            else:
                # Shortcut connections
                if hasattr(block, 'downsample') and block.downsample is not None:
                    conv_layers.append((f"downsample_{1}", block.downsample[0]))
                    
                    # Calculate importance of each filter
                    weight = block.downsample[0].weight.data
                    if criterion == "l1":
                        importance = torch.sum(torch.abs(weight.view(weight.size(0), -1)), dim=1)
                    elif criterion == "l2":
                        importance = torch.sqrt(torch.sum(torch.pow(weight.view(weight.size(0), -1), 2), dim=1))
                    else:  # l1_l2
                        l1 = torch.sum(torch.abs(weight.view(weight.size(0), -1)), dim=1)
                        l2 = torch.sqrt(torch.sum(torch.pow(weight.view(weight.size(0), -1), 2), dim=1))
                        l1_norm = l1 / torch.max(l1)
                        l2_norm = l2 / torch.max(l2)
                        importance = 0.5 * l1_norm + 0.5 * l2_norm
                        
                    importances.append(importance)
        
        # Repeat for other layers
        for layer_idx, layer in enumerate([model_copy.module.layer2, model_copy.module.layer3, model_copy.module.layer4]):
            for block_idx, block in enumerate(layer):
                if not shortcutflag:
                    # Regular convolutional layers
                    layer_offset = 7 if layer_idx == 0 else (15 if layer_idx == 1 else 27)
                    conv_layers.append((f"conv_{layer_offset+block_idx*2+1}", block.conv1))
                    conv_layers.append((f"conv_{layer_offset+block_idx*2+2}", block.conv2))
                    
                    # Calculate importance of each filter
                    for conv in [block.conv1, block.conv2]:
                        weight = conv.weight.data
                        if criterion == "l1":
                            importance = torch.sum(torch.abs(weight.view(weight.size(0), -1)), dim=1)
                        elif criterion == "l2":
                            importance = torch.sqrt(torch.sum(torch.pow(weight.view(weight.size(0), -1), 2), dim=1))
                        else:  # l1_l2
                            l1 = torch.sum(torch.abs(weight.view(weight.size(0), -1)), dim=1)
                            l2 = torch.sqrt(torch.sum(torch.pow(weight.view(weight.size(0), -1), 2), dim=1))
                            l1_norm = l1 / torch.max(l1)
                            l2_norm = l2 / torch.max(l2)
                            importance = 0.5 * l1_norm + 0.5 * l2_norm
                            
                        importances.append(importance)
                else:
                    # Shortcut connections
                    if hasattr(block, 'downsample') and block.downsample is not None:
                        conv_layers.append((f"downsample_{layer_idx+2}", block.downsample[0]))
                        
                        # Calculate importance of each filter
                        weight = block.downsample[0].weight.data
                        if criterion == "l1":
                            importance = torch.sum(torch.abs(weight.view(weight.size(0), -1)), dim=1)
                        elif criterion == "l2":
                            importance = torch.sqrt(torch.sum(torch.pow(weight.view(weight.size(0), -1), 2), dim=1))
                        else:  # l1_l2
                            l1 = torch.sum(torch.abs(weight.view(weight.size(0), -1)), dim=1)
                            l2 = torch.sqrt(torch.sum(torch.pow(weight.view(weight.size(0), -1), 2), dim=1))
                            l1_norm = l1 / torch.max(l1)
                            l2_norm = l2 / torch.max(l2)
                            importance = 0.5 * l1_norm + 0.5 * l2_norm
                            
                        importances.append(importance)
    
    # Flatten and normalize importances
    all_importances = torch.cat(importances)
    all_importances = all_importances / torch.max(all_importances)
    
    # Determine global threshold
    sorted_importances, _ = torch.sort(all_importances)
    threshold_idx = int(len(sorted_importances) * prune_ratio)
    threshold = sorted_importances[threshold_idx]
    
    # Determine number of filters to prune for each layer
    prune_layers = []
    prune_channels = []
    
    for i, ((layer_name, conv), importance) in enumerate(zip(conv_layers, importances)):
        # Count filters below threshold
        num_to_prune = torch.sum(importance < threshold).item()
        
        # Ensure we don't prune all filters
        max_to_prune = max(0, conv.out_channels - 1)
        num_to_prune = min(num_to_prune, max_to_prune)
        
        if num_to_prune > 0:
            prune_layers.append(layer_name)
            prune_channels.append(num_to_prune)
    
    # Prune the model
    if len(prune_layers) > 0:
        model_copy = prune_net(model_copy, False, prune_layers, prune_channels, net_name, shortcutflag, criterion)
    
    return model_copy

# OPTIMIZATION: Added iterative pruning implementation
def iterative_pruning(model, args, train_loader, test_loader):
    """
    Prune a neural network model iteratively with retraining between iterations.
    
    Args:
        model: Neural network model
        args: Command line arguments
        train_loader: Training data loader
        test_loader: Test data loader
        
    Returns:
        Pruned model
    """
    from train import training
    
    # Create a copy of the model to avoid modifying the original
    model_copy = copy.deepcopy(model)
    
    # Calculate total pruning amount per layer
    total_prune_channels = args.prune_channels
    
    # Calculate pruning amount per iteration
    iter_prune_channels = []
    for channel in total_prune_channels:
        iter_prune_channels.append([channel // args.prune_iterations] * (args.prune_iterations - 1) + 
                                  [channel - (channel // args.prune_iterations) * (args.prune_iterations - 1)])
    
    # Iteratively prune and retrain
    for i in range(args.prune_iterations):
        print(f"Pruning iteration {i+1}/{args.prune_iterations}")
        
        # Get pruning amount for this iteration
        current_prune_channels = [channels[i] for channels in iter_prune_channels]
        
        # Prune the model
        model_copy = prune_net(
            model_copy, 
            args.independentflag, 
            args.prune_layers, 
            current_prune_channels, 
            args.net, 
            args.shortcutflag, 
            args.criterion
        )
        
        # Retrain the model
        if i < args.prune_iterations - 1:  # Skip retraining for the last iteration
            checkpoint_path = os.path.join("./checkpoint", args.net, f"iter_prune_{i}")
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
                
            model_copy = training(
                model_copy, 
                args.intermediate_epochs, 
                train_loader, 
                test_loader, 
                True,  # retrain flag
                args.retrainlr, 
                args.optim, 
                checkpoint_path, 
                checkpoint_path
            )
    
    return model_copy

# OPTIMIZATION: Added layer sensitivity analysis
def analyze_layer_sensitivity(model, train_loader, test_loader, layers, prune_ratio=0.3, criterion="l1"):
    """
    Analyze the sensitivity of each layer to pruning.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        layers: List of layers to analyze
        prune_ratio: Ratio of filters to prune for sensitivity analysis
        criterion: Pruning criterion ('l1', 'l2', 'l1_l2')
        
    Returns:
        Dictionary mapping layer names to sensitivity scores
    """
    from train import eval_epoch
    
    # Get baseline accuracy
    baseline_acc, _, _, _ = eval_epoch(model, test_loader)
    
    # Analyze each layer
    sensitivity = {}
    
    for layer in layers:
        # Create a copy of the model
        model_copy = copy.deepcopy(model)
        
        # Determine number of channels to prune
        if isinstance(model_copy.module.features[0], nn.Conv2d):  # VGG16
            layer_idx = int(layer.split('_')[1]) * 2 - 2
            conv = model_copy.module.features[layer_idx]
        else:  # ResNet34
            # Parse layer name to get the convolutional layer
            if "downsample" in layer:
                layer_idx = int(layer.split('_')[1])
                if layer_idx == 1:
                    conv = model_copy.module.layer1[0].downsample[0]
                elif layer_idx == 2:
                    conv = model_copy.module.layer2[0].downsample[0]
                else:
                    conv = model_copy.module.layer3[0].downsample[0]
            else:
                layer_idx = int(layer.split('_')[1])
                if layer_idx <= 6:
                    block_idx = (layer_idx - 1) // 2
                    conv_idx = (layer_idx - 1) % 2
                    conv = model_copy.module.layer1[block_idx].conv1 if conv_idx == 0 else model_copy.module.layer1[block_idx].conv2
                elif layer_idx <= 14:
                    block_idx = (layer_idx - 7) // 2
                    conv_idx = (layer_idx - 7) % 2
                    conv = model_copy.module.layer2[block_idx].conv1 if conv_idx == 0 else model_copy.module.layer2[block_idx].conv2
                elif layer_idx <= 26:
                    block_idx = (layer_idx - 15) // 2
                    conv_idx = (layer_idx - 15) % 2
                    conv = model_copy.module.layer3[block_idx].conv1 if conv_idx == 0 else model_copy.module.layer3[block_idx].conv2
                else:
                    block_idx = (layer_idx - 27) // 2
                    conv_idx = (layer_idx - 27) % 2
                    conv = model_copy.module.layer4[block_idx].conv1 if conv_idx == 0 else model_copy.module.layer4[block_idx].conv2
        
        num_channels = int(conv.out_channels * prune_ratio)
        
        # Prune the layer
        net_name = 'vgg16' if isinstance(model_copy.module.features[0], nn.Conv2d) else 'resnet34'
        shortcutflag = "downsample" in layer
        model_copy = prune_net(model_copy, False, [layer], [num_channels], net_name, shortcutflag, criterion)
        
        # Evaluate pruned model
        pruned_acc, _, _, _ = eval_epoch(model_copy, test_loader)
        
        # Calculate sensitivity
        sensitivity[layer] = baseline_acc - pruned_acc
    
    return sensitivity

# OPTIMIZATION: Added quantization support
def quantize_model(model, backend='fbgemm'):
    """
    Quantize a neural network model for further compression.
    
    Args:
        model: Neural network model
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        
    Returns:
        Quantized model
    """
    import torch.quantization
    
    # Create a copy of the model to avoid modifying the original
    model_copy = copy.deepcopy(model)
    
    # Set model to evaluation mode
    model_copy.eval()
    
    # Set quantization backend
    torch.backends.quantized.engine = backend
    
    # Define quantization configuration
    qconfig = torch.quantization.get_default_qconfig(backend)
    
    # Prepare model for quantization
    model_copy.qconfig = qconfig
    torch.quantization.prepare(model_copy, inplace=True)
    
    # Convert model to quantized version
    torch.quantization.convert(model_copy, inplace=True)
    
    return model_copy

# OPTIMIZATION: Added quantization-aware training
def prepare_qat(model, backend='fbgemm'):
    """
    Prepare a model for quantization-aware training.
    
    Args:
        model: Neural network model
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        
    Returns:
        Model prepared for quantization-aware training
    """
    import torch.quantization
    
    # Create a copy of the model to avoid modifying the original
    model_copy = copy.deepcopy(model)
    
    # Set quantization backend
    torch.backends.quantized.engine = backend
    
    # Define quantization configuration
    qconfig = torch.quantization.get_default_qat_qconfig(backend)
    
    # Prepare model for quantization-aware training
    model_copy.qconfig = qconfig
    torch.quantization.prepare_qat(model_copy, inplace=True)
    
    return model_copy

if __name__ == '__main__':
    # Example usage
    from netModels.VGG import MyVgg16
    from netModels.ResNet34 import MyResNet34
    from tools.get_data import get_test_loader
    from tools.get_parameters import get_args
    from train import eval_epoch
    
    args = get_args()
    
    # Load model
    if args.net == 'vgg16':
        model = MyVgg16(10)
    else:
        model = MyResNet34()
    
    # Prune model
    if args.pruneflag:
        # Load test data
        test_loader = get_test_loader(args)
        
        # Evaluate original model
        top1_org, top5_org, _, _ = eval_epoch(model, test_loader)
        print(f"Original model: Top1 {top1_org:.4f}, Top5 {top5_org:.4f}")
        
        # Get FLOPs and parameters of original model
        flops_org, params_org = get_flops_params(model, args.net)
        print(f"Original model: FLOPs {flops_org}, Params {params_org}")
        
        # Prune model
        if args.global_pruning:
            # OPTIMIZATION: Use global pruning
            pruned_model = global_pruning(model, args.global_ratio, args.net, args.shortcutflag, args.criterion)
        elif args.iterative_pruning:
            # OPTIMIZATION: Use iterative pruning
            train_loader = get_train_loader(args)
            pruned_model = iterative_pruning(model, args, train_loader, test_loader)
        else:
            # Use original pruning method
            pruned_model = prune_net(model, args.independentflag, args.prune_layers, args.prune_channels, args.net, args.shortcutflag, args.criterion)
        
        # Evaluate pruned model
        top1_pruned, top5_pruned, _, _ = eval_epoch(pruned_model, test_loader)
        print(f"Pruned model: Top1 {top1_pruned:.4f}, Top5 {top5_pruned:.4f}")
        
        # Get FLOPs and parameters of pruned model
        flops_pruned, params_pruned = get_flops_params(pruned_model, args.net)
        print(f"Pruned model: FLOPs {flops_pruned}, Params {params_pruned}")
        
        # OPTIMIZATION: Quantize model if requested
        if args.quantize:
            quantized_model = quantize_model(pruned_model, args.quantize_backend)
            
            # Evaluate quantized model
            top1_quant, top5_quant, _, _ = eval_epoch(quantized_model, test_loader)
            print(f"Quantized model: Top1 {top1_quant:.4f}, Top5 {top5_quant:.4f}")
