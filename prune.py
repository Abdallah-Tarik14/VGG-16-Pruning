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
7. Dynamic device handling for better compatibility
8. Enhanced error handling
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
    try:
        # Ensure tensor is on CPU for numpy operations
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
            l1_norm = l1_norm / np.max(l1_norm) if np.max(l1_norm) > 0 else l1_norm
            l2_norm = l2_norm / np.max(l2_norm) if np.max(l2_norm) > 0 else l2_norm
            # Combine with equal weights
            norm = 0.5 * l1_norm + 0.5 * l2_norm
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
            
        norm_argsort = np.argsort(norm)
        return norm_argsort
    except Exception as e:
        print(f"Error in channels_index: {e}")
        raise

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
    try:
        norm_argsort = channels_index(weight_torch, criterion)
        # Select channels to keep (exclude the smallest num_channels)
        selected_channels = norm_argsort[num_channels:]
        return selected_channels
    except Exception as e:
        print(f"Error in select_channels: {e}")
        raise

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
    try:
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
    except Exception as e:
        print(f"Error in prune_conv: {e}")
        raise

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
    try:
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
    except Exception as e:
        print(f"Error in prune_related_conv: {e}")
        raise

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
    try:
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
    except Exception as e:
        print(f"Error in prune_batchnorm: {e}")
        raise

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
    try:
        # Handle both DataParallel and non-DataParallel models
        features = model.module.features if hasattr(model, 'module') else model.features
        
        # Get the convolutional layer to prune
        conv_to_prune = features[layer_index]
        
        # Select channels to keep
        selected_channels = select_channels(conv_to_prune.weight, num_channels, criterion)
        
        # Prune the convolutional layer
        new_conv = prune_conv(conv_to_prune, selected_channels, device)
        
        # Prune the batch normalization layer
        bn_to_prune = features[layer_index + 1]
        new_bn = prune_batchnorm(bn_to_prune, selected_channels, device)
        
        # Replace the pruned layers in the model
        if hasattr(model, 'module'):
            model.module.features[layer_index] = new_conv
            model.module.features[layer_index + 1] = new_bn
        else:
            model.features[layer_index] = new_conv
            model.features[layer_index + 1] = new_bn
        
        # If not the last convolutional layer, prune the next convolutional layer's input channels
        if layer_index + 3 < len(features):
            next_conv = features[layer_index + 3]
            if isinstance(next_conv, nn.Conv2d):
                new_next_conv = prune_related_conv(next_conv, selected_channels, device)
                if hasattr(model, 'module'):
                    model.module.features[layer_index + 3] = new_next_conv
                else:
                    model.features[layer_index + 3] = new_next_conv
        
        # If it's the last convolutional layer, prune the classifier's input features
        else:
            classifier = model.module.classifier if hasattr(model, 'module') else model.classifier
            if isinstance(classifier[0], nn.Linear):
                in_features = classifier[0].in_features
                out_features = classifier[0].out_features
                
                # Calculate new in_features based on selected channels
                conv_out_channels = conv_to_prune.out_channels
                new_in_features = len(selected_channels) * (in_features // conv_out_channels)
                
                # Create new linear layer
                new_linear = nn.Linear(new_in_features, out_features)
                
                # Copy weights for selected channels
                old_weights = classifier[0].weight.data
                new_weights = torch.zeros((out_features, new_in_features))
                
                for i, idx in enumerate(selected_channels):
                    new_weights[:, i * (new_in_features // len(selected_channels)):(i + 1) * (new_in_features // len(selected_channels))] = \
                        old_weights[:, idx * (in_features // conv_out_channels):(idx + 1) * (in_features // conv_out_channels)]
                
                new_linear.weight.data = new_weights.to(device)
                new_linear.bias.data = classifier[0].bias.data.clone()
                
                # Replace the linear layer
                if hasattr(model, 'module'):
                    model.module.classifier[0] = new_linear
                else:
                    model.classifier[0] = new_linear
        
        return model
    except Exception as e:
        print(f"Error in prune_vgg16_conv_layer: {e}")
        raise

# OPTIMIZATION: Added global pruning function
def global_pruning(model, pruning_ratio, net_name, shortcutflag=False, criterion="l1"):
    """
    Perform global pruning on a model.
    
    Args:
        model: Neural network model
        pruning_ratio: Ratio of filters to prune globally
        net_name: Name of the network architecture
        shortcutflag: Whether to prune shortcut connections
        criterion: Pruning criterion ('l1', 'l2', 'l1_l2')
        
    Returns:
        Pruned model
    """
    try:
        # Determine device
        device = next(model.parameters()).device
        
        # Get all convolutional layers
        conv_layers = []
        if net_name == 'vgg16':
            features = model.module.features if hasattr(model, 'module') else model.features
            for i, layer in enumerate(features):
                if isinstance(layer, nn.Conv2d):
                    conv_layers.append((i, layer))
        else:  # resnet34
            # Implementation for ResNet34 would go here
            pass
        
        # Calculate importance of all filters
        all_filters = []
        for i, conv in conv_layers:
            weight = conv.weight
            for j in range(weight.size(0)):  # For each filter
                filter_weight = weight[j:j+1]
                if criterion == "l1":
                    importance = filter_weight.abs().sum().item()
                elif criterion == "l2":
                    importance = filter_weight.pow(2).sum().sqrt().item()
                else:  # l1_l2
                    l1 = filter_weight.abs().sum().item()
                    l2 = filter_weight.pow(2).sum().sqrt().item()
                    importance = 0.5 * l1 + 0.5 * l2
                all_filters.append((i, j, importance))
        
        # Sort filters by importance
        all_filters.sort(key=lambda x: x[2])
        
        # Determine number of filters to prune
        total_filters = len(all_filters)
        num_to_prune = int(total_filters * pruning_ratio)
        
        # Group filters by layer
        filters_to_prune = {}
        for i in range(num_to_prune):
            layer_idx, filter_idx, _ = all_filters[i]
            if layer_idx not in filters_to_prune:
                filters_to_prune[layer_idx] = []
            filters_to_prune[layer_idx].append(filter_idx)
        
        # Prune filters layer by layer
        for layer_idx, filter_indices in filters_to_prune.items():
            # Get all filters in this layer
            conv = conv_layers[layer_idx][1]
            total_filters_in_layer = conv.weight.size(0)
            
            # Create mask of filters to keep
            keep_mask = torch.ones(total_filters_in_layer, dtype=torch.bool)
            keep_mask[filter_indices] = False
            
            # Get indices of filters to keep
            selected_channels = torch.nonzero(keep_mask).squeeze().tolist()
            if isinstance(selected_channels, int):  # Handle case of single channel
                selected_channels = [selected_channels]
            
            # Prune the layer
            if net_name == 'vgg16':
                model = prune_vgg16_conv_layer(model, layer_idx, len(filter_indices), device, criterion)
        
        return model
    except Exception as e:
        print(f"Error in global_pruning: {e}")
        raise

# OPTIMIZATION: Added iterative pruning function
def iterative_pruning(model, args, train_loader, test_loader):
    """
    Perform iterative pruning on a model.
    
    Args:
        model: Neural network model
        args: Command line arguments
        train_loader: Training data loader
        test_loader: Test data loader
        
    Returns:
        Pruned model
    """
    try:
        from train import training
        
        # Determine device
        device = next(model.parameters()).device
        
        # Get pruning parameters
        prune_layers = args.prune_layers
        prune_channels = args.prune_channels
        iterations = args.prune_iterations
        intermediate_epochs = args.intermediate_epochs
        
        # Calculate channels to prune per iteration
        channels_per_iteration = []
        for channels in prune_channels:
            channels_per_iteration.append(channels // iterations)
        
        # Iterative pruning
        for i in range(iterations):
            print(f"\nIteration {i+1}/{iterations}")
            
            # Prune model
            model = prune_net(
                model,
                args.independentflag,
                prune_layers,
                channels_per_iteration,
                args.net,
                args.shortcutflag,
                args.criterion
            )
            
            # Retrain model
            model = training(
                model,
                intermediate_epochs,
                train_loader,
                test_loader,
                True,
                args.retrainlr,
                args.optim,
                os.path.join("./checkpoint", args.net, "iterative", f"iter_{i+1}"),
                os.path.join("./checkpoint", args.net, "iterative", f"iter_{i+1}")
            )
        
        return model
    except Exception as e:
        print(f"Error in iterative_pruning: {e}")
        raise

def prune_net(model, independentflag, prune_layers, prune_channels, net_name, shortcutflag=False, criterion="l1"):
    """
    Prune a neural network model.
    
    Args:
        model: Neural network model
        independentflag: Whether to use independent pruning strategy
        prune_layers: Layers to prune
        prune_channels: Number of channels to prune for each layer
        net_name: Name of the network architecture
        shortcutflag: Whether to prune shortcut connections
        criterion: Pruning criterion ('l1', 'l2', 'l1_l2')
        
    Returns:
        Pruned model
    """
    try:
        # Determine device
        device = next(model.parameters()).device
        
        # Create a copy of the model for pruning
        pruned_model = copy.deepcopy(model)
        
        # Prune each layer
        for i, (layer, channels) in enumerate(zip(prune_layers, prune_channels)):
            print(f"Pruning layer {layer} with {channels} channels...")
            
            if net_name == 'vgg16':
                # Extract layer index from layer name
                layer_index = int(layer.split('_')[1])
                
                # Prune the layer
                pruned_model = prune_vgg16_conv_layer(pruned_model, layer_index, channels, device, criterion)
            else:  # resnet34
                # Prune ResNet34 layer
                pruned_model = prune_resnet_conv_layer(pruned_model, layer, channels, device, shortcutflag, criterion)
        
        # Calculate FLOPs and parameters
        flops, params = get_flops_params(pruned_model, net_name)
        print(f"Pruned model: {flops / 1e6:.2f}M FLOPs, {params / 1e6:.2f}M parameters")
        
        return pruned_model
    except Exception as e:
        print(f"Error in prune_net: {e}")
        raise

# OPTIMIZATION: Added function for layer sensitivity analysis
def analyze_layer_sensitivity(model, test_loader, net_name, device, criterion="l1"):
    """
    Analyze the sensitivity of each layer to pruning.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        net_name: Name of the network architecture
        device: Device to use (CPU or GPU)
        criterion: Pruning criterion ('l1', 'l2', 'l1_l2')
        
    Returns:
        Dictionary mapping layer names to sensitivity scores
    """
    try:
        from train import eval_epoch
        
        # Get baseline accuracy
        baseline_acc, _, _, _ = eval_epoch(model, test_loader)
        
        # Initialize sensitivity scores
        sensitivity = {}
        
        # Analyze each layer
        if net_name == 'vgg16':
            features = model.module.features if hasattr(model, 'module') else model.features
            for i, layer in enumerate(features):
                if isinstance(layer, nn.Conv2d):
                    # Create a copy of the model
                    pruned_model = copy.deepcopy(model)
                    
                    # Prune 10% of channels
                    num_channels = max(1, int(layer.out_channels * 0.1))
                    pruned_model = prune_vgg16_conv_layer(pruned_model, i, num_channels, device, criterion)
                    
                    # Evaluate pruned model
                    pruned_acc, _, _, _ = eval_epoch(pruned_model, test_loader)
                    
                    # Calculate sensitivity
                    sensitivity[f"conv_{i}"] = baseline_acc - pruned_acc
        else:  # resnet34
            # Implementation for ResNet34 would go here
            pass
        
        return sensitivity
    except Exception as e:
        print(f"Error in analyze_layer_sensitivity: {e}")
        raise

# OPTIMIZATION: Added function for quantization
def quantize_model(model, backend="fbgemm"):
    """
    Quantize a model for inference.
    
    Args:
        model: Neural network model
        backend: Quantization backend ('fbgemm' or 'qnnpack')
        
    Returns:
        Quantized model
    """
    try:
        import torch.quantization
        
        # Set quantization backend
        if backend == "fbgemm":
            torch.backends.quantized.engine = 'fbgemm'
        else:  # qnnpack
            torch.backends.quantized.engine = 'qnnpack'
        
        # Create a copy of the model for quantization
        quantized_model = copy.deepcopy(model)
        
        # Prepare model for quantization
        quantized_model.eval()
        
        # Fuse modules
        quantized_model = torch.quantization.fuse_modules(quantized_model, [['conv', 'bn', 'relu']])
        
        # Prepare for static quantization
        quantized_model = torch.quantization.prepare(quantized_model)
        
        # Calibrate with data (would need a calibration dataset)
        # for data, target in calibration_loader:
        #     quantized_model(data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(quantized_model)
        
        return quantized_model
    except Exception as e:
        print(f"Error in quantize_model: {e}")
        raise
