from netModels.ResNet34 import MyResNet34
from netModels.VGG import MyVgg16
import torch
import torch.nn as nn
import numpy as np

"""
OPTIMIZATION HIGHLIGHTS:
1. Enhanced FLOPs calculation with better accuracy
2. Improved parameter counting with detailed breakdown
3. Support for different model architectures
4. Memory usage estimation
5. Layer-wise analysis capabilities
"""

def get_flops_params(model, net_name):
    """
    Calculate FLOPs and parameters for a neural network model.
    
    Args:
        model: Neural network model
        net_name: Name of the network architecture
        
    Returns:
        Tuple of (FLOPs, parameters)
    """
    # Get input size based on network architecture
    if net_name == 'vgg16':
        input_size = (1, 3, 32, 32)  # CIFAR-10 input size
    else:  # resnet34
        input_size = (1, 3, 224, 224)  # ImageNet input size
    
    # Create dummy input
    dummy_input = torch.randn(*input_size)
    
    # Calculate FLOPs
    flops = calculate_flops(model, dummy_input)
    
    # Calculate parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return flops, params

# OPTIMIZATION: Enhanced FLOPs calculation
def calculate_flops(model, input_tensor):
    """
    Calculate FLOPs for a neural network model.
    
    Args:
        model: Neural network model
        input_tensor: Input tensor
        
    Returns:
        Number of FLOPs
    """
    # Set model to evaluation mode
    model.eval()
    
    # Register hooks for each module
    flops_dict = {}
    handles = []
    
    def conv_hook(module, input, output):
        # Get input dimensions
        input = input[0]
        batch_size, input_channels, input_height, input_width = input.size()
        
        # Get output dimensions
        output_channels, output_height, output_width = output.size()[1:]
        
        # Get kernel dimensions
        kernel_height, kernel_width = module.kernel_size
        
        # Calculate FLOPs
        flops = batch_size * output_channels * output_height * output_width * (input_channels * kernel_height * kernel_width + 1)
        
        # Store FLOPs
        flops_dict[module] = flops
    
    def linear_hook(module, input, output):
        # Get input dimensions
        input = input[0]
        batch_size = input.size(0)
        
        # Calculate FLOPs
        flops = batch_size * module.in_features * module.out_features
        
        # Store FLOPs
        flops_dict[module] = flops
    
    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    
    # Forward pass
    with torch.no_grad():
        model(input_tensor)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Calculate total FLOPs
    total_flops = sum(flops_dict.values())
    
    return total_flops

# OPTIMIZATION: Added function for detailed parameter breakdown
def get_parameter_breakdown(model):
    """
    Get a detailed breakdown of parameters for a neural network model.
    
    Args:
        model: Neural network model
        
    Returns:
        Dictionary mapping layer names to parameter counts
    """
    # Initialize parameter breakdown
    param_breakdown = {}
    
    # Iterate over named parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Get layer name
            layer_name = name.split('.')[0]
            
            # Update parameter count
            if layer_name in param_breakdown:
                param_breakdown[layer_name] += param.numel()
            else:
                param_breakdown[layer_name] = param.numel()
    
    return param_breakdown

# OPTIMIZATION: Added function for memory usage estimation
def estimate_memory_usage(model, input_tensor):
    """
    Estimate memory usage for a neural network model.
    
    Args:
        model: Neural network model
        input_tensor: Input tensor
        
    Returns:
        Estimated memory usage in bytes
    """
    # Set model to evaluation mode
    model.eval()
    
    # Register hooks for each module
    memory_dict = {}
    handles = []
    
    def hook(module, input, output):
        # Calculate memory usage for input
        input_memory = sum(x.numel() * x.element_size() for x in input if x is not None)
        
        # Calculate memory usage for output
        output_memory = output.numel() * output.element_size()
        
        # Calculate memory usage for parameters
        param_memory = sum(p.numel() * p.element_size() for p in module.parameters() if p.requires_grad)
        
        # Store memory usage
        memory_dict[module] = input_memory + output_memory + param_memory
    
    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            handles.append(module.register_forward_hook(hook))
    
    # Forward pass
    with torch.no_grad():
        model(input_tensor)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Calculate total memory usage
    total_memory = sum(memory_dict.values())
    
    return total_memory

# OPTIMIZATION: Added function for layer-wise analysis
def analyze_layers(model, input_tensor):
    """
    Analyze layers of a neural network model.
    
    Args:
        model: Neural network model
        input_tensor: Input tensor
        
    Returns:
        Dictionary mapping layer names to analysis results
    """
    # Set model to evaluation mode
    model.eval()
    
    # Register hooks for each module
    layer_dict = {}
    handles = []
    
    def hook(module, input, output):
        # Get layer name
        for name, m in model.named_modules():
            if m is module:
                layer_name = name
                break
        else:
            layer_name = str(module)
        
        # Calculate FLOPs
        if isinstance(module, nn.Conv2d):
            # Get input dimensions
            input = input[0]
            batch_size, input_channels, input_height, input_width = input.size()
            
            # Get output dimensions
            output_channels, output_height, output_width = output.size()[1:]
            
            # Get kernel dimensions
            kernel_height, kernel_width = module.kernel_size
            
            # Calculate FLOPs
            flops = batch_size * output_channels * output_height * output_width * (input_channels * kernel_height * kernel_width + 1)
        elif isinstance(module, nn.Linear):
            # Get input dimensions
            input = input[0]
            batch_size = input.size(0)
            
            # Calculate FLOPs
            flops = batch_size * module.in_features * module.out_features
        else:
            flops = 0
        
        # Calculate parameters
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # Calculate memory usage
        memory = output.numel() * output.element_size()
        
        # Store analysis results
        layer_dict[layer_name] = {
            'type': module.__class__.__name__,
            'input_shape': tuple(input[0].shape),
            'output_shape': tuple(output.shape),
            'flops': flops,
            'params': params,
            'memory': memory
        }
    
    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d)):
            handles.append(module.register_forward_hook(hook))
    
    # Forward pass
    with torch.no_grad():
        model(input_tensor)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    return layer_dict

if __name__ == '__main__':
    # Example usage
    vgg16 = MyVgg16(10)
    resnet34 = MyResNet34()
    
    # Calculate FLOPs and parameters
    vgg16_flops, vgg16_params = get_flops_params(vgg16, 'vgg16')
    resnet34_flops, resnet34_params = get_flops_params(resnet34, 'resnet34')
    
    print(f"VGG16: {vgg16_flops / 1e9:.2f} GFLOPs, {vgg16_params / 1e6:.2f}M parameters")
    print(f"ResNet34: {resnet34_flops / 1e9:.2f} GFLOPs, {resnet34_params / 1e6:.2f}M parameters")
    
    # Get parameter breakdown
    vgg16_breakdown = get_parameter_breakdown(vgg16)
    resnet34_breakdown = get_parameter_breakdown(resnet34)
    
    print("\nVGG16 parameter breakdown:")
    for layer, params in vgg16_breakdown.items():
        print(f"{layer}: {params / 1e6:.2f}M parameters")
    
    print("\nResNet34 parameter breakdown:")
    for layer, params in resnet34_breakdown.items():
        print(f"{layer}: {params / 1e6:.2f}M parameters")
