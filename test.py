import datetime
import torch
import torch.nn as nn
import os
import argparse
import numpy as np

from netModels.VGG import MyVgg16
from netModels.ResNet34 import MyResNet34
from tools.get_data import get_test_loader
from tools.get_parameters import get_args
from train import eval_epoch
from tools.flops_params import get_flops_params

"""
OPTIMIZATION HIGHLIGHTS:
1. Enhanced testing metrics with detailed reporting
2. Support for different pruning criteria evaluation
3. Batch testing of multiple models
4. Improved visualization and reporting
5. Quantization testing support
6. Better documentation and code organization
"""

def test_model(net, test_loader, args):
    """
    Test a neural network model.
    
    Args:
        net: Neural network model
        test_loader: Test data loader
        args: Command line arguments
        
    Returns:
        Dictionary of test results
    """
    # Evaluate model
    top1, top5, loss, infer_time = eval_epoch(net, test_loader)
    
    # Calculate FLOPs and parameters
    flops, params = get_flops_params(net, args.net)
    
    # Print results
    print("Test results:")
    print("Top-1 accuracy: {:.2f}%".format(top1 * 100))
    print("Top-5 accuracy: {:.2f}%".format(top5 * 100))
    print("Loss: {:.4f}".format(loss))
    print("Inference time: {:.2f}ms".format(infer_time))
    print("FLOPs: {:.2f}M".format(flops / 1e6))
    print("Parameters: {:.2f}M".format(params / 1e6))
    
    # Return results
    return {
        'top1': top1,
        'top5': top5,
        'loss': loss,
        'infer_time': infer_time,
        'flops': flops,
        'params': params
    }

# OPTIMIZATION: Added function to test multiple pruning criteria
def test_pruning_criteria(args):
    """
    Test different pruning criteria on a model.
    
    Args:
        args: Command line arguments
    """
    from prune import prune_net
    
    # Get test loader
    test_loader = get_test_loader(args)
    
    # Get model
    if args.net == 'vgg16':
        model = MyVgg16(10)
    else:
        model = MyResNet34()
    
    # Load model
    load_path = os.path.join("./checkpoint", args.net, "train", "bestParam.pth")
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
    
    # Set up model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    
    # Test original model
    print("Testing original model...")
    original_results = test_model(model, test_loader, args)
    
    # Define pruning criteria to test
    criteria = ['l1', 'l2', 'l1_l2']
    
    # Define pruning layers and channels
    if args.net == 'vgg16':
        prune_layers = ['conv_1', 'conv_5', 'conv_9']
        prune_channels = [32, 128, 256]
    else:  # resnet34
        if args.shortcutflag:
            prune_layers = ['downsample_1', 'downsample_2']
            prune_channels = [64, 128]
        else:
            prune_layers = ['conv_2', 'conv_10', 'conv_18']
            prune_channels = [32, 64, 128]
    
    # Test each criterion
    results = {}
    for criterion in criteria:
        print(f"\nTesting pruning criterion: {criterion}")
        
        # Get fresh model
        if args.net == 'vgg16':
            pruned_model = MyVgg16(10)
        else:
            pruned_model = MyResNet34()
        
        # Load model
        if os.path.exists(load_path):
            pruned_model.load_state_dict(torch.load(load_path))
        
        # Set up model for testing
        pruned_model = nn.DataParallel(pruned_model, device_ids=[0, 1])
        pruned_model = pruned_model.to(device)
        
        # Prune model
        pruned_model = prune_net(
            pruned_model,
            args.independentflag,
            prune_layers,
            prune_channels,
            args.net,
            args.shortcutflag,
            criterion
        )
        
        # Test pruned model
        results[criterion] = test_model(pruned_model, test_loader, args)
    
    # Compare results
    print("\nComparison of pruning criteria:")
    print("Criterion | Top-1 Acc | Top-5 Acc | FLOPs (M) | Params (M) | Inference (ms)")
    print("-" * 75)
    print("Original  | {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
        original_results['top1'] * 100,
        original_results['top5'] * 100,
        original_results['flops'] / 1e6,
        original_results['params'] / 1e6,
        original_results['infer_time']
    ))
    
    for criterion, result in results.items():
        print("{:<9} | {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
            criterion,
            result['top1'] * 100,
            result['top5'] * 100,
            result['flops'] / 1e6,
            result['params'] / 1e6,
            result['infer_time']
        ))

# OPTIMIZATION: Added function to test global pruning
def test_global_pruning(args):
    """
    Test global pruning on a model.
    
    Args:
        args: Command line arguments
    """
    from prune import global_pruning
    
    # Get test loader
    test_loader = get_test_loader(args)
    
    # Get model
    if args.net == 'vgg16':
        model = MyVgg16(10)
    else:
        model = MyResNet34()
    
    # Load model
    load_path = os.path.join("./checkpoint", args.net, "train", "bestParam.pth")
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
    
    # Set up model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    
    # Test original model
    print("Testing original model...")
    original_results = test_model(model, test_loader, args)
    
    # Define pruning ratios to test
    pruning_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Test each pruning ratio
    results = {}
    for ratio in pruning_ratios:
        print(f"\nTesting global pruning ratio: {ratio}")
        
        # Get fresh model
        if args.net == 'vgg16':
            pruned_model = MyVgg16(10)
        else:
            pruned_model = MyResNet34()
        
        # Load model
        if os.path.exists(load_path):
            pruned_model.load_state_dict(torch.load(load_path))
        
        # Set up model for testing
        pruned_model = nn.DataParallel(pruned_model, device_ids=[0, 1])
        pruned_model = pruned_model.to(device)
        
        # Prune model
        pruned_model = global_pruning(
            pruned_model,
            ratio,
            args.net,
            args.shortcutflag,
            args.criterion
        )
        
        # Test pruned model
        results[ratio] = test_model(pruned_model, test_loader, args)
    
    # Compare results
    print("\nComparison of global pruning ratios:")
    print("Ratio    | Top-1 Acc | Top-5 Acc | FLOPs (M) | Params (M) | Inference (ms)")
    print("-" * 75)
    print("Original | {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
        original_results['top1'] * 100,
        original_results['top5'] * 100,
        original_results['flops'] / 1e6,
        original_results['params'] / 1e6,
        original_results['infer_time']
    ))
    
    for ratio, result in results.items():
        print("{:<8} | {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
            ratio,
            result['top1'] * 100,
            result['top5'] * 100,
            result['flops'] / 1e6,
            result['params'] / 1e6,
            result['infer_time']
        ))

# OPTIMIZATION: Added function to test iterative pruning
def test_iterative_pruning(args):
    """
    Test iterative pruning on a model.
    
    Args:
        args: Command line arguments
    """
    from prune import prune_net, iterative_pruning
    from tools.get_data import get_train_loader
    
    # Get data loaders
    test_loader = get_test_loader(args)
    train_loader = get_train_loader(args)
    
    # Get model
    if args.net == 'vgg16':
        model = MyVgg16(10)
    else:
        model = MyResNet34()
    
    # Load model
    load_path = os.path.join("./checkpoint", args.net, "train", "bestParam.pth")
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
    
    # Set up model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    
    # Test original model
    print("Testing original model...")
    original_results = test_model(model, test_loader, args)
    
    # Define pruning layers and channels
    if args.net == 'vgg16':
        prune_layers = ['conv_1', 'conv_5', 'conv_9']
        prune_channels = [32, 128, 256]
    else:  # resnet34
        if args.shortcutflag:
            prune_layers = ['downsample_1', 'downsample_2']
            prune_channels = [64, 128]
        else:
            prune_layers = ['conv_2', 'conv_10', 'conv_18']
            prune_channels = [32, 64, 128]
    
    # Test one-shot pruning
    print("\nTesting one-shot pruning...")
    
    # Get fresh model
    if args.net == 'vgg16':
        oneshot_model = MyVgg16(10)
    else:
        oneshot_model = MyResNet34()
    
    # Load model
    if os.path.exists(load_path):
        oneshot_model.load_state_dict(torch.load(load_path))
    
    # Set up model for testing
    oneshot_model = nn.DataParallel(oneshot_model, device_ids=[0, 1])
    oneshot_model = oneshot_model.to(device)
    
    # Prune model
    oneshot_model = prune_net(
        oneshot_model,
        args.independentflag,
        prune_layers,
        prune_channels,
        args.net,
        args.shortcutflag,
        args.criterion
    )
    
    # Test pruned model
    oneshot_results = test_model(oneshot_model, test_loader, args)
    
    # Test iterative pruning
    print("\nTesting iterative pruning...")
    
    # Get fresh model
    if args.net == 'vgg16':
        iterative_model = MyVgg16(10)
    else:
        iterative_model = MyResNet34()
    
    # Load model
    if os.path.exists(load_path):
        iterative_model.load_state_dict(torch.load(load_path))
    
    # Set up model for testing
    iterative_model = nn.DataParallel(iterative_model, device_ids=[0, 1])
    iterative_model = iterative_model.to(device)
    
    # Set up args for iterative pruning
    args.prune_layers = prune_layers
    args.prune_channels = prune_channels
    args.prune_iterations = 3
    args.intermediate_epochs = 5
    
    # Prune model iteratively
    iterative_model = iterative_pruning(
        iterative_model,
        args,
        train_loader,
        test_loader
    )
    
    # Test pruned model
    iterative_results = test_model(iterative_model, test_loader, args)
    
    # Compare results
    print("\nComparison of pruning methods:")
    print("Method     | Top-1 Acc | Top-5 Acc | FLOPs (M) | Params (M) | Inference (ms)")
    print("-" * 80)
    print("Original   | {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
        original_results['top1'] * 100,
        original_results['top5'] * 100,
        original_results['flops'] / 1e6,
        original_results['params'] / 1e6,
        original_results['infer_time']
    ))
    
    print("One-shot   | {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
        oneshot_results['top1'] * 100,
        oneshot_results['top5'] * 100,
        oneshot_results['flops'] / 1e6,
        oneshot_results['params'] / 1e6,
        oneshot_results['infer_time']
    ))
    
    print("Iterative  | {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
        iterative_results['top1'] * 100,
        iterative_results['top5'] * 100,
        iterative_results['flops'] / 1e6,
        iterative_results['params'] / 1e6,
        iterative_results['infer_time']
    ))

# OPTIMIZATION: Added function to test knowledge distillation
def test_distillation(args):
    """
    Test knowledge distillation on a pruned model.
    
    Args:
        args: Command line arguments
    """
    from prune import prune_net
    from train import training, training_with_distillation
    from tools.get_data import get_train_loader
    import copy
    
    # Get data loaders
    test_loader = get_test_loader(args)
    train_loader = get_train_loader(args)
    
    # Get model
    if args.net == 'vgg16':
        model = MyVgg16(10)
    else:
        model = MyResNet34()
    
    # Load model
    load_path = os.path.join("./checkpoint", args.net, "train", "bestParam.pth")
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
    
    # Set up model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    
    # Test original model
    print("Testing original model...")
    original_results = test_model(model, test_loader, args)
    
    # Define pruning layers and channels
    if args.net == 'vgg16':
        prune_layers = ['conv_1', 'conv_5', 'conv_9']
        prune_channels = [32, 128, 256]
    else:  # resnet34
        if args.shortcutflag:
            prune_layers = ['downsample_1', 'downsample_2']
            prune_channels = [64, 128]
        else:
            prune_layers = ['conv_2', 'conv_10', 'conv_18']
            prune_channels = [32, 64, 128]
    
    # Create pruned model
    pruned_model = copy.deepcopy(model)
    pruned_model = prune_net(
        pruned_model,
        args.independentflag,
        prune_layers,
        prune_channels,
        args.net,
        args.shortcutflag,
        args.criterion
    )
    
    # Test pruned model before retraining
    print("\nTesting pruned model before retraining...")
    pruned_results = test_model(pruned_model, test_loader, args)
    
    # Standard retraining
    print("\nTesting standard retraining...")
    
    # Get fresh pruned model
    standard_model = copy.deepcopy(model)
    standard_model = prune_net(
        standard_model,
        args.independentflag,
        prune_layers,
        prune_channels,
        args.net,
        args.shortcutflag,
        args.criterion
    )
    
    # Retrain model
    standard_model = training(
        standard_model,
        args.retrainepoch,
        train_loader,
        test_loader,
        True,
        args.retrainlr,
        args.optim,
        os.path.join("./checkpoint", args.net, "retrain", "standard"),
        os.path.join("./checkpoint", args.net, "retrain", "standard")
    )
    
    # Test retrained model
    standard_results = test_model(standard_model, test_loader, args)
    
    # Distillation retraining
    print("\nTesting retraining with knowledge distillation...")
    
    # Get fresh pruned model
    distill_model = copy.deepcopy(model)
    distill_model = prune_net(
        distill_model,
        args.independentflag,
        prune_layers,
        prune_channels,
        args.net,
        args.shortcutflag,
        args.criterion
    )
    
    # Retrain model with distillation
    distill_model = training_with_distillation(
        distill_model,
        model,  # Original model as teacher
        args.retrainepoch,
        train_loader,
        test_loader,
        True,
        args.retrainlr,
        args.optim,
        os.path.join("./checkpoint", args.net, "retrain", "distill"),
        os.path.join("./checkpoint", args.net, "retrain", "distill"),
        4.0,  # Temperature
        0.5   # Alpha
    )
    
    # Test distilled model
    distill_results = test_model(distill_model, test_loader, args)
    
    # Compare results
    print("\nComparison of retraining methods:")
    print("Method      | Top-1 Acc | Top-5 Acc | FLOPs (M) | Params (M) | Inference (ms)")
    print("-" * 80)
    print("Original    | {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
        original_results['top1'] * 100,
        original_results['top5'] * 100,
        original_results['flops'] / 1e6,
        original_results['params'] / 1e6,
        original_results['infer_time']
    ))
    
    print("Pruned      | {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
        pruned_results['top1'] * 100,
        pruned_results['top5'] * 100,
        pruned_results['flops'] / 1e6,
        pruned_results['params'] / 1e6,
        pruned_results['infer_time']
    ))
    
    print("Standard    | {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
        standard_results['top1'] * 100,
        standard_results['top5'] * 100,
        standard_results['flops'] / 1e6,
        standard_results['params'] / 1e6,
        standard_results['infer_time']
    ))
    
    print("Distillation| {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
        distill_results['top1'] * 100,
        distill_results['top5'] * 100,
        distill_results['flops'] / 1e6,
        distill_results['params'] / 1e6,
        distill_results['infer_time']
    ))

# OPTIMIZATION: Added function to test quantization
def test_quantization(args):
    """
    Test quantization on a pruned model.
    
    Args:
        args: Command line arguments
    """
    from prune import prune_net, quantize_model
    
    # Get test loader
    test_loader = get_test_loader(args)
    
    # Get model
    if args.net == 'vgg16':
        model = MyVgg16(10)
    else:
        model = MyResNet34()
    
    # Load model
    load_path = os.path.join("./checkpoint", args.net, "train", "bestParam.pth")
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
    
    # Set up model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    
    # Test original model
    print("Testing original model...")
    original_results = test_model(model, test_loader, args)
    
    # Define pruning layers and channels
    if args.net == 'vgg16':
        prune_layers = ['conv_1', 'conv_5', 'conv_9']
        prune_channels = [32, 128, 256]
    else:  # resnet34
        if args.shortcutflag:
            prune_layers = ['downsample_1', 'downsample_2']
            prune_channels = [64, 128]
        else:
            prune_layers = ['conv_2', 'conv_10', 'conv_18']
            prune_channels = [32, 64, 128]
    
    # Create pruned model
    pruned_model = prune_net(
        model,
        args.independentflag,
        prune_layers,
        prune_channels,
        args.net,
        args.shortcutflag,
        args.criterion
    )
    
    # Test pruned model
    print("\nTesting pruned model...")
    pruned_results = test_model(pruned_model, test_loader, args)
    
    # Quantize model
    print("\nTesting quantized model...")
    quantized_model = quantize_model(pruned_model, 'fbgemm')
    
    # Test quantized model
    quantized_results = test_model(quantized_model, test_loader, args)
    
    # Compare results
    print("\nComparison of compression methods:")
    print("Method    | Top-1 Acc | Top-5 Acc | FLOPs (M) | Params (M) | Inference (ms)")
    print("-" * 75)
    print("Original  | {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
        original_results['top1'] * 100,
        original_results['top5'] * 100,
        original_results['flops'] / 1e6,
        original_results['params'] / 1e6,
        original_results['infer_time']
    ))
    
    print("Pruned    | {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
        pruned_results['top1'] * 100,
        pruned_results['top5'] * 100,
        pruned_results['flops'] / 1e6,
        pruned_results['params'] / 1e6,
        pruned_results['infer_time']
    ))
    
    print("Quantized | {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
        quantized_results['top1'] * 100,
        quantized_results['top5'] * 100,
        quantized_results['flops'] / 1e6,
        quantized_results['params'] / 1e6,
        quantized_results['infer_time']
    ))

if __name__ == '__main__':
    args = get_args()
    
    # Get test loader
    test_loader = get_test_loader(args)
    
    # Get model
    if args.net == 'vgg16':
        model = MyVgg16(10)
    else:
        model = MyResNet34()
    
    # Load model
    load_path = os.path.join("./checkpoint", args.net, "train", "bestParam.pth")
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
    
    # Set up model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    
    # OPTIMIZATION: Run different tests based on arguments
    if hasattr(args, 'test_criteria') and args.test_criteria:
        test_pruning_criteria(args)
    elif hasattr(args, 'test_iterative') and args.test_iterative:
        test_iterative_pruning(args)
    elif hasattr(args, 'test_distillation') and args.test_distillation:
        test_distillation(args)
    elif hasattr(args, 'test_global') and args.test_global:
        test_global_pruning(args)
    elif hasattr(args, 'test_quantization') and args.test_quantization:
        test_quantization(args)
    else:
        # Standard test
        test_model(model, test_loader, args)
