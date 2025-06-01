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
7. Dynamic device handling for better compatibility
8. Enhanced error handling
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
    try:
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
    except Exception as e:
        print(f"Error in test_model: {e}")
        raise

# OPTIMIZATION: Added function to test multiple pruning criteria
def test_pruning_criteria(args):
    """
    Test different pruning criteria on a model.
    
    Args:
        args: Command line arguments
    """
    try:
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
            model.load_state_dict(torch.load(load_path, map_location='cpu'))
        else:
            print(f"Warning: Model checkpoint not found at {load_path}")
        
        # Set up model for testing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
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
                pruned_model.load_state_dict(torch.load(load_path, map_location='cpu'))
            
            # Set up model for testing
            if torch.cuda.device_count() > 1:
                pruned_model = nn.DataParallel(pruned_model)
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
    except Exception as e:
        print(f"Error in test_pruning_criteria: {e}")
        raise

# OPTIMIZATION: Added function to test global pruning
def test_global_pruning(args):
    """
    Test global pruning on a model.
    
    Args:
        args: Command line arguments
    """
    try:
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
            model.load_state_dict(torch.load(load_path, map_location='cpu'))
        else:
            print(f"Warning: Model checkpoint not found at {load_path}")
        
        # Set up model for testing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
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
                pruned_model.load_state_dict(torch.load(load_path, map_location='cpu'))
            
            # Set up model for testing
            if torch.cuda.device_count() > 1:
                pruned_model = nn.DataParallel(pruned_model)
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
    except Exception as e:
        print(f"Error in test_global_pruning: {e}")
        raise

# OPTIMIZATION: Added function to test iterative pruning
def test_iterative_pruning(args):
    """
    Test iterative pruning on a model.
    
    Args:
        args: Command line arguments
    """
    try:
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
            model.load_state_dict(torch.load(load_path, map_location='cpu'))
        else:
            print(f"Warning: Model checkpoint not found at {load_path}")
        
        # Set up model for testing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
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
            oneshot_model.load_state_dict(torch.load(load_path, map_location='cpu'))
        
        # Set up model for testing
        if torch.cuda.device_count() > 1:
            oneshot_model = nn.DataParallel(oneshot_model)
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
            iterative_model.load_state_dict(torch.load(load_path, map_location='cpu'))
        
        # Set up model for testing
        if torch.cuda.device_count() > 1:
            iterative_model = nn.DataParallel(iterative_model)
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
    except Exception as e:
        print(f"Error in test_iterative_pruning: {e}")
        raise

# OPTIMIZATION: Added function to test quantization
def test_quantization(args):
    """
    Test quantization on a model.
    
    Args:
        args: Command line arguments
    """
    try:
        import torch.quantization
        
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
            model.load_state_dict(torch.load(load_path, map_location='cpu'))
        else:
            print(f"Warning: Model checkpoint not found at {load_path}")
        
        # Test original model (on CPU for fair comparison with quantized model)
        model = model.cpu()
        print("Testing original model...")
        original_results = test_model(model, test_loader, args)
        
        # Prepare model for quantization
        model.eval()
        
        # Set quantization backend
        if args.quantize_backend == "fbgemm":
            torch.backends.quantized.engine = 'fbgemm'
        else:  # qnnpack
            torch.backends.quantized.engine = 'qnnpack'
        
        # Fuse modules
        model_fused = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
        
        # Prepare for static quantization
        model_prepared = torch.quantization.prepare(model_fused)
        
        # Calibrate with data
        print("Calibrating quantization...")
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            model_prepared(inputs)
            if batch_idx >= 10:  # Calibrate with 10 batches
                break
        
        # Convert to quantized model
        model_quantized = torch.quantization.convert(model_prepared)
        
        # Test quantized model
        print("Testing quantized model...")
        quantized_results = test_model(model_quantized, test_loader, args)
        
        # Compare results
        print("\nComparison of original vs quantized model:")
        print("Model     | Top-1 Acc | Top-5 Acc | FLOPs (M) | Params (M) | Inference (ms)")
        print("-" * 80)
        print("Original  | {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
            original_results['top1'] * 100,
            original_results['top5'] * 100,
            original_results['flops'] / 1e6,
            original_results['params'] / 1e6,
            original_results['infer_time']
        ))
        
        print("Quantized | {:.2f}%    | {:.2f}%    | {:.2f}     | {:.2f}      | {:.2f}".format(
            quantized_results['top1'] * 100,
            quantized_results['top5'] * 100,
            quantized_results['flops'] / 1e6,
            quantized_results['params'] / 1e6,
            quantized_results['infer_time']
        ))
        
        # Calculate model size
        import os
        import tempfile
        
        # Save original model
        with tempfile.NamedTemporaryFile() as f:
            torch.save(model.state_dict(), f.name)
            original_size = os.path.getsize(f.name) / (1024 * 1024)  # Size in MB
        
        # Save quantized model
        with tempfile.NamedTemporaryFile() as f:
            torch.save(model_quantized.state_dict(), f.name)
            quantized_size = os.path.getsize(f.name) / (1024 * 1024)  # Size in MB
        
        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Size reduction: {(1 - quantized_size / original_size) * 100:.2f}%")
    except Exception as e:
        print(f"Error in test_quantization: {e}")
        raise

def main():
    """
    Main function for testing.
    """
    try:
        # Get command line arguments
        args = get_args()
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Set GPU devices
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        
        # Test pruning criteria
        if args.test_criteria:
            test_pruning_criteria(args)
        
        # Test global pruning
        elif args.test_global:
            test_global_pruning(args)
        
        # Test iterative pruning
        elif args.test_iterative:
            test_iterative_pruning(args)
        
        # Test quantization
        elif args.test_quantization:
            test_quantization(args)
        
        # Regular testing
        else:
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
                model.load_state_dict(torch.load(load_path, map_location=device))
                print(f"Loaded model from {load_path}")
            else:
                print(f"Warning: Model checkpoint not found at {load_path}")
            
            # Set up model for testing
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(device)
            
            # Test model
            test_model(model, test_loader, args)
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    main()
