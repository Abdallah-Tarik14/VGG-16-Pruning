import torch.nn as nn
import torch.cuda
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import datetime

from netModels.VGG import MyVgg16
from netModels.ResNet34 import MyResNet34
from tools.get_data import get_test_loader
from tools.get_data import get_train_loader
from tools.get_parameters import get_args
from prune import prune_net, analyze_layer_sensitivity, global_pruning
from train import eval_epoch, training

"""
OPTIMIZATION HIGHLIGHTS:
1. Enhanced visualization with better plots and heatmaps
2. Layer sensitivity analysis for smarter pruning decisions
3. Comparison of different pruning criteria
4. Global pruning analysis
5. Improved documentation and code organization
6. Higher quality image output with better formatting
"""

CHECK_POINT_PATH = "./checkpoint"

colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c']
lines = ['-', '--', '-.']

vgg16_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10',
               'conv_11', 'conv_12', 'conv_13']
vgg16_total_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

first_layers = ['conv_2', 'conv_4', 'conv_6', 'conv_8', 'conv_10', 'conv_12', 'conv_14', 'conv_16', 'conv_18', 'conv_20',
               'conv_22', 'conv_24', 'conv_26', 'conv_28', 'conv_30', 'conv_32']
first_total_channels = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512]

shortcut_layers = ['downsample_1', 'downsample_2', 'downsample_3']
shortcut_total_channels = [128, 256, 512]

device_ids = [0, 1]

def get_net(net_name):
    """
    Get the neural network model.
    
    Args:
        net_name: Name of the network architecture
        
    Returns:
        Neural network model
    """
    net = None
    if net_name == 'vgg16':
        net = MyVgg16(10)
    elif net_name == 'resnet34':
        net = MyResNet34()
    else:
        print('We don\'t support this net.')
        exit(0)
    
    net = nn.DataParallel(net, device_ids=device_ids)
    net = net.cuda()
    return net

def get_list(net_name, shortcutflag):
    """
    Get the list of layers and their channel counts.
    
    Args:
        net_name: Name of the network architecture
        shortcutflag: Whether to include shortcut connections
        
    Returns:
        Tuple of (layer names, channel counts)
    """
    if net_name == 'vgg16':
        return vgg16_layers, vgg16_total_channels
    elif net_name == 'resnet34':
        if shortcutflag:
            return shortcut_layers, shortcut_total_channels
        else:
            return first_layers, first_total_channels
    else:
        print('We don\'t support this net.')
        exit(0)

def filter_retrain(net, name):
    """
    Retrain the network after pruning.
    
    Args:
        net: The neural network model
        name: Name for the checkpoint
        
    Returns:
        Retrained network
    """
    args = get_args()
    train_loader = get_train_loader(args)
    test_loader = get_test_loader(args)
    
    time = str(datetime.date.today() + datetime.timedelta(days=2))
    checkpoint_path = os.path.join(CHECK_POINT_PATH, args.net)
    retrain_checkpoint_path = os.path.join(checkpoint_path, 'retrain', time, name)
    retrain_most_recent_path = os.path.join(checkpoint_path, 'retrain', name)
    
    if not os.path.exists(retrain_checkpoint_path):
        os.makedirs(retrain_checkpoint_path)
    if not os.path.exists(retrain_most_recent_path):
        os.makedirs(retrain_most_recent_path)
        
    return training(net, args.retrainepoch, train_loader, test_loader, True, args.retrainlr, args.optim,
                  retrain_most_recent_path, retrain_checkpoint_path)

# OPTIMIZATION: Enhanced Filter Visualization
def sort_filter(args):
    """
    Sort and visualize filters by importance.
    
    Args:
        args: Command line arguments
    """
    # Get the desired net
    load_path = os.path.join(CHECK_POINT_PATH, args.net, "train", "bestParam.pth")
    net = get_net(args.net)
    new_net = get_net(args.net)
    
    if os.path.exists(load_path):
        net.load_state_dict(torch.load(load_path))
        
    # Make the figure
    shortcut = ""
    if args.shortcutflag:
        shortcut = "shortcut_"
        
    plt.figure(figsize=(12, 8))  # OPTIMIZATION: Larger figure size for better visualization
    
    conv_count = 0
    figure_count = 1
    
    # Store data for combined visualization
    all_filters_data = {}
    
    for layer in net.module.modules():
        if isinstance(layer, nn.Conv2d):
            # Exclude shortcut conv or residual conv
            if args.shortcutflag:
                if layer.kernel_size != (1, 1):
                    continue
            else:
                if layer.kernel_size == (1, 1):
                    continue
                    
            line_style = colors[conv_count % len(colors)] + lines[conv_count // len(colors) % len(lines)]
            weight = layer.weight.data.cpu().numpy()
            abs_sum_sorted = np.sort(np.sum(((np.abs(weight)).reshape(weight.shape[0], -1)), axis=1), axis=0)[::-1]
            norm_filter = abs_sum_sorted/abs_sum_sorted[0]
            
            # Store data for later
            all_filters_data[f'conv_{conv_count+1}'] = norm_filter
            
            conv_count += 1
            plt.plot(np.linspace(0, 100, norm_filter.shape[0]), norm_filter, line_style, 
                    label=shortcut + 'conv %d' % conv_count)
            
            # If there are too many convs in a figure, make a new one
            if conv_count % 17 == 0:
                plt.title("Data: %s" % args.dataset + ", Model: %s" % args.net, fontsize=14)
                plt.ylabel("Normalized Abs Sum of Filter Weight", fontsize=12)
                plt.xlabel("Filter Index / # Filters (%)", fontsize=12)
                plt.legend(loc='upper right')
                plt.xlim([0, 140])
                plt.grid(alpha=0.3)  # OPTIMIZATION: Add light grid for better readability
                plt.savefig(args.net + "_" + shortcut + str(figure_count) + "_" + "filters_ranked.png", 
                           bbox_inches='tight', dpi=300)  # OPTIMIZATION: Higher DPI for better quality
                plt.show()
                plt.figure(figsize=(12, 8))
                figure_count += 1
    
    plt.title("Data: %s" % args.dataset + ", Model: %s" % args.net, fontsize=14)
    plt.ylabel("Normalized Abs Sum of Filter Weight", fontsize=12)
    plt.xlabel("Filter Index / # Filters (%)", fontsize=12)
    plt.legend(loc='upper right')
    plt.xlim([0, 140])
    plt.grid(alpha=0.3)
    plt.savefig(args.net + "_" + shortcut + str(figure_count) + "_" + "filters_ranked.png", 
               bbox_inches='tight', dpi=300)
    plt.show()
    
    # OPTIMIZATION: Create a heatmap visualization of filter importance
    if all_filters_data:
        plt.figure(figsize=(14, 10))
        
        # Prepare data for heatmap
        max_len = max(len(data) for data in all_filters_data.values())
        heatmap_data = np.zeros((len(all_filters_data), max_len))
        
        for i, (layer_name, norm_filter) in enumerate(all_filters_data.items()):
            heatmap_data[i, :len(norm_filter)] = norm_filter
            
        plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
        plt.colorbar(label='Normalized Filter Importance')
        plt.title(f"Filter Importance Heatmap - {args.net}", fontsize=14)
        plt.ylabel("Convolutional Layer", fontsize=12)
        plt.xlabel("Filter Index (sorted by importance)", fontsize=12)
        plt.yticks(range(len(all_filters_data)), list(all_filters_data.keys()))
        plt.savefig(args.net + "_" + shortcut + "importance_heatmap.png", bbox_inches='tight', dpi=300)
        plt.show()

# OPTIMIZATION: Enhanced Pruning Analysis with Layer Sensitivity
def prune_analysis(args):
    """
    Analyze the effect of pruning on model performance.
    
    Args:
        args: Command line arguments
    """
    # Get the desired net layers, and channels
    layers, total_channels = get_list(args.net, args.shortcutflag)
    load_path = os.path.join(CHECK_POINT_PATH, args.net, "train", "bestParam.pth")
    
    # Get the args for eval
    test_loader = get_test_loader(args)
    train_loader = get_train_loader(args)  # OPTIMIZATION: Added for sensitivity analysis
    independentflag = False
    
    if args.shortcutflag:
        shortcut = "shortcut_"
    else:
        shortcut = ""
        
    # The parameter for prune
    max_prune_ratio = 0.90
    accuracy1 = {}
    accuracy5 = {}
    
    # OPTIMIZATION: Add sensitivity analysis
    if hasattr(args, 'analyze_sensitivity') and args.analyze_sensitivity:
        net = get_net(args.net)
        if os.path.exists(load_path):
            net.load_state_dict(torch.load(load_path))
            
        print("Analyzing layer sensitivity...")
        sensitivity = analyze_layer_sensitivity(net, train_loader, test_loader, layers, prune_ratio=0.3)
        
        # Visualize sensitivity
        plt.figure(figsize=(12, 6))
        layers_sorted = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
        layer_names = [layer for layer, _ in layers_sorted]
        sensitivities = [sens for _, sens in layers_sorted]
        
        plt.bar(range(len(layer_names)), sensitivities, color='skyblue')
        plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
        plt.title(f"Layer Sensitivity Analysis - {args.net}", fontsize=14)
        plt.ylabel("Accuracy Drop (higher = more sensitive)", fontsize=12)
        plt.xlabel("Layer", fontsize=12)
        plt.tight_layout()
        plt.savefig(args.net + "_" + shortcut + "sensitivity_analysis.png", bbox_inches='tight', dpi=300)
        plt.show()
        
        print("Layer sensitivity analysis:")
        for layer, sens in sorted(sensitivity.items(), key=lambda x: x[1], reverse=True):
            print(f"Layer {layer}: {sens:.4f} accuracy drop")
    
    # For all layers in the net
    for conv, channels in zip(layers, total_channels):
        new_net = get_net(args.net)
        if os.path.exists(load_path):
            new_net.load_state_dict(torch.load(load_path))
            
        print("evaluating")
        top1, top5, loss, infer_time = eval_epoch(new_net, test_loader)
        print("Eval before pruning" + ": Loss:{:.3f}\t acc1:{:.3%}\t acc5:{:.3%}\t Inference time:{:.3}\n"
             .format(loss, top1, top5, infer_time / len(test_loader.dataset)))
             
        accuracy1[conv] = [top1]
        accuracy5[conv] = [top5]
        
        prune_layers = [conv]
        prune_channels = np.linspace(0, int(channels * max_prune_ratio), 10, dtype=int)
        prune_channels = (prune_channels[1:] - prune_channels[:-1]).tolist()
        
        # For each layer
        for index, prune_channel in enumerate(prune_channels):
            # Prune
            new_net = prune_net(new_net, independentflag, prune_layers, [prune_channel], args.net, args.shortcutflag)
            
            # Eval
            print("evaluating")
            top1, top5, loss, infer_time = eval_epoch(new_net, test_loader)
            print("Eval after pruning " + conv, index, ":\t Loss:{:.3f}\t acc1:{:.3%}\t acc5:{:.3%}\t "
                 "Inference time:{:.3}\n".format(loss, top1, top5, infer_time / len(test_loader.dataset)))
                 
            accuracy1[conv].append(top1)
            accuracy5[conv].append(top5)
            
    with open('top1', 'w') as fout:
        json.dump(accuracy1, fout)
    with open('top5', 'w') as fout:
        json.dump(accuracy5, fout)
        
    # OPTIMIZATION: Enhanced visualization
    plt.figure(figsize=(12, 8))
    for index, (conv, acc1) in enumerate(accuracy1.items()):
        line_style = colors[index % len(colors)] + lines[index // len(colors) % len(lines)] + 'o'
        xs = np.linspace(0, 90, len(acc1))
        plt.plot(xs, acc1, line_style, label=conv+' '+str(total_channels[index]))
        
    plt.title("Data: %s" % args.dataset + ", Model: %s" % args.net + ", pruned smallest filters (greedy)", fontsize=14)
    plt.ylabel("Accuracy (top1)", fontsize=12)
    plt.xlabel("Filters Pruned Away (%)", fontsize=12)
    plt.legend(loc='upper right')
    plt.xlim([0, 100])
    plt.grid(alpha=0.3)
    plt.savefig(shortcut+args.dataset + "_pruned_top1.png", bbox_inches='tight', dpi=300)
    plt.show()
    
    plt.figure(figsize=(12, 8))
    for index, (conv, acc5) in enumerate(accuracy5.items()):
        line_style = colors[index % len(colors)] + lines[index // len(colors) % len(lines)] + 'o'
        xs = np.linspace(0, 90, len(acc5))
        plt.plot(xs, acc5, line_style, label=conv + ' ' + str(total_channels[index]))
        
    plt.title("Data: %s" % args.dataset + ", Model: %s" % args.net + ", pruned smallest filters (greedy)", fontsize=14)
    plt.ylabel("Accuracy (top5)", fontsize=12)
    plt.xlabel("Filters Pruned Away (%)", fontsize=12)
    plt.legend(loc='upper right')
    plt.xlim([0, 100])
    plt.grid(alpha=0.3)
    plt.savefig(shortcut+args.dataset + "_pruned_top5.png", bbox_inches='tight', dpi=300)
    plt.show()
    
    # OPTIMIZATION: Add comparison of different pruning criteria
    if hasattr(args, 'compare_criteria') and args.compare_criteria:
        criteria = ['l1', 'l2', 'l1_l2']
        plt.figure(figsize=(12, 8))
        
        # Select a representative layer for comparison
        test_layer = layers[len(layers)//2]
        test_channels = total_channels[len(total_channels)//2]
        
        for criterion_idx, criterion in enumerate(criteria):
            criterion_acc = []
            
            # Get fresh network
            new_net = get_net(args.net)
            if os.path.exists(load_path):
                new_net.load_state_dict(torch.load(load_path))
                
            # Baseline accuracy
            top1, _, _, _ = eval_epoch(new_net, test_loader)
            criterion_acc.append(top1)
            
            # Test different pruning amounts
            prune_amounts = np.linspace(0.1, 0.7, 4)
            for prune_amount in prune_amounts:
                prune_channel = int(test_channels * prune_amount)
                
                # Get fresh network
                new_net = get_net(args.net)
                if os.path.exists(load_path):
                    new_net.load_state_dict(torch.load(load_path))
                    
                # Prune with specific criterion
                new_net = prune_net(new_net, independentflag, [test_layer], [prune_channel], 
                                   args.net, args.shortcutflag, criterion=criterion)
                
                # Evaluate
                top1, _, _, _ = eval_epoch(new_net, test_loader)
                criterion_acc.append(top1)
                
            # Plot results
            line_style = colors[criterion_idx] + lines[criterion_idx % len(lines)] + 'o'
            xs = np.concatenate(([0], prune_amounts * 100))
            plt.plot(xs, criterion_acc, line_style, label=f'Criterion: {criterion}')
            
        plt.title(f"Comparison of Pruning Criteria on {test_layer}", fontsize=14)
        plt.ylabel("Accuracy (top1)", fontsize=12)
        plt.xlabel("Filters Pruned Away (%)", fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
        plt.savefig(args.net + "_" + shortcut + "criteria_comparison.png", bbox_inches='tight', dpi=300)
        plt.show()

def prune_retrain_analysis(args):
    """
    Analyze the effect of pruning and retraining on model performance.
    
    Args:
        args: Command line arguments
    """
    load_path = os.path.join(CHECK_POINT_PATH, args.net, "train", "bestParam.pth")
    test_loader = get_test_loader(args)
    
    if args.shortcutflag:
        shortcut = "shortcut_"
    else:
        shortcut = ""
        
    new_net = get_net(args.net)
    if os.path.exists(load_path):
        new_net.load_state_dict(torch.load(load_path))
        
    top1_org, top5_org, loss, infer_time = eval_epoch(new_net, test_loader)
    
    layers, total_channels = get_list(args.net, args.shortcutflag)
    independentflag = False
    max_prune_ratio = 0.90
    min_prune_ratio = 0.20
    
    accuracy1 = {}
    accuracy5 = {}
    
    # For all layers
    for conv, channels in zip(layers, total_channels):
        accuracy1[conv] = [top1_org]
        accuracy5[conv] = [top5_org]
        
        prune_layers = [conv]
        prune_channels = np.linspace(int(channels * min_prune_ratio), int(channels * max_prune_ratio), 8, dtype=int).tolist()
        
        # For each layer
        for index, prune_channel in enumerate(prune_channels):
            # Get net and prune
            new_net = get_net(args.net)
            if os.path.exists(load_path):
                new_net.load_state_dict(torch.load(load_path))
                
            new_net = prune_net(new_net, independentflag, prune_layers, [prune_channel], args.net, args.shortcutflag)
            
            # Retrain
            new_net = filter_retrain(new_net, conv + ':pruned %d' % prune_channel)
            
            # Eval
            top1, top5, loss, infer_time = eval_epoch(new_net, test_loader)
            print("Eval after pruning" + conv, index, ":\t Loss:{:.3f}\t acc1:{:.3%}\t acc5:{:.3%}\t "
                 "Inference time:{:.3}\n".format(loss, top1, top5, infer_time / len(test_loader.dataset)))
                 
            accuracy1[conv].append(top1)
            accuracy5[conv].append(top5)
            
    with open('top1_backup', 'w') as fout:
        json.dump(accuracy1, fout)
    with open('top5_backup', 'w') as fout:
        json.dump(accuracy5, fout)
        
    # OPTIMIZATION: Enhanced visualization
    plt.figure(figsize=(12, 8))
    for index, (conv, acc1) in enumerate(accuracy1.items()):
        line_style = colors[index % len(colors)] + lines[index // len(colors) % len(lines)] + 'o'
        xs = np.linspace(0, 90, len(acc1))
        plt.plot(xs, acc1, line_style, label=conv+' '+str(total_channels[index]))
        
    plt.title("Data: %s" % args.dataset + ", Model: %s" % args.net + 
             ", pruned smallest filters and retrained", fontsize=14)
    plt.ylabel("Accuracy (top1)", fontsize=12)
    plt.xlabel("Filters Pruned Away (%)", fontsize=12)
    plt.legend(loc='upper right')
    plt.xlim([0, 100])
    plt.grid(alpha=0.3)
    plt.savefig(shortcut+args.dataset + "_pruned_retrained_top1.png", bbox_inches='tight', dpi=300)
    plt.show()
    
    plt.figure(figsize=(12, 8))
    for index, (conv, acc5) in enumerate(accuracy5.items()):
        line_style = colors[index % len(colors)] + lines[index // len(colors) % len(lines)] + 'o'
        xs = np.linspace(0, 90, len(acc5))
        plt.plot(xs, acc5, line_style, label=conv + ' ' + str(total_channels[index]))
        
    plt.title("Data: %s" % args.dataset + ", Model: %s" % args.net + 
             ", pruned smallest filters and retrained", fontsize=14)
    plt.ylabel("Accuracy (top5)", fontsize=12)
    plt.xlabel("Filters Pruned Away (%)", fontsize=12)
    plt.legend(loc='upper right')
    plt.xlim([0, 100])
    plt.grid(alpha=0.3)
    plt.savefig(shortcut+args.dataset + "_pruned_retrained_top5.png", bbox_inches='tight', dpi=300)
    plt.show()

# OPTIMIZATION: Global Pruning Analysis
def global_pruning_analysis(args):
    """
    Analyze the effect of global pruning on model performance.
    
    Args:
        args: Command line arguments
    """
    load_path = os.path.join(CHECK_POINT_PATH, args.net, "train", "bestParam.pth")
    test_loader = get_test_loader(args)
    train_loader = get_train_loader(args)
    
    if args.shortcutflag:
        shortcut = "shortcut_"
    else:
        shortcut = ""
        
    # Get original network and evaluate
    net = get_net(args.net)
    if os.path.exists(load_path):
        net.load_state_dict(torch.load(load_path))
        
    top1_org, top5_org, loss_org, infer_time_org = eval_epoch(net, test_loader)
    print(f"Original model: Top1 {top1_org:.2%}, Top5 {top5_org:.2%}, Time {infer_time_org:.3f}ms")
    
    # Test different global pruning ratios
    prune_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    top1_results = [top1_org]
    top5_results = [top5_org]
    infer_times = [infer_time_org]
    
    for ratio in prune_ratios:
        # Get fresh network
        new_net = get_net(args.net)
        if os.path.exists(load_path):
            new_net.load_state_dict(torch.load(load_path))
            
        # Apply global pruning
        pruned_net = global_pruning(new_net, ratio, args.net, args.shortcutflag)
        
        # Evaluate
        top1, top5, loss, infer_time = eval_epoch(pruned_net, test_loader)
        print(f"Global pruning {ratio:.1f}: Top1 {top1:.2%}, Top5 {top5:.2%}, Time {infer_time:.3f}ms")
        
        top1_results.append(top1)
        top5_results.append(top5)
        infer_times.append(infer_time)
        
    # Visualize results
    plt.figure(figsize=(12, 8))
    xs = [0] + prune_ratios
    
    plt.subplot(2, 1, 1)
    plt.plot(xs, top1_results, 'ro-', label='Top-1 Accuracy')
    plt.plot(xs, top5_results, 'bo-', label='Top-5 Accuracy')
    plt.title(f"Global Pruning Analysis - {args.net}", fontsize=14)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(xs, infer_times, 'go-', label='Inference Time')
    plt.xlabel("Global Pruning Ratio", fontsize=12)
    plt.ylabel("Inference Time (ms)", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.net + "_" + shortcut + "global_pruning_analysis.png", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    args = get_args()
    
    if args.sortflag:
        sort_filter(args)
    elif args.pruneflag:
        prune_analysis(args)
    elif args.retrainflag:
        prune_retrain_analysis(args)
    elif hasattr(args, 'global_analysis') and args.global_analysis:
        global_pruning_analysis(args)
    else:
        print("Please specify the flag.")
