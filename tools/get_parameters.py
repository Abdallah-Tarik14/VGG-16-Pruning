import argparse

"""
OPTIMIZATION HIGHLIGHTS:
1. Enhanced command-line interface with more options
2. Added learning rate scheduler options
3. Added knowledge distillation parameters
4. Added global and iterative pruning parameters
5. Added analysis and testing parameters
6. Better documentation and organization
"""

def get_args():
    """
    Parse command line arguments for the pruning filters implementation.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Implementation of Pruning Filters for Efficient ConvNets")
    
    # Basic parameters
    parser.add_argument("-net", default='resnet34', help='Network architecture (resnet34 or vgg16)')
    parser.add_argument("-dataset", default='imagenet', help='Dataset to use (imagenet or cifar10)')
    parser.add_argument("-b", default=256, type=int, help='Batch size')
    
    # Training parameters
    parser.add_argument("-lr", default=0.1, help='Initial learning rate', type=float)
    parser.add_argument("-e", default=90, help='Number of epochs', type=int)
    parser.add_argument("-optim", default="SGD", help='Optimizer (SGD, Adam, RMSprop)')
    parser.add_argument("-gpu", default="0,1", help='GPU devices to use', type=str)
    parser.add_argument("-trainflag", action='store_true', help='Whether to train the model', default=False)
    
    # OPTIMIZATION: Added learning rate scheduler options
    parser.add_argument("-lr_scheduler", default="step", help='Learning rate scheduler (step, cosine, plateau)', type=str)
    parser.add_argument("-lr_step", default=30, help='Step size for StepLR scheduler', type=int)
    parser.add_argument("-lr_gamma", default=0.1, help='Gamma for StepLR scheduler', type=float)
    
    # Retraining parameters
    parser.add_argument("-retrainflag", action='store_true', help='Whether to retrain after pruning', default=False)
    parser.add_argument("-retrainepoch", default=20, help='Number of epochs for retraining', type=int)
    parser.add_argument("-retrainlr", default=0.001, help='Learning rate for retraining', type=float)
    
    # OPTIMIZATION: Added knowledge distillation parameters
    parser.add_argument("-distill", action='store_true', help='Whether to use knowledge distillation during retraining', default=False)
    parser.add_argument("-temp", default=4.0, help='Temperature for knowledge distillation', type=float)
    parser.add_argument("-alpha", default=0.5, help='Weight for distillation loss (alpha) vs hard loss (1-alpha)', type=float)
    
    # Pruning parameters
    parser.add_argument("-pruneflag", action='store_true', help='Whether to prune the model', default=False)
    parser.add_argument("-sortflag", action='store_true', help='Whether to sort filters by importance', default=False)
    parser.add_argument("-independentflag", action='store_true', help='Whether to use independent pruning strategy', default=False)
    parser.add_argument("-shortcutflag", action='store_true', help='Whether to prune shortcut connections', default=True)
    parser.add_argument("-prune_channels", nargs='+', type=int, help='Number of channels to prune for each layer')
    parser.add_argument("-prune_layers", nargs='+', help='Layers to prune')
    
    # OPTIMIZATION: Added new pruning parameters
    parser.add_argument("-criterion", default="l1", help='Pruning criterion (l1, l2, l1_l2)', type=str)
    parser.add_argument("-global_pruning", action='store_true', help='Whether to use global pruning strategy', default=False)
    parser.add_argument("-global_ratio", default=0.1, help='Global pruning ratio', type=float)
    parser.add_argument("-iterative_pruning", action='store_true', help='Whether to use iterative pruning', default=False)
    parser.add_argument("-prune_iterations", default=3, help='Number of pruning iterations for iterative pruning', type=int)
    parser.add_argument("-intermediate_epochs", default=5, help='Number of epochs between pruning iterations', type=int)
    
    # OPTIMIZATION: Added analysis parameters
    parser.add_argument("-analyze_sensitivity", action='store_true', help='Whether to analyze layer sensitivity', default=False)
    parser.add_argument("-compare_criteria", action='store_true', help='Whether to compare different pruning criteria', default=False)
    parser.add_argument("-global_analysis", action='store_true', help='Whether to analyze global pruning', default=False)
    
    # OPTIMIZATION: Added quantization parameters
    parser.add_argument("-quantize", action='store_true', help='Whether to quantize the model after pruning', default=False)
    parser.add_argument("-qat", action='store_true', help='Whether to use quantization-aware training', default=False)
    parser.add_argument("-quantize_backend", default="fbgemm", help='Backend for quantization (fbgemm or qnnpack)', type=str)
    
    # OPTIMIZATION: Added test parameters
    parser.add_argument("-test_criteria", action='store_true', help='Whether to test different pruning criteria', default=False)
    parser.add_argument("-test_iterative", action='store_true', help='Whether to test iterative pruning', default=False)
    parser.add_argument("-test_sensitivity", action='store_true', help='Whether to test layer sensitivity analysis', default=False)
    parser.add_argument("-test_distillation", action='store_true', help='Whether to test knowledge distillation', default=False)
    parser.add_argument("-test_global", action='store_true', help='Whether to test global pruning', default=False)
    parser.add_argument("-test_quantization", action='store_true', help='Whether to test quantization', default=False)
    
    # OPTIMIZATION: Added gradient accumulation parameters
    parser.add_argument("-gradient_accumulation", action='store_true', help='Whether to use gradient accumulation', default=False)
    parser.add_argument("-accumulation_steps", default=4, help='Number of steps to accumulate gradients', type=int)
    
    # OPTIMIZATION: Added mixed precision parameters
    parser.add_argument("-mixed_precision", action='store_true', help='Whether to use mixed precision training', default=False)
    
    args = parser.parse_args()
    return args
