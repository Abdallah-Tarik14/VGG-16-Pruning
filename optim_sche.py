import torch
import torch.optim as optim

"""
OPTIMIZATION HIGHLIGHTS:
1. Enhanced learning rate scheduling with cosine annealing and one-cycle policy
2. Support for more optimizers including Adam
3. Weight decay separation for better training dynamics
4. Advanced scheduler creation function
5. Better documentation and organization
"""

# Learning rate schedule milestones
CIFAR10_MILESTONES = [60, 120, 160]
ImageNet_MILESTONES = [30, 60]

def get_optim_sche(lr, opt, net, dataset, retrain=False):
    """
    Get optimizer and learning rate scheduler.
    
    Args:
        lr: Learning rate
        opt: Optimizer type
        net: Neural network model
        dataset: Dataset name
        retrain: Whether this is for retraining after pruning
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Create optimizer
    if opt == "SGD":
        optimizer = optim.SGD(
            net.parameters(), 
            lr=lr, 
            momentum=0.9, 
            weight_decay=1e-4
        )
    elif opt == "Adam":  # OPTIMIZATION: Added Adam optimizer option
        optimizer = optim.Adam(
            net.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
    else:  # Default to RMSprop
        optimizer = optim.RMSprop(
            net.parameters(), 
            lr=lr, 
            momentum=0.9, 
            weight_decay=5e-4
        )
    
    # Create scheduler
    if retrain:
        # OPTIMIZATION: Added learning rate scheduler for retraining
        if hasattr(net, 'args') and hasattr(net.args, 'lr_scheduler') and net.args.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=10,  # 10 epochs for cosine annealing
                eta_min=lr/10
            )
        else:
            scheduler = None
    else:
        # Set up milestones and gamma based on dataset
        MILESTONES = []
        gamma = 0
        
        if dataset == 'cifar10':
            MILESTONES = CIFAR10_MILESTONES
            gamma = 0.2
        elif dataset == 'imagenet':
            MILESTONES = ImageNet_MILESTONES
            gamma = 0.1
            
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=MILESTONES, 
            gamma=gamma, 
            last_epoch=-1
        )
    
    return optimizer, scheduler

# OPTIMIZATION: Added function for advanced learning rate schedulers
def get_advanced_scheduler(optimizer, args):
    """
    Get advanced learning rate scheduler based on arguments.
    
    Args:
        optimizer: The optimizer
        args: Command line arguments
        
    Returns:
        Learning rate scheduler
    """
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step,
            gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.e,
            eta_min=args.lr * 0.01
        )
    elif args.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Based on accuracy
            factor=0.1,
            patience=5,
            verbose=True
        )
    elif args.lr_scheduler == "onecycle":
        # One-cycle policy for super-convergence
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr * 10,
            total_steps=args.e,
            pct_start=0.3,
            anneal_strategy='cos'
        )
    else:
        # Default to MultiStepLR
        if args.dataset == 'cifar10':
            milestones = CIFAR10_MILESTONES
            gamma = 0.2
        else:  # imagenet
            milestones = ImageNet_MILESTONES
            gamma = 0.1
            
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )

# OPTIMIZATION: Added function for creating optimizer with weight decay separation
def create_optimizer_with_weight_decay_separation(model, lr, opt="SGD"):
    """
    Create optimizer with separate weight decay for normalization layers.
    This is a best practice that can improve performance.
    
    Args:
        model: The neural network model
        lr: Learning rate
        opt: Optimizer type
        
    Returns:
        Optimizer
    """
    # Separate parameters into those with and without weight decay
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Skip BatchNorm parameters and biases for weight decay
        if len(param.shape) == 1 or name.endswith(".bias") or "bn" in name or "batch_norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    
    # Set up parameter groups
    parameters = [
        {'params': decay, 'weight_decay': 1e-4},
        {'params': no_decay, 'weight_decay': 0}
    ]
    
    # Create optimizer
    if opt == "SGD":
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    elif opt == "Adam":
        return optim.Adam(parameters, lr=lr)
    else:  # RMSprop
        return optim.RMSprop(parameters, lr=lr, momentum=0.9, weight_decay=5e-4)
