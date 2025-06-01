import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import datetime
import copy

"""
OPTIMIZATION HIGHLIGHTS:
1. Enhanced learning rate scheduling with cosine annealing and one-cycle policy
2. Knowledge distillation for better retraining after pruning
3. Mixed precision training for faster computation
4. Improved evaluation metrics and reporting
5. Better checkpoint management
6. Gradient accumulation for effective larger batch sizes
"""

def training(net, epoch, trainloader, testloader, retrain=False, lr=0.01, opt="SGD", save_path="./checkpoint", checkpoint_path="./checkpoint"):
    """
    Train a neural network model.
    
    Args:
        net: Neural network model
        epoch: Number of epochs to train
        trainloader: Training data loader
        testloader: Test data loader
        retrain: Whether this is retraining after pruning
        lr: Learning rate
        opt: Optimizer type
        save_path: Path to save the best model
        checkpoint_path: Path to save checkpoints
        
    Returns:
        Trained model
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Get optimizer and scheduler
    from tools.optim_sche import get_optim_sche
    optimizer, scheduler = get_optim_sche(lr, opt, net, trainloader.dataset.dataset.__class__.__name__.lower(), retrain)
    
    # OPTIMIZATION: Add mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Initialize best accuracy
    best_acc = 0.0
    
    # Training loop
    for e in range(epoch):
        # Set to training mode
        net.train()
        
        # Initialize metrics
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Start time
        start_time = time.time()
        
        # Iterate over batches
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # OPTIMIZATION: Use mixed precision training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                
                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f"Epoch: {e+1}/{epoch} | Batch: {batch_idx}/{len(trainloader)} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%")
        
        # End time
        end_time = time.time()
        
        # Calculate training metrics
        train_loss = train_loss / len(trainloader)
        train_acc = 100. * correct / total
        
        # Evaluate on test set
        test_acc, test_acc5, test_loss, infer_time = eval_epoch(net, testloader)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Print epoch results
        print(f"Epoch: {e+1}/{epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}% | Time: {end_time-start_time:.2f}s")
        
        # Save checkpoint if it's the best model
        if test_acc > best_acc:
            best_acc = test_acc
            
            # Save model
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            torch.save(net.state_dict(), os.path.join(save_path, "bestParam.pth"))
            print(f"Best model saved with accuracy: {best_acc:.3f}%")
        
        # Save regular checkpoint
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        
        torch.save({
            'epoch': e+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'best_acc': best_acc,
        }, os.path.join(checkpoint_path, f"checkpoint_epoch_{e+1}.pth"))
    
    return net

# OPTIMIZATION: Added knowledge distillation for better retraining
def training_with_distillation(student_net, teacher_net, epoch, trainloader, testloader, retrain=False, lr=0.01, opt="SGD", save_path="./checkpoint", checkpoint_path="./checkpoint", temp=4.0, alpha=0.5):
    """
    Train a student network using knowledge distillation from a teacher network.
    
    Args:
        student_net: Student neural network model (pruned model)
        teacher_net: Teacher neural network model (original model)
        epoch: Number of epochs to train
        trainloader: Training data loader
        testloader: Test data loader
        retrain: Whether this is retraining after pruning
        lr: Learning rate
        opt: Optimizer type
        save_path: Path to save the best model
        checkpoint_path: Path to save checkpoints
        temp: Temperature for softening the teacher's output
        alpha: Weight for distillation loss (alpha) vs hard loss (1-alpha)
        
    Returns:
        Trained student model
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_net = student_net.to(device)
    teacher_net = teacher_net.to(device)
    
    # Set teacher to evaluation mode
    teacher_net.eval()
    
    # Define loss functions
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    
    # Get optimizer and scheduler
    from tools.optim_sche import get_optim_sche
    optimizer, scheduler = get_optim_sche(lr, opt, student_net, trainloader.dataset.dataset.__class__.__name__.lower(), retrain)
    
    # OPTIMIZATION: Add mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Initialize best accuracy
    best_acc = 0.0
    
    # Training loop
    for e in range(epoch):
        # Set student to training mode
        student_net.train()
        
        # Initialize metrics
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Start time
        start_time = time.time()
        
        # Iterate over batches
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # OPTIMIZATION: Use mixed precision training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # Forward pass for student
                    student_outputs = student_net(inputs)
                    
                    # Forward pass for teacher (no gradient)
                    with torch.no_grad():
                        teacher_outputs = teacher_net(inputs)
                    
                    # Calculate soft targets
                    soft_targets = torch.nn.functional.softmax(teacher_outputs / temp, dim=1)
                    soft_student = torch.nn.functional.log_softmax(student_outputs / temp, dim=1)
                    
                    # Calculate losses
                    hard_loss = criterion_ce(student_outputs, targets)
                    soft_loss = criterion_kl(soft_student, soft_targets) * (temp * temp)
                    loss = alpha * soft_loss + (1 - alpha) * hard_loss
                
                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass for student
                student_outputs = student_net(inputs)
                
                # Forward pass for teacher (no gradient)
                with torch.no_grad():
                    teacher_outputs = teacher_net(inputs)
                
                # Calculate soft targets
                soft_targets = torch.nn.functional.softmax(teacher_outputs / temp, dim=1)
                soft_student = torch.nn.functional.log_softmax(student_outputs / temp, dim=1)
                
                # Calculate losses
                hard_loss = criterion_ce(student_outputs, targets)
                soft_loss = criterion_kl(soft_student, soft_targets) * (temp * temp)
                loss = alpha * soft_loss + (1 - alpha) * hard_loss
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f"Epoch: {e+1}/{epoch} | Batch: {batch_idx}/{len(trainloader)} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%")
        
        # End time
        end_time = time.time()
        
        # Calculate training metrics
        train_loss = train_loss / len(trainloader)
        train_acc = 100. * correct / total
        
        # Evaluate on test set
        test_acc, test_acc5, test_loss, infer_time = eval_epoch(student_net, testloader)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Print epoch results
        print(f"Epoch: {e+1}/{epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}% | Time: {end_time-start_time:.2f}s")
        
        # Save checkpoint if it's the best model
        if test_acc > best_acc:
            best_acc = test_acc
            
            # Save model
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            torch.save(student_net.state_dict(), os.path.join(save_path, "bestParam.pth"))
            print(f"Best model saved with accuracy: {best_acc:.3f}%")
        
        # Save regular checkpoint
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        
        torch.save({
            'epoch': e+1,
            'model_state_dict': student_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'best_acc': best_acc,
        }, os.path.join(checkpoint_path, f"checkpoint_epoch_{e+1}.pth"))
    
    return student_net

# OPTIMIZATION: Added gradient accumulation for effective larger batch sizes
def training_with_gradient_accumulation(net, epoch, trainloader, testloader, retrain=False, lr=0.01, opt="SGD", save_path="./checkpoint", checkpoint_path="./checkpoint", accumulation_steps=4):
    """
    Train a neural network model with gradient accumulation.
    
    Args:
        net: Neural network model
        epoch: Number of epochs to train
        trainloader: Training data loader
        testloader: Test data loader
        retrain: Whether this is retraining after pruning
        lr: Learning rate
        opt: Optimizer type
        save_path: Path to save the best model
        checkpoint_path: Path to save checkpoints
        accumulation_steps: Number of steps to accumulate gradients
        
    Returns:
        Trained model
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Get optimizer and scheduler
    from tools.optim_sche import get_optim_sche
    optimizer, scheduler = get_optim_sche(lr, opt, net, trainloader.dataset.dataset.__class__.__name__.lower(), retrain)
    
    # OPTIMIZATION: Add mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Initialize best accuracy
    best_acc = 0.0
    
    # Training loop
    for e in range(epoch):
        # Set to training mode
        net.train()
        
        # Initialize metrics
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Start time
        start_time = time.time()
        
        # Iterate over batches
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # OPTIMIZATION: Use mixed precision training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = net(inputs)
                    loss = criterion(outputs, targets) / accumulation_steps
                
                # Scale gradients
                scaler.scale(loss).backward()
                
                # Optimize every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(trainloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Forward pass
                outputs = net(inputs)
                loss = criterion(outputs, targets) / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Optimize every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(trainloader):
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Update metrics
            train_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f"Epoch: {e+1}/{epoch} | Batch: {batch_idx}/{len(trainloader)} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%")
        
        # End time
        end_time = time.time()
        
        # Calculate training metrics
        train_loss = train_loss / len(trainloader)
        train_acc = 100. * correct / total
        
        # Evaluate on test set
        test_acc, test_acc5, test_loss, infer_time = eval_epoch(net, testloader)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Print epoch results
        print(f"Epoch: {e+1}/{epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}% | Time: {end_time-start_time:.2f}s")
        
        # Save checkpoint if it's the best model
        if test_acc > best_acc:
            best_acc = test_acc
            
            # Save model
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            torch.save(net.state_dict(), os.path.join(save_path, "bestParam.pth"))
            print(f"Best model saved with accuracy: {best_acc:.3f}%")
        
        # Save regular checkpoint
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        
        torch.save({
            'epoch': e+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'best_acc': best_acc,
        }, os.path.join(checkpoint_path, f"checkpoint_epoch_{e+1}.pth"))
    
    return net

def eval_epoch(net, testloader):
    """
    Evaluate a neural network model on a test set.
    
    Args:
        net: Neural network model
        testloader: Test data loader
        
    Returns:
        Tuple of (top1 accuracy, top5 accuracy, loss, inference time)
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    # Set to evaluation mode
    net.eval()
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize metrics
    test_loss = 0.0
    correct = 0
    correct5 = 0
    total = 0
    
    # Start time
    start_time = time.time()
    
    # Disable gradient computation
    with torch.no_grad():
        # Iterate over batches
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            # Update metrics
            test_loss += loss.item()
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # OPTIMIZATION: Added Top-5 accuracy
            _, predicted5 = outputs.topk(5, 1, True, True)
            predicted5 = predicted5.t()
            correct5_batch = predicted5.eq(targets.view(1, -1).expand_as(predicted5))
            correct5 += correct5_batch[:5].reshape(-1).float().sum(0, keepdim=True).item()
    
    # End time
    end_time = time.time()
    
    # Calculate metrics
    test_loss = test_loss / len(testloader)
    test_acc = correct / total
    test_acc5 = correct5 / total
    infer_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    return test_acc, test_acc5, test_loss, infer_time

# OPTIMIZATION: Added function to resume training from checkpoint
def resume_training(net, checkpoint_path, trainloader, testloader, additional_epochs=10, lr=None):
    """
    Resume training from a checkpoint.
    
    Args:
        net: Neural network model
        checkpoint_path: Path to the checkpoint file
        trainloader: Training data loader
        testloader: Test data loader
        additional_epochs: Number of additional epochs to train
        lr: Learning rate (if None, use the one from checkpoint)
        
    Returns:
        Trained model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    # Get optimizer and scheduler
    from tools.optim_sche import get_optim_sche
    optimizer, scheduler = get_optim_sche(lr if lr is not None else 0.001, "SGD", net, trainloader.dataset.dataset.__class__.__name__.lower(), True)
    
    # Load optimizer and scheduler states
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Update learning rate if specified
    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Get starting epoch and best accuracy
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    
    print(f"Resuming training from epoch {start_epoch} with best accuracy {best_acc:.3f}%")
    
    # Set save paths
    save_path = os.path.dirname(checkpoint_path)
    checkpoint_path = os.path.join(save_path, "resumed")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    # Training loop
    for e in range(start_epoch, start_epoch + additional_epochs):
        # Set to training mode
        net.train()
        
        # Initialize metrics
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Start time
        start_time = time.time()
        
        # Iterate over batches
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net(inputs)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f"Epoch: {e+1}/{start_epoch + additional_epochs} | Batch: {batch_idx}/{len(trainloader)} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%")
        
        # End time
        end_time = time.time()
        
        # Calculate training metrics
        train_loss = train_loss / len(trainloader)
        train_acc = 100. * correct / total
        
        # Evaluate on test set
        test_acc, test_acc5, test_loss, infer_time = eval_epoch(net, testloader)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Print epoch results
        print(f"Epoch: {e+1}/{start_epoch + additional_epochs} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}% | Time: {end_time-start_time:.2f}s")
        
        # Save checkpoint if it's the best model
        if test_acc > best_acc:
            best_acc = test_acc
            
            # Save model
            torch.save(net.state_dict(), os.path.join(save_path, "bestParam.pth"))
            print(f"Best model saved with accuracy: {best_acc:.3f}%")
        
        # Save regular checkpoint
        torch.save({
            'epoch': e+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'best_acc': best_acc,
        }, os.path.join(checkpoint_path, f"checkpoint_epoch_{e+1}.pth"))
    
    return net

if __name__ == '__main__':
    # Example usage
    from netModels.VGG import MyVgg16
    from netModels.ResNet34 import MyResNet34
    from tools.get_data import get_train_loader, get_test_loader
    from tools.get_parameters import get_args
    
    args = get_args()
    
    # Load data
    train_loader = get_train_loader(args)
    test_loader = get_test_loader(args)
    
    # Load model
    if args.net == 'vgg16':
        model = MyVgg16(10)
    else:
        model = MyResNet34()
    
    # Set up checkpoint paths
    checkpoint_path = os.path.join("./checkpoint", args.net)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    # Train model
    if args.trainflag:
        # OPTIMIZATION: Use advanced training methods based on arguments
        if hasattr(args, 'distill') and args.distill:
            # Load original model for distillation
            original_model = copy.deepcopy(model)
            original_model.load_state_dict(torch.load(os.path.join(checkpoint_path, "train", "bestParam.pth")))
            
            # Train with distillation
            model = training_with_distillation(
                model, 
                original_model, 
                args.e, 
                train_loader, 
                test_loader, 
                False, 
                args.lr, 
                args.optim, 
                os.path.join(checkpoint_path, "train"), 
                os.path.join(checkpoint_path, "train"),
                args.temp,
                args.alpha
            )
        elif hasattr(args, 'gradient_accumulation') and args.gradient_accumulation:
            # Train with gradient accumulation
            model = training_with_gradient_accumulation(
                model, 
                args.e, 
                train_loader, 
                test_loader, 
                False, 
                args.lr, 
                args.optim, 
                os.path.join(checkpoint_path, "train"), 
                os.path.join(checkpoint_path, "train"),
                args.accumulation_steps
            )
        else:
            # Standard training
            model = training(
                model, 
                args.e, 
                train_loader, 
                test_loader, 
                False, 
                args.lr, 
                args.optim, 
                os.path.join(checkpoint_path, "train"), 
                os.path.join(checkpoint_path, "train")
            )
    
    # Retrain model
    if args.retrainflag:
        # OPTIMIZATION: Use advanced retraining methods based on arguments
        if hasattr(args, 'distill') and args.distill:
            # Load original model for distillation
            original_model = copy.deepcopy(model)
            original_model.load_state_dict(torch.load(os.path.join(checkpoint_path, "train", "bestParam.pth")))
            
            # Retrain with distillation
            model = training_with_distillation(
                model, 
                original_model, 
                args.retrainepoch, 
                train_loader, 
                test_loader, 
                True, 
                args.retrainlr, 
                args.optim, 
                os.path.join(checkpoint_path, "retrain"), 
                os.path.join(checkpoint_path, "retrain"),
                args.temp,
                args.alpha
            )
        else:
            # Standard retraining
            model = training(
                model, 
                args.retrainepoch, 
                train_loader, 
                test_loader, 
                True, 
                args.retrainlr, 
                args.optim, 
                os.path.join(checkpoint_path, "retrain"), 
                os.path.join(checkpoint_path, "retrain")
            )
