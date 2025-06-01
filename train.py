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
7. Dynamic device handling for better compatibility
8. Enhanced error handling
"""

def eval_epoch(net, testloader):
    """
    Evaluate a neural network model on a test dataset.
    
    Args:
        net: Neural network model
        testloader: Test data loader
        
    Returns:
        Tuple of (top1 accuracy, top5 accuracy, loss, inference time)
    """
    try:
        # Set device
        device = next(net.parameters()).device
        
        # Set to evaluation mode
        net.eval()
        
        # Initialize metrics
        test_loss = 0.0
        correct = 0
        correct_5 = 0
        total = 0
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Start time
        start_time = time.time()
        
        # Evaluate
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                
                # Update metrics
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Calculate top-5 accuracy
                _, pred_5 = outputs.topk(5, 1, True, True)
                pred_5 = pred_5.t()
                correct_5_tensor = pred_5.eq(targets.view(1, -1).expand_as(pred_5))
                correct_5 += correct_5_tensor.sum().item()
        
        # End time
        end_time = time.time()
        
        # Calculate metrics
        acc = correct / total
        acc_5 = correct_5 / total
        loss = test_loss / len(testloader)
        infer_time = (end_time - start_time) * 1000 / total  # ms per sample
        
        return acc, acc_5, loss, infer_time
    except Exception as e:
        print(f"Error in eval_epoch: {e}")
        raise

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
    try:
        # Set device
        device = next(net.parameters()).device
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Get optimizer and scheduler
        from tools.optim_sche import get_optim_sche
        optimizer, scheduler = get_optim_sche(lr, opt, net, trainloader.dataset.dataset.__class__.__name__.lower(), retrain)
        
        # OPTIMIZATION: Add mixed precision training
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Initialize best accuracy
        best_acc = 0.0
        
        # Create directories if they don't exist
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        
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
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_acc)
                else:
                    scheduler.step()
            
            # Print epoch results
            print(f"Epoch: {e+1}/{epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.3f}% | Time: {end_time-start_time:.2f}s")
            
            # Save checkpoint if it's the best model
            if test_acc > best_acc:
                best_acc = test_acc
                
                # Save model
                torch.save(net.state_dict(), os.path.join(save_path, "bestParam.pth"))
                print(f"Best model saved with accuracy: {best_acc*100:.3f}%")
            
            # Save regular checkpoint
            torch.save({
                'epoch': e+1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_acc': best_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            }, os.path.join(checkpoint_path, f"checkpoint_epoch_{e+1}.pth"))
        
        return net
    except Exception as e:
        print(f"Error in training: {e}")
        raise

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
    try:
        # Set device
        device = next(student_net.parameters()).device
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
        
        # Create directories if they don't exist
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        
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
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_acc)
                else:
                    scheduler.step()
            
            # Print epoch results
            print(f"Epoch: {e+1}/{epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.3f}% | Time: {end_time-start_time:.2f}s")
            
            # Save checkpoint if it's the best model
            if test_acc > best_acc:
                best_acc = test_acc
                
                # Save model
                torch.save(student_net.state_dict(), os.path.join(save_path, "bestParam.pth"))
                print(f"Best model saved with accuracy: {best_acc*100:.3f}%")
            
            # Save regular checkpoint
            torch.save({
                'epoch': e+1,
                'model_state_dict': student_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_acc': best_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            }, os.path.join(checkpoint_path, f"checkpoint_epoch_{e+1}.pth"))
        
        return student_net
    except Exception as e:
        print(f"Error in training_with_distillation: {e}")
        raise

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
    try:
        # Set device
        device = next(net.parameters()).device
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Get optimizer and scheduler
        from tools.optim_sche import get_optim_sche
        optimizer, scheduler = get_optim_sche(lr, opt, net, trainloader.dataset.dataset.__class__.__name__.lower(), retrain)
        
        # OPTIMIZATION: Add mixed precision training
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Initialize best accuracy
        best_acc = 0.0
        
        # Create directories if they don't exist
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        
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
            
            # Zero gradients at the beginning of each epoch
            optimizer.zero_grad()
            
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
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_acc)
                else:
                    scheduler.step()
            
            # Print epoch results
            print(f"Epoch: {e+1}/{epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.3f}% | Time: {end_time-start_time:.2f}s")
            
            # Save checkpoint if it's the best model
            if test_acc > best_acc:
                best_acc = test_acc
                
                # Save model
                torch.save(net.state_dict(), os.path.join(save_path, "bestParam.pth"))
                print(f"Best model saved with accuracy: {best_acc*100:.3f}%")
            
            # Save regular checkpoint
            torch.save({
                'epoch': e+1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_acc': best_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            }, os.path.join(checkpoint_path, f"checkpoint_epoch_{e+1}.pth"))
        
        return net
    except Exception as e:
        print(f"Error in training_with_gradient_accumulation: {e}")
        raise

# OPTIMIZATION: Added function to resume training from checkpoint
def resume_training(net, checkpoint_path, trainloader, testloader, opt="SGD", save_path="./checkpoint"):
    """
    Resume training from a checkpoint.
    
    Args:
        net: Neural network model
        checkpoint_path: Path to the checkpoint file
        trainloader: Training data loader
        testloader: Test data loader
        opt: Optimizer type
        save_path: Path to save the best model
        
    Returns:
        Trained model
    """
    try:
        # Set device
        device = next(net.parameters()).device
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        
        # Get optimizer and scheduler
        from tools.optim_sche import get_optim_sche
        optimizer, scheduler = get_optim_sche(0.001, opt, net, trainloader.dataset.dataset.__class__.__name__.lower(), True)
        
        # Load optimizer and scheduler states
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Get starting epoch and best accuracy
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        
        print(f"Resuming training from epoch {start_epoch} with best accuracy {best_acc*100:.3f}%")
        
        # Continue training
        remaining_epochs = 90 - start_epoch  # Assuming total epochs is 90
        if remaining_epochs <= 0:
            print("Training already completed")
            return net
        
        return training(net, remaining_epochs, trainloader, testloader, True, 0.001, opt, save_path, os.path.dirname(checkpoint_path))
    except Exception as e:
        print(f"Error in resume_training: {e}")
        raise

# OPTIMIZATION: Added function to load best model
def load_best_model(net, save_path):
    """
    Load the best model from a saved checkpoint.
    
    Args:
        net: Neural network model
        save_path: Path to the saved model
        
    Returns:
        Model with best parameters
    """
    try:
        # Set device
        device = next(net.parameters()).device
        
        # Load best model
        best_path = os.path.join(save_path, "bestParam.pth")
        if os.path.exists(best_path):
            net.load_state_dict(torch.load(best_path, map_location=device))
            print(f"Loaded best model from {best_path}")
        else:
            print(f"Best model not found at {best_path}")
        
        return net
    except Exception as e:
        print(f"Error in load_best_model: {e}")
        raise
