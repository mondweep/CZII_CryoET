import os
import torch
from src.training.train_model import CryoET3DCNN, train_model

def continue_training():
    # Use the latest checkpoint (epoch 30)
    checkpoint_path = 'models/checkpoint_epoch_30_20241216_012051.pth'
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    
    # Initialize model and load state
    model = CryoET3DCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get the starting epoch
    start_epoch = checkpoint['epoch'] + 1
    
    # Initialize metrics history if not present in checkpoint
    metrics_history = checkpoint.get('metrics_history', {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_precision': [], 'train_recall': [], 'train_f1': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    })
    
    try:
        # Continue training from where we left off
        print(f"Starting training from epoch {start_epoch}")
        model, updated_metrics = train_model(
            model=model,
            optimizer=optimizer,
            start_epoch=start_epoch,
            num_epochs=50,
            metrics_history=metrics_history
        )
        return model, updated_metrics
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    try:
        model, metrics = continue_training()
        if model is not None:
            print("Training completed successfully")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc() 