import os
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from ..core.data_loader import CryoETDataset
from ..core.model import CryoET3DCNN

def continue_training():
    # Load environment variables
    load_dotenv()
    
    print("Loading checkpoint...")
    checkpoint = torch.load('model_checkpoint_epoch_40.pth')
    
    # Initialize model and load state
    model = CryoET3DCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize dataset
    dataset = CryoETDataset(
        raw_path=os.getenv('DATASET_PATH'),
        annotated_path=os.getenv('ANNOTATED_DATASET_PATH')
    )
    
    # Get training data
    print("Preparing data...")
    X, y = dataset.create_training_patches()
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension
    y = torch.LongTensor(y)
    
    # Create data loaders
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 1, 1, 2, 2, 1]).to(device))
    
    # Continue training
    print("\nResuming training from epoch 41...")
    num_epochs = 50
    start_epoch = 41
    
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_loss = 0
        batch_count = 0
        
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("Training batches: ", end='', flush=True)
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            if batch_count % 5 == 0:
                print("🔄", end='', flush=True)
        
        avg_train_loss = train_loss / batch_count
        print(f"\nAverage training loss: {avg_train_loss:.4f}")
        
        # Save checkpoint every epoch
        checkpoint_path = f'model_checkpoint_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    return model

if __name__ == "__main__":
    try:
        model = continue_training()
        print("\nTraining completed successfully!")
        # Save final model
        torch.save(model.state_dict(), 'cryo_et_model_final.pth')
        print("Saved final model as: cryo_et_model_final.pth")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc() 