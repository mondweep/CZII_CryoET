import torch
from ..core.model import CryoET3DCNN
from ..core.data_loader import CryoETDataset

def predict(model_path, data):
    """
    Make predictions using the trained model
    """
    # Load model
    model = CryoET3DCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Process data and make predictions
    with torch.no_grad():
        outputs = model(data)
        _, predicted = outputs.max(1)
    
    return predicted 