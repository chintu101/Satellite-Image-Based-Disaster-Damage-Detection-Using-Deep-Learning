import torch
from model import SiameseUNet

# Load the old file permissively
checkpoint = torch.load('model.pth', map_location='cpu', weights_only=False)

# Resave as a proper state_dict in modern format
if isinstance(checkpoint, dict):
    state_dict = checkpoint
else:
    state_dict = checkpoint.state_dict()

# Verify it loads into the architecture cleanly
model = SiameseUNet()
model.load_state_dict(state_dict)

# Save in modern format
torch.save(model.state_dict(), 'model.pth')
print("✅ model.pth re-saved successfully in modern PyTorch format!")