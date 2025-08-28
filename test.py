
import torch

from models.party_a import PartyAModel
from models.party_b import ShadowModel, PrivateModel, LocalHead
from models.top import TopModel
from utils.data_loader import load_bcw, create_dataloader

# Hyperparameters
BATCH_SIZE = 32
NUM_CLASSES = 2
EMBEDDING_DIM = 8

# Feature dimensions for BCW dataset
PARTY_A_FEATURES = 15
PARTY_B_PUBLIC_FEATURES = 8
PARTY_B_PRIVATE_FEATURES = 7

def test():
    # 1. Load Test Data
    _, (X_a_test, X_b_test, y_test), public_indices, private_indices = load_bcw()
    test_loader = create_dataloader(X_a_test, X_b_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load Models
    party_a_model = PartyAModel(input_dim=PARTY_A_FEATURES, output_dim=EMBEDDING_DIM)
    shadow_model = ShadowModel(input_dim=PARTY_B_PUBLIC_FEATURES, output_dim=EMBEDDING_DIM)
    private_model = PrivateModel(input_dim=PARTY_B_PRIVATE_FEATURES, output_dim=EMBEDDING_DIM)
    local_head = LocalHead(input_dim=EMBEDDING_DIM * 2, output_dim=NUM_CLASSES)
    top_model = TopModel(input_dim=EMBEDDING_DIM * 2, output_dim=NUM_CLASSES)

    try:
        party_a_model.load_state_dict(torch.load('party_a_model.pth'))
        shadow_model.load_state_dict(torch.load('shadow_model.pth'))
        private_model.load_state_dict(torch.load('private_model.pth'))
        local_head.load_state_dict(torch.load('local_head.pth'))
        top_model.load_state_dict(torch.load('top_model.pth'))
    except FileNotFoundError:
        print("Error: Model files not found. Please run train.py first.")
        return

    party_a_model.eval()
    shadow_model.eval()
    private_model.eval()
    local_head.eval()
    top_model.eval()

    # 3. Evaluation
    correct_global = 0
    correct_local = 0
    total = 0
    with torch.no_grad():
        for batch_X_a, batch_X_b, batch_y in test_loader:
            # Split Party B's features
            batch_X_b_public = batch_X_b[:, public_indices]
            batch_X_b_private = batch_X_b[:, private_indices]

            # Forward pass
            E_a = party_a_model(batch_X_a)
            E_shadow = shadow_model(batch_X_b_public)
            E_private = private_model(batch_X_b_private)

            # Global prediction
            E_fused_top = torch.cat((E_a, E_shadow), dim=1)
            global_prediction = top_model(E_fused_top)
            _, predicted_global = torch.max(global_prediction.data, 1)
            
            # Local prediction
            E_fused_local = torch.cat((E_shadow, E_private), dim=1)
            local_prediction = local_head(E_fused_local)
            _, predicted_local = torch.max(local_prediction.data, 1)

            total += batch_y.size(0)
            correct_global += (predicted_global == batch_y).sum().item()
            correct_local += (predicted_local == batch_y).sum().item()

    print(f'Accuracy of the global model on the test set: {100 * correct_global / total:.2f} %')
    print(f'Accuracy of the defender's local model on the test set: {100 * correct_local / total:.2f} %')

if __name__ == '__main__':
    test()
