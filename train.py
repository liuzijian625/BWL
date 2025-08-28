
import torch
import torch.nn as nn
import torch.optim as optim

from models.party_a import PartyAModel
from models.party_b import ShadowModel, PrivateModel, LocalHead
from models.top import TopModel
from losses.boundary_wandering_loss import boundary_wandering_loss
from utils.data_loader import load_bcw, create_dataloader

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
ALPHA = 0.5
NUM_CLASSES = 2
EMBEDDING_DIM = 8

# Feature dimensions for BCW dataset
PARTY_A_FEATURES = 15
PARTY_B_PUBLIC_FEATURES = 8
PARTY_B_PRIVATE_FEATURES = 7

def train():
    # 1. Data Loading
    (X_a_train, X_b_train, y_train), _, public_indices, private_indices = load_bcw()
    train_loader = create_dataloader(X_a_train, X_b_train, y_train, batch_size=BATCH_SIZE)

    # 2. Model Initialization
    party_a_model = PartyAModel(input_dim=PARTY_A_FEATURES, output_dim=EMBEDDING_DIM)
    shadow_model = ShadowModel(input_dim=PARTY_B_PUBLIC_FEATURES, output_dim=EMBEDDING_DIM)
    private_model = PrivateModel(input_dim=PARTY_B_PRIVATE_FEATURES, output_dim=EMBEDDING_DIM)
    local_head = LocalHead(input_dim=EMBEDDING_DIM * 2, output_dim=NUM_CLASSES)
    top_model = TopModel(input_dim=EMBEDDING_DIM * 2, output_dim=NUM_CLASSES)

    # 3. Optimizers
    optimizer_A = optim.Adam(party_a_model.parameters(), lr=LEARNING_RATE)
    optimizer_shadow = optim.Adam(shadow_model.parameters(), lr=LEARNING_RATE)
    optimizer_private = optim.Adam(private_model.parameters(), lr=LEARNING_RATE)
    optimizer_local_head = optim.Adam(local_head.parameters(), lr=LEARNING_RATE)
    optimizer_top = optim.Adam(top_model.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    for epoch in range(EPOCHS):
        for i, (batch_X_a, batch_X_b, batch_y) in enumerate(train_loader):
            optimizer_A.zero_grad()
            optimizer_shadow.zero_grad()
            optimizer_private.zero_grad()
            optimizer_local_head.zero_grad()
            optimizer_top.zero_grad()

            # Split Party B's features based on public/private indices
            batch_X_b_public = batch_X_b[:, public_indices]
            batch_X_b_private = batch_X_b[:, private_indices]

            # --- PHASE 1: FORWARD PASS ---
            E_a = party_a_model(batch_X_a)
            E_shadow = shadow_model(batch_X_b_public)
            E_private = private_model(batch_X_b_private)

            E_fused_top = torch.cat((E_a, E_shadow), dim=1)
            prediction = top_model(E_fused_top)

            # --- PHASE 2: LOSS CALCULATION ---
            pred_loss = criterion(prediction, batch_y)
            bw_loss = boundary_wandering_loss(E_shadow, batch_y)
            total_loss = pred_loss + ALPHA * bw_loss

            # --- PHASE 3: BACKWARD PASS (Simulated) ---
            total_loss.backward(retain_graph=True)

            optimizer_A.step()
            optimizer_top.step()
            optimizer_shadow.step()

            # Defender's Decoupled Update (Track 2)
            optimizer_private.zero_grad()
            optimizer_local_head.zero_grad()
            
            E_fused_local = torch.cat((E_shadow.detach(), E_private), dim=1)
            local_prediction = local_head(E_fused_local)
            local_loss = criterion(local_prediction, batch_y)
            local_loss.backward()
            
            optimizer_private.step()
            optimizer_local_head.step()

        print(f'Epoch [{epoch+1}/{EPOCHS}], Pred Loss: {pred_loss.item():.4f}, BW Loss: {bw_loss.item():.4f}')

    print("Training finished.")

    # Save models
    torch.save(party_a_model.state_dict(), 'party_a_model.pth')
    torch.save(shadow_model.state_dict(), 'shadow_model.pth')
    torch.save(private_model.state_dict(), 'private_model.pth')
    torch.save(local_head.state_dict(), 'local_head.pth')
    torch.save(top_model.state_dict(), 'top_model.pth')
    print("Models saved.")

if __name__ == '__main__':
    train()
