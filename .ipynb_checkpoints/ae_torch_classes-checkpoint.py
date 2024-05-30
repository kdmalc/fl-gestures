import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class GestureDatasetAE(Dataset):
    # NOTE: I think this formulation makes it so the dataloader won't return (X, Y) as per usual (eg like TensorDataset does). The AE doesn't need it (X,Y tho)
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RNNAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, seq_len, progressive=False):
        super(RNNAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.progressive = progressive
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        current_hidden_dim = hidden_dim
        for i in range(num_layers):
            encoder_layers.append(nn.RNN(prev_dim, current_hidden_dim, batch_first=True))
            prev_dim = current_hidden_dim
            if progressive and current_hidden_dim > 1:
                current_hidden_dim = max(1, current_hidden_dim // 2)  # Ensure hidden_dim does not go below 1
        self.encoder = nn.ModuleList(encoder_layers)
        
        # Decoder
        decoder_layers = []
        if progressive:
            current_hidden_dim = max(1, current_hidden_dim) * 2 
            # Multiply by 2 bc the current_hidden_dim is halved at the end of the encoder
        for i in range(num_layers):
            next_hidden_dim = min(hidden_dim, current_hidden_dim * 2) if progressive else hidden_dim
            decoder_layers.append(nn.RNN(current_hidden_dim, next_hidden_dim, batch_first=True))
            current_hidden_dim = next_hidden_dim
        decoder_layers.append(nn.Linear(current_hidden_dim, input_dim))  # Final layer to match input_dim
        self.decoder = nn.ModuleList(decoder_layers)
        
    def forward(self, x):
        batch_size = x.size(0)
        # Encoding
        for rnn in self.encoder:
            x, _ = rnn(x)
        # Decoding
        for rnn in self.decoder[:-1]:
            x, _ = rnn(x)
        x = self.decoder[-1](x)  # Final linear layer to match input dimension
        return x
    
    
def create_and_train_ae_model(input_dim, hidden_dim_lst, num_layers_lst, ae_class_obj=RNNAutoencoder, seq_len=64):
    raise ValueError("This function has not been completed yet.")
    # This needs to be rewritten so I can pass in lists and not rely on progressive's auto-halving...
    model = ae_class_obj(input_dim, hidden_dim, num_layers, seq_len, progressive)
    
    
def grid_search_rnn_autoencoder(train_loader, val_loader, input_dim, seq_len, param_grid, num_epochs=10):
    best_model = None
    best_loss = float('inf')
    best_params = None
    
    for trial_num, params in enumerate(ParameterGrid(param_grid)):
        hidden_dim = params['hidden_dim']
        num_layers = params['num_layers']
        progressive = params['progressive']

        print(f"Trial: {trial_num}; hidden_dim: {hidden_dim}; num_layers: {num_layers}; progressive: {progressive}")
        
        # Initialize the model with current params
        model = RNNAutoencoder(input_dim, hidden_dim, num_layers, seq_len, progressive)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
        
        # Validation loop
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            best_params = params
            
        print(f'Params: {params}, Validation Loss: {val_loss}\n')
    
    print(f'Best Params: {best_params}, Best Validation Loss: {best_loss}')
    return best_model, best_params
    
    