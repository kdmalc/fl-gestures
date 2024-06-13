import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


class GestureDatasetAE(Dataset):
    # NOTE: I think this formulation makes it so the dataloader won't return (X, Y) as per usual (eg like TensorDataset does). The AE doesn't need it (X,Y tho)
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RNNAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, seq_len, mirror=False, progressive=False, verbose=False):
        super(RNNAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.progressive = progressive
        self.verbose = verbose

        # Check that only one of progressive or hidden_dim list is provided
        if isinstance(hidden_dim, list) and progressive:
            raise ValueError("Both 'hidden_dim' as list and 'progressive' cannot be true simultaneously.")
        if isinstance(hidden_dim, list) and mirror:
            num_layers = len(hidden_dim)
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]*num_layers*2 # Not sure if *2 is always necessary... is for how 

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        if progressive:
            current_hidden_dim = hidden_dim[0]
            for i in range(num_layers):
                if self.verbose:
                    print(f"Encoder Layer {i}: prev_dim: {prev_dim}, current_hidden_dim: {current_hidden_dim}")
                encoder_layers.append(nn.RNN(prev_dim, current_hidden_dim, batch_first=True))
                prev_dim = current_hidden_dim
                if current_hidden_dim > 1:
                    current_hidden_dim = max(1, current_hidden_dim // 2)
        else:
            for i in range(num_layers):
                current_hidden_dim = hidden_dim[i]
                encoder_layers.append(nn.RNN(prev_dim, current_hidden_dim, batch_first=True))
                prev_dim = current_hidden_dim
        self.encoder = nn.ModuleList(encoder_layers)

        if self.verbose:
            print(f"Bottleneck current_hidden_dim: {current_hidden_dim}")

        # Decoder
        decoder_layers = []
        if progressive:
            #current_hidden_dim = max(1, hidden_dim) // (2 ** (num_layers - 1)) * 2  # Start from the halved state at the end of encoder
            current_hidden_dim = max(1, current_hidden_dim) * 2
            if self.verbose:
                print(f"Starting decoder: current_hidden_dim (post*2): {current_hidden_dim}")
            for i in range(num_layers):
                next_hidden_dim = min(hidden_dim[0], current_hidden_dim * 2)
                if self.verbose:
                    print(f"Decoder Layer {i}: current_hidden_dim: {current_hidden_dim}, next_hidden_dim: {next_hidden_dim}")
                decoder_layers.append(nn.RNN(current_hidden_dim, next_hidden_dim, batch_first=True))
                current_hidden_dim = next_hidden_dim
        elif mirror:
            mirrored_hidden_dims = hidden_dim[::-1]
            for i in range(num_layers):
                current_hidden_dim = mirrored_hidden_dims[i]
                if self.verbose:
                    print(f"Decoder Layer {i}: prev_dim: {prev_dim}, current_hidden_dim: {current_hidden_dim}")
                decoder_layers.append(nn.RNN(prev_dim, current_hidden_dim, batch_first=True))
                prev_dim = current_hidden_dim
        else:
            # Split hidden_dim list into encoder and decoder parts
            encoder_hidden_dims = hidden_dim[:num_layers]
            decoder_hidden_dims = hidden_dim[num_layers:]
            if len(decoder_hidden_dims) != num_layers:
                raise ValueError("Hidden_dim list length must be twice the number of layers (num_layers).")
            for i in range(num_layers):
                current_hidden_dim = decoder_hidden_dims[i]
                decoder_layers.append(nn.RNN(prev_dim, current_hidden_dim, batch_first=True))
                prev_dim = current_hidden_dim

        decoder_layers.append(nn.Linear(current_hidden_dim, input_dim))  # Final layer to match input_dim
        self.decoder = nn.ModuleList(decoder_layers)

    def forward(self, x):
        batch_size = x.size(0)
        if self.verbose:
            print(f'FORWARD\nInitial input shape: {x.shape}')

        # Encoding
        for idx, rnn in enumerate(self.encoder):
            x, _ = rnn(x)
            if self.verbose:
                print(f'After encoder layer {idx}, shape: {x.shape}')
        # Decoding
        for idx, rnn in enumerate(self.decoder[:-1]):
            x, _ = rnn(x)
            if self.verbose:
                print(f'After decoder layer {idx}, shape: {x.shape}')

        x = self.decoder[-1](x)  # Final linear layer to match input dimension
        if self.verbose:
            print(f'After final linear layer, shape: {x.shape}')
        return x

    def encode(self, x):
        # Ensure the input is a tensor
        if isinstance(x, list):
            # Debugging: print shapes of each tensor in the list
            print("Shapes in list to stack:")
            for i, t in enumerate(x):
                print(f"Entry {i} shape: {t.shape}")
            # Ensure all tensors have the same shape before stacking
            max_shape = torch.tensor([tensor.shape for tensor in x]).max(dim=0)[0]
            x = [torch.nn.functional.pad(t, (0, max_shape[-1] - t.shape[-1])) if t.shape != max_shape else t for t in x]
            x = torch.stack(x)

        for rnn in self.encoder:
            x, _ = rnn(x)
        return x
    
    
def create_and_train_ae_model(input_dim, hidden_dim_lst, num_layers, train_loader, val_loader, mirror_bool=False, progressive_bool=False, ae_class_obj=RNNAutoencoder, seq_len=64, num_epochs=10, verbose=False, plot_loss=True, learning_rate=0.001, fig_size=(7,5)):
    
    model = ae_class_obj(input_dim, hidden_dim_lst, num_layers, seq_len, mirror=mirror_bool, progressive=progressive_bool, verbose=verbose)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_loss_log = []
    val_loss_log = []
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = train_loss/len(train_loader)
        train_loss_log.append(train_loss)
        # Validation loop
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_log.append(val_loss)
        print(f'Epoch {epoch}: Train Loss: {train_loss};  Validation Loss: {val_loss}')

    if plot_loss:
        epochs_range = range(1, len(train_loss_log) + 1)
        plt.figure(figsize=fig_size)
        plt.plot(epochs_range, train_loss_log, 'b', label='Training Loss')
        plt.plot(epochs_range, val_loss_log, 'purple', label='Validation Loss')
        plt.title("Model Loss During Training")
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.show()
        
    return model, train_loss_log, val_loss_log
    
    
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
    
    