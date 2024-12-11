# THIS TRAINS A SINGLE GENERIC CNN AND TESTS ON WITHHELD USERS
## NO CLUSTERING!!!

import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from moments_engr import *
from DNN_FT_funcs import *
np.random.seed(42) 

do_normal_logging = False
finetune = False

path1 = 'C:\\Users\\kdmen\\Box\\Meta_Gesture_2024\\saved_datasets\\filtered_datasets\\$BStand_EMG_df.pkl'
with open(path1, 'rb') as file:
    raw_userdef_data_df = pickle.load(file)  # (204800, 19)

userdef_df = raw_userdef_data_df.groupby(['Participant', 'Gesture_ID', 'Gesture_Num']).apply(create_feature_vectors)
userdef_df = userdef_df.reset_index(drop=True)

all_participants = userdef_df['Participant'].unique()
# Shuffle the participants
np.random.shuffle(all_participants)
# Split into two groups
#train_participants = all_participants[:24]  # First 24 participants
test_participants = all_participants[24:]  # Remaining 8 participants

#convert Gesture_ID to numerical with new Gesture_Encoded column
label_encoder = LabelEncoder()
userdef_df['Gesture_Encoded'] = label_encoder.fit_transform(userdef_df['Gesture_ID'])
label_encoder2 = LabelEncoder()
userdef_df['Cluster_ID'] = label_encoder2.fit_transform(userdef_df['Participant'])

# Prepare data
data_splits = prepare_data(
    userdef_df, 'feature', 'Gesture_Encoded', 
    all_participants, test_participants, 
    training_trials_per_gesture=8, finetuning_trials_per_gesture=3,
)

# Train base model
results = main_training_pipeline(
    data_splits, 
    all_participants=all_participants, 
    test_participants=test_participants,
    model_type='CNN',  # Or 'RNN'
    num_epochs=50
)
# RESULTS CONTAINS:
#{
#        'model': model,
#        'train_performance': train_performance,
#        'intra_test_performance': intra_test_performance,
#        'cross_test_performance': cross_test_performance,
#        'train_accuracy': train_results['accuracy'],
#        'intra_test_accuracy': intra_test_results['accuracy'],
#        'cross_test_accuracy': cross_test_results['accuracy'],
#        'train_loss_log': train_loss_log,
#        'intra_test_loss_log': intra_test_loss_log,
#        'cross_test_loss_log': cross_test_loss_log
#    }

'''

class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes, 
                 use_batch_norm=False, dropout_rate=0.5):
        super(CNNModel, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        #self.init_params = {"use_batch_norm": self.use_batch_norm, "dropout_rate": self.dropout_rate}
        
        # Convolutional Layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32) if self.use_batch_norm else nn.Identity()
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64) if self.use_batch_norm else nn.Identity()
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * (input_dim // 4), 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Activation and Pooling
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)

    def forward(self, x):
        # Reshape input to (batch_size, 1, sequence_length)
        x = x.unsqueeze(1)
        
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Flatten and Fully Connected Layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

model = CNNModel(input_dim, num_classes).to('cpu')
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
# Loss and optimizer
optimizer = get_optimizer(model, lr=lr, use_weight_decay=use_weight_decay, weight_decay=weight_decay)
# Training
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, optimizer)

# Evaluation
train_results = evaluate_model(model, train_loader)
intra_test_results = evaluate_model(model, intra_test_loader)
cross_test_results = evaluate_model(model, cross_test_loader)

'''









if finetune:
    # Fine-tune on first test participant's data
    first_test_participant_data = data_splits['novel_trainFT']
    # Select just the first given participant ID from this dataset, probably for both train_data and labels

    participant_id = 'P132'
    # Filter based on participant_id
    indices = [i for i, pid in enumerate(first_test_participant_data['participant_ids']) if pid == participant_id]
    # Extract the corresponding features and labels
    features_p132 = [first_test_participant_data['features'][i] for i in indices]
    labels_p132 = [first_test_participant_data['labels'][i] for i in indices]

    fine_tuned_model = fine_tune_model(
        results['model'], 
        {'features':features_p132, 'labels':labels_p132}
    )

    # NEED TO EVALUATE FINETUNED MODEL, at least on said user's data...

if do_normal_logging:
    # Example usage in main script
    visualize_model_performance(results)

    # Example usage in main script
    log_file = log_performance(results)
    print(f"Detailed performance log saved to: {log_file}")
