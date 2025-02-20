# THIS TRAINS A SINGLE GENERIC CNN AND TESTS ON WITHHELD USERS
## NO CLUSTERING!!!

from moments_engr import *
from DNN_FT_funcs import *
from hyperparam_tuned_configs import *

from global_seed import set_seed
set_seed()

import os  
cwd = os.getcwd()
print("Current Working Directory: ", cwd)
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M")


FINETUNE = False
LOG_AND_VISUALIZE = False
MODEL_STR = "CNNModel3layer" #"DynamicMomonaNet" "ELEC573Net" "OriginalELEC573CNN"
FEATENG = "moments"  # "moments" "FS" None
MY_CONFIG = determine_config(MODEL_STR, feateng=FEATENG)
# TODO: Turn this on!
#MY_CONFIG["user_split_json_filepath"] = "April_25\\fixed_user_splits\\24_8_user_splits_RS17.json"


expdef_df = load_expdef_gestures(feateng_method=MY_CONFIG["feature_engr"])
data_splits, all_participants, test_participants = make_data_split(expdef_df, MY_CONFIG, return_participants=True)

# Train base (AKA generic, pretrained) model
results = main_training_pipeline(
    data_splits, 
    all_participants=all_participants, 
    test_participants=test_participants,
    model_type=MODEL_STR,
    config=MY_CONFIG)

#full_path = os.path.join(cwd, 'ELEC573_Proj', 'models', 'generic_CNN_model.pth')
full_path = MY_CONFIG['models_save_dir']
os.makedirs(full_path, exist_ok=True)  # Ensure the directory exists
print("Full Path:", full_path)
# TODO: Could just use my existing save_model() func here...
torch.save(results["model"].state_dict(), f"{full_path}\\pretrained_{MODEL_STR}_model.pth")

# TODO: This only finetunes on one person lol should at least finetune over a couple
## Well I guess if this is just for debugging not performance this is fine
if FINETUNE:
    # Fine-tune on first test participant's data
    first_test_participant_data = data_splits['novel_trainFT']
    # Select just the first given participant ID from this dataset, probably for both train_data and labels

    participant_id = first_test_participant_data['participant_ids'][0]
    # Filter based on participant_id
    indices = [i for i, pid in enumerate(first_test_participant_data['participant_ids']) if pid == participant_id]
    ft_dataset = GestureDataset([first_test_participant_data['feature'][i] for i in indices], [first_test_participant_data['labels'][i] for i in indices])
    ft_loader = DataLoader(ft_dataset, batch_size=MY_CONFIG["batch_size"], shuffle=True)

    indices = [i for i, datasplit_pid in enumerate(first_test_participant_data['participant_ids']) if datasplit_pid == participant_id]
    # ^ These indices should be the same as the above indices I think...
    intra_test_dataset = GestureDataset([first_test_participant_data['feature'][i] for i in indices], [first_test_participant_data['labels'][i] for i in indices])
    intra_test_loader = DataLoader(intra_test_dataset, batch_size=MY_CONFIG["batch_size"], shuffle=True)

    finetuned_model, frozen_original_model, train_loss_log, test_loss_log = fine_tune_model(
        results['model'], ft_loader, MY_CONFIG, MY_CONFIG["timestamp"], test_loader=intra_test_loader, pid=participant_id
    )

if LOG_AND_VISUALIZE:
    # Does this visualize ONLY heatmaps, and not train/test loss curves? I only really care about the latter...
    #visualize_model_acc_heatmap(results)

    visualize_train_test_loss_curves(results, MY_CONFIG)
    visualize_train_test_loss_curves(None, MY_CONFIG, train_loss_log, test_loss_log, my_title="1subj FT Train Test Curves", ft=True)

    log_performance(results, MY_CONFIG)
