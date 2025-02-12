import numpy as np
import json
from sklearn.model_selection import KFold


SETUP_KFCV = False
RANDOM_SEED = 101
np.random.seed(RANDOM_SEED)

all_users = ['P008', 'P119', 'P131', 'P122', 'P110', 'P111', 'P010', 'P132',
       'P115', 'P102', 'P106', 'P121', 'P107', 'P116', 'P114', 'P128',
       'P103', 'P104', 'P004', 'P105', 'P126', 'P005', 'P127', 'P123',
       'P011', 'P125', 'P109', 'P112', 'P118', 'P006', 'P124', 'P108']


if SETUP_KFCV==False:
    np.random.shuffle(all_users)
    # Train/test split (24 for training, 8 for testing)
    train_users = all_users[:24]
    test_users = all_users[24:]

    # Save to a JSON file
    split_dict = {"all_users": all_users, "train_users": train_users, "test_users": test_users}
    with open("user_splits.json", "w") as f:
        json.dump(split_dict, f, indent=4)

    print("User splits saved to user_splits.json")
else:
    K = 5  # Number of folds for cross-validation
    kf = KFold(n_splits=K, shuffle=True, random_state=RANDOM_SEED)

    # Store folds
    cv_splits = []
    for train_idx, test_idx in kf.split(all_users):
        train_users = [all_users[i] for i in train_idx]
        test_users = [all_users[i] for i in test_idx]
        cv_splits.append({"train": train_users, "test": test_users})

    # Save to JSON
    with open("cv_splits.json", "w") as f:
        json.dump(cv_splits, f, indent=4)

    print(f"Saved {K}-fold cross-validation splits to cv_splits.json")

