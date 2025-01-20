import numpy as np
import pandas as pd

df = pd.concat([pd.read_csv("../target.csv"), pd.read_csv("../training_data.csv")], ignore_index=True).drop_duplicates()

length = df.shape[0]

train_indices = np.array(np.random.choice(range(length), int(length * 0.8), replace=False), dtype=int)
val_indices = np.array(list(set(range(length)) - set(train_indices)), dtype=int)

np.save("./train_indices.npy", train_indices, allow_pickle=True)
np.save("./val_indices.npy", val_indices, allow_pickle=True)