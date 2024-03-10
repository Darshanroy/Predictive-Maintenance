import os
import pickle

def load_pickles_from_folder(folder_path):
    pickle_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    loaded_pickles = {}

    for file_name in pickle_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'rb') as file:
            loaded_pickles[file_name] = pickle.load(file)

    return loaded_pickles




