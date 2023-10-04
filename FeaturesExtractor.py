
from librosa import load
import concurrent.futures
from utils import extract_features
import numpy as np
import os
from tqdm import tqdm
from sklearn.decomposition import PCA

max_threads = 4

class FeaturesExtractor:

  features = np.array([])
  labels = np.array([])


  def _read_files_from_folder(self, path, file_name):
    audio, sr = load(path, sr=None)
    extracted = extract_features(audio, sr=sr)
    label = int(file_name.split("_")[0])
    return (extracted, label)

  def _read_folder(self, path):
    file_names = list(filter(lambda x: x.endswith(".wav"), os.listdir(path)))
    files_count = len(file_names)
    features_array = []
    label_array = []
    with tqdm(total=files_count) as pbar:
      with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        futures = [executor.submit(self._read_files_from_folder, f"{path}/{file_name}", file_name) for file_name in file_names]
        for future in concurrent.futures.as_completed(futures):
          try:
              features_array_from_folder, label_array_from_folder = future.result()
              features_array.append(features_array_from_folder)
              label_array.append(label_array_from_folder)
              pbar.update(1)
          except Exception as e:
              print(f"Error processing audio: {e}")
    return (features_array, label_array)

  def load_samples_new(self, path):
    folders = ["crema", "ravdess", "tess", "esd"]
    features_array = []
    label_array = []
    for folder in folders:
      print(f"Start reading for folder {path}/{folder}")
      features_array_from_folder, label_array_from_folder = self._read_folder(f"{path}/{folder}")
      features_array += features_array_from_folder
      label_array += label_array_from_folder
    
    print("Feature extraction is completed")

    label_array = np.asarray(label_array)
    features_array = np.asarray(features_array)

    np.savez("pickles/data", features=features_array, label=label_array)
    self.features = features_array
    self.labels = label_array

    print("Successfully extracted features from the dataset")
    

  def load_from_pickle(self):
    print("Loading information from pickled file")
    files = np.load("pickles/data.npz")
  
    features_array = files["features"]
    label_array = files["label"]

    files.close()
    
    self.features = features_array
    self.labels = label_array

    print("Successfully loaded features and labels from pickle")

  def load_samples(self, path, mode="new"):
    if mode == "new":
      self.load_samples_new(path)
    else:
      self.load_from_pickle()

  
  def decompose_audio(self, path):
    file, sr = load(path, sr=None)
    features = extract_features(file, sr=sr)
    return features.reshape(1, -1)

  