import os


def change_crema():
    i = 0
    for filename in os.listdir(f"samples/crema"):
      emotion = filename.split("_")[2]
      emotion_num = 1
      if emotion == "ANG":
        emotion_num = 5
      elif emotion == "DIS":
        emotion_num = 7
      elif emotion == "FEA":
        emotion_num = 6
      elif emotion == "HAP":
        emotion_num = 3
      elif emotion == "NEU":
        emotion_num = 1
      elif emotion == "SAD":
        emotion_num = 4
      os.rename(f"samples/crema/{filename}", f"samples/crema/{emotion_num}_{i}.wav")
      i+= 1
def change_ravdess():
    folders2 = os.listdir(f"samples/ravdess")
    i=0
    for folder2 in folders2:
      if folder2 == ".DS_Store": continue
      files = os.listdir(f"samples/ravdess/{folder2}")
      for filename in files:
        emotion = int(filename.split("-")[2])
        os.rename(f"samples/ravdess/{folder2}/{filename}", f"samples/ravdess/{emotion}_{i}.wav")
        i+= 1
def change_tess():
    i = 0
    folders2 = os.listdir("samples/tess")
    for folder2 in folders2:
      if folder2 == ".DS_Store": continue
      emotion = folder2.split("_")[1]
      emotion_num = 1
      if emotion == "angry":
        emotion_num = 5
      elif emotion == "disgust":
        emotion_num = 7
      elif emotion == "fear":
        emotion_num = 6
      elif emotion == "happy":
        emotion_num = 3
      elif emotion == "neutral":
        emotion_num = 1
      elif emotion == "sad":
        emotion_num = 4
      elif emotion == "surprise":
        emotion_num = 8
      files = os.listdir(f"samples/tess/{folder2}")
      for filename in files:
        os.rename(f"samples/tess/{folder2}/{filename}", f"samples/tess/{emotion_num}_{i}.wav")
        i += 1

def change_esd():
  i=0
  for base_folder in os.listdir("samples/esd"):
    if base_folder == ".DS_Store": continue
    for subfolder in os.listdir(f"samples/esd/{base_folder}"):
      if subfolder == ".DS_Store": continue
      emotion_num = 1
      if subfolder == "Angry":
        emotion_num = 5
      elif subfolder == "Happy":
        emotion_num = 3
      elif subfolder == "Neutral":
        emotion_num = 1
      elif subfolder == "Sad":
        emotion_num = 4
      elif subfolder == "Surprise":
        emotion_num = 8
      for filename in os.listdir(f"samples/esd/{base_folder}/{subfolder}"):
        if filename == ".DS_Store": continue
        os.rename(f"samples/esd/{base_folder}/{subfolder}/{filename}", f"samples/esd/{emotion_num}_{i}.wav")
        i += 1

change_esd()