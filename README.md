# Voice emotion recognition neural network

In this project that I made during the machine learning course I tried to implement a machine learning system, which would be able to classify the emotion of the speaker based on a small audio sample. For this purposes two different models were constructed: a Multi-Layer Perceptron model and a Convolution Neural Network model. Both models achieved **approximately 75-80% accuracy** in classifying 8 different emotions. More detailed description of the project can be found in pdf file in the root.

## Datasets

For training purposes I combined four datasets:

- «The Ryerson Audio-Visual Database
  of Emotional Speech and Song (RAVDESS)»
- Toronto emotional speech set (TESS)
- Crowd Sourced Emotional Multimodal
  Actors Dataset (CREMA-D)
- Emotional Speech Database
  (ESD)

There are 29182 samples at total

## Technologies used

- Python
- Tensorflow/Keras
- Scikit-learn
- Numpy
- Librosa

## Training and running on own machine

If you want to train the model with own datasets, place them in subfolders in folder "samples" in the root of the project. Each sample must have the name in format "{label 1-8}\_{some id}.wav". So, the relative path can be, for example, _samples/dataset_name/1_123.wav_. Then, you should add subfolder's name to the **utils/FeaturesExtractor.py** file's **load_samples_new** function's array. Note, that the **feature extraction uses multithreading**; you can adjust number of threads in the same file.

Otherwise, you can use provided data.npz, cnn_model.keras and mlp_model.skops pickles. You can do this by

```py
system = EmotionRecognition("samples", mode="pickle")
system.train_models(load_saved=True)
```

Note, that output accuracy may be higher when using pretrained models with pickled data, since the validation and test sets are re-shuffled and may contain same files, that have already been used for training.
