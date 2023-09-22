import matplotlib.pyplot as plt
from FeaturesExtractor import FeaturesExtractor
import numpy as np
from sklearn.manifold import TSNE

data = FeaturesExtractor()

data.load_samples("samples", mode="pickle")

pca = TSNE(n_components=2, init="random", perplexity=30, learning_rate="auto")

reduced_features = pca.fit_transform(data.features)

plt.figure(figsize=(9, 9))

x = reduced_features[:, 0]
y = reduced_features[:, 1]

colors = data.labels

sc = plt.scatter(x, y, c=colors, cmap='viridis', marker="o", s=20) # type: ignore

plt.colorbar(sc, label='Values from color array')

# Add labels and a title
plt.xlabel('TSNE 1')
plt.ylabel('TSNE 2')
plt.title('Dataset visualisation scatter plot')

# Show the plot
plt.show()