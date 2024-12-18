import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt

nltk.download('stopwords')

data1 = pd.read_csv("./full_dataset/goemotions_1.csv")
data2 = pd.read_csv("./full_dataset/goemotions_2.csv")
data3 = pd.read_csv("./full_dataset/goemotions_3.csv")

data = pd.concat([data1, data2, data3], ignore_index=True)
print("Datasets loaded and concatenated successfully!")

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return " ".join(words)

data['text_clean'] = data['text'].apply(preprocess_text)
print("Text preprocessing complete!")

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text_clean']).toarray()
print(f"TF-IDF vectorization complete! Shape: {X.shape}")

input_dim = X.shape[1]
encoding_dim = 128  

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder = Model(inputs=input_layer, outputs=encoded)  

autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
print("Autoencoder model built successfully!")

history = autoencoder.fit(X, X, epochs=800, batch_size=32, validation_split=0.2, verbose=1)
print("Autoencoder training complete!")

autoencoder.save("feel_meV1_autoencoder.h5")
print("Autoencoder model saved as 'feel_meV1_autoencoder.h5'")

encoder.save("feel_meV1_encoder.h5")
print("Encoder model saved as 'feel_meV1_encoder.h5'")

latent_features = encoder.predict(X)
print(f"Latent feature extraction complete! Shape: {latent_features.shape}")

num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(latent_features)

data['cluster'] = kmeans.labels_

inertia = kmeans.inertia_
silhouette_avg = silhouette_score(latent_features, kmeans.labels_)
print(f"Inertia: {inertia}")
print(f"Silhouette Score: {silhouette_avg}")

cluster_sizes = data['cluster'].value_counts()
plt.figure(figsize=(8, 5))
plt.bar(cluster_sizes.index, cluster_sizes.values)
plt.title("Cluster Sizes")
plt.xlabel("Cluster")
plt.ylabel("Number of Samples")
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Autoencoder Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

print("\n=== Cluster Examples ===\n")
for cluster_id in range(num_clusters):
    print(f"\nCluster {cluster_id}:")
    print(data[data['cluster'] == cluster_id]['text'].head(5).to_string(index=False))  # Show first 5 examples

while True:
    input_text = input("\nEnter a text to analyze (or type 'exit' to quit): ")
    if input_text.lower() == 'exit':
        print("Exiting the program.")
        break

    input_clean = preprocess_text(input_text)
    input_vector = vectorizer.transform([input_clean]).toarray()
    input_latent = encoder.predict(input_vector)
    
    predicted_cluster = kmeans.predict(input_latent)[0]
    print(f"Input text: {input_text}")
    print(f"Predicted cluster: {predicted_cluster}")
