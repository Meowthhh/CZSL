import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import scipy.io as sio

# Placeholder for AWA1 attribute tensor (replace with your actual data)
# Shape: [50, 85] (50 classes, 85 attributes)
matcontent = sio.loadmat("D:\\Research\\Prospects\\CZSL\\Code\\DK_CZSL\\src\\data\\AWA1\\att_splits.mat")
attribute_tensor = matcontent['att'].T
print(f"Shape of attributes: {attribute_tensor.shape}")

# AWA1 class names (based on standard AWA1 dataset)
class_names = [
    'antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'dalmatian', 'persian+cat', 'horse',
    'german+shepherd', 'blue+whale', 'siamese+cat', 'skunk', 'mole', 'tiger', 'hippopotamus',
    'leopard', 'moose', 'spider+monkey', 'humpback+whale', 'elephant', 'gorilla', 'ox',
    'fox', 'sheep', 'seal', 'chimpanzee', 'hamster', 'squirrel', 'rhinoceros', 'rabbit',
    'bat', 'giraffe', 'wolf', 'chihuahua', 'rat', 'weasel', 'otter', 'buffalo', 'zebra',
    'giant+panda', 'deer', 'bobcat', 'pig', 'lion', 'mouse', 'polar+bear', 'collie',
    'walrus', 'raccoon', 'cow', 'dolphin'
]

# Step 1: Standardize the attribute tensor
scaler = StandardScaler()
attribute_tensor_scaled = scaler.fit_transform(attribute_tensor)

# Step 2: Apply t-SNE for dimensionality reduction to 2D
tsne_model = TSNE(n_components=2, random_state=42, perplexity=10, n_iter=1000)
embeddings_2d = tsne_model.fit_transform(attribute_tensor_scaled)

# Step 3: Interactive Plotly visualization
df_plotly = {
    'x': embeddings_2d[:, 0],
    'y': embeddings_2d[:, 1],
    'class': class_names
}
fig = px.scatter(
    df_plotly,
    x='x',
    y='y',
    color='class',
    hover_data=['class'],
    title='t-SNE Visualization of AWA1 Attribute Tensor',
    width=800,
    height=600
)
fig.update_traces(marker=dict(size=10))
fig.update_layout(showlegend=True, legend=dict(font=dict(size=10)))
# fig.write('awa1_tsne_plotly.html')  # Save interactive plot
fig.show()

# Step 4: High-quality Matplotlib visualization
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    hue=class_names,
    palette='tab20',
    s=100,
    edgecolor='black',
    legend=False  # Disable legend to avoid clutter
)
# Annotate points with class names
for i, name in enumerate(class_names):
    plt.text(
        embeddings_2d[i, 0],
        embeddings_2d[i, 1] - 0.6,
        name,
        fontsize=6,
        alpha=0.8
    )
plt.title('t-SNE Visualization of AWA1 Attribute Tensor', fontsize=14)
plt.xlabel('t-SNE 1', fontsize=12)
plt.ylabel('t-SNE 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('awa1_tsne_matplotlib.png', dpi=300)  # Save high-res image
plt.show()