# 📊 Thematic Analysis of Customer Feedback

**Interactive clustering and topic discovery for airline customer feedback using embeddings, dimensionality reduction, and machine learning.**

A complete end-to-end pipeline for analyzing customer feedback tweets, discovering hidden themes/topics through clustering, and exploring results via an interactive Streamlit web interface.

---

## 🎯 Project Overview

This project applies **thematic analysis** to airline customer feedback tweets by:
1. Converting tweets to semantic embeddings using SBERT (Sentence-BERT)
2. Reducing embedding dimensionality with PCA
3. Discovering clusters/topics using **KMeans** and **DBSCAN**
4. Generating human-readable cluster labels via TF-IDF
5. Providing interactive visualization through a **Streamlit web app**

**Use Cases:**
- Customer feedback analysis & sentiment tracking
- Topic discovery in social media data
- Product feedback clustering
- Voice of customer (VoC) programs

---

## ✨ Features

### 📋 Core Analysis
- **Embedding Generation**: Uses SBERT (`all-MiniLM-L6-v2`) for fast, high-quality sentence embeddings
- **Dimensionality Reduction**: PCA preserving 95% variance for visualization & efficiency
- **KMeans Clustering**: Configurable number of clusters (2-30) with silhouette & Davies-Bouldin scoring
- **DBSCAN Clustering**: Density-based clustering with noise detection and k-distance graphs
- **Cluster Labeling**: Auto-generated labels via TF-IDF keyword extraction

### 🎨 Streamlit Frontend
- **Interactive Configuration**: Real-time parameter adjustment (cluster count, EPS, min_samples)
- **Multi-Tab Dashboard**:
  - 📈 **KMeans Overview**: Metrics, distribution charts, PCA scatter plot
  - 🎯 **Cluster Details**: Explore individual clusters with representative tweets
  - 🔍 **DBSCAN Analysis**: K-distance graph, noise detection
  - 💾 **Export Data**: Download results as CSV
- **Real-time Visualization**: Matplotlib/Seaborn plots embedded in the web app
- **Sentiment Analysis**: Per-cluster sentiment breakdowns

### 📊 Jupyter Notebook
- Complete analysis pipeline in `notebook/CustomerAnalysis.ipynb`
- Step-by-step data cleaning, embedding, clustering, and evaluation
- Metrics computation and visualization

---

## 📦 Project Structure

```
Thematic-Analysis-of-Customer-Feedback/
├── app.py                          # 🎨 Streamlit web frontend
├── requirements_streamlit.txt      # 🔧 Python dependencies
├── README.md                       # 📖 This file
├── dataset/
│   └── Tweets.csv                  # 📊 Sample airline tweets dataset
└── notebook/
    └── CustomerAnalysis.ipynb      # 📓 Jupyter analysis notebook
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git
- 2GB free disk space (for models)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/SiddhantSaha7/Thematic-Analysis-of-Customer-Feedback.git
cd Thematic-Analysis-of-Customer-Feedback
```

2. **Create a virtual environment**:
```bash
python3 -m venv venv_streamlit
source venv_streamlit/bin/activate  # On Windows: venv_streamlit\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements_streamlit.txt
```

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

### Running the Jupyter Notebook

```bash
jupyter notebook notebook/CustomerAnalysis.ipynb
```

---

## 📖 Usage Guide

### Streamlit Web App

1. **Configure Parameters** (Left Sidebar):
   - Select data source (default dataset or upload CSV)
   - Set KMeans cluster count (2-30)
   - Adjust DBSCAN EPS and min_samples

2. **Click "🚀 Run Analysis"** to process data

3. **Explore Results**:
   - **📈 KMeans Overview**: View cluster distribution and PCA visualization
   - **🎯 Cluster Details**: Click cluster ID to see representative tweets & sentiment
   - **🔍 DBSCAN Analysis**: Examine k-distance graph and noise points
   - **💾 Export Data**: Download cluster results and summaries

### Input CSV Format

Expected columns for custom dataset:
- `text` (required): Original tweet/feedback text
- `airline_sentiment` (optional): Sentiment label (positive/negative/neutral)
- Other columns preserved in export

**Example**:
```csv
text,airline_sentiment
"Great service and punctuality!",positive
"Flight was delayed 2 hours",negative
```

---

## 🔧 Configuration & Customization

### Environment Variables
Thread limits set automatically to prevent kernel crashes:
```python
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
```

### Model Configuration (in `app.py`)
- **SBERT Model**: Change line ~85 to use different embedding model
  ```python
  embedder = SentenceTransformer('model-name', device=device)
  ```
- **PCA Variance**: Adjust `n_components=0.95` (line ~120)
- **Default Clusters**: Change `n_clusters=17` (line ~130)

### Performance Tuning
- **Faster embeddings**: Use `'sentence-transformers/all-MiniLM-L6-v2'` (default, ~50MB)
- **Better quality**: Use `'sentence-transformers/all-mpnet-base-v2'` (~400MB, slower)
- **GPU acceleration**: Automatically enabled if CUDA/PyTorch GPU available

---

## 📊 Understanding the Results

### Silhouette Score
- Range: -1 to 1
- Higher is better (indicates well-separated clusters)
- 0.5+ suggests reasonable clustering

### Davies-Bouldin Score
- Range: 0 to ∞
- Lower is better (indicates distinct clusters)
- <1.0 suggests good separation

### DBSCAN Noise Points
- Cluster ID = -1 indicates noise/outliers
- Helps identify tweets that don't fit any topic
- Noise % shows proportion of non-clustered data

---

## 🐛 Troubleshooting

### "Dataset not found"
Ensure `dataset/Tweets.csv` exists in project root.

### Slow first run
First run downloads SBERT model (~200MB). Subsequent runs use cache.

### CUDA out of memory
App auto-fallbacks to CPU. Reduce dataset size if needed.

### Import errors
Reinstall dependencies:
```bash
pip install --upgrade --force-reinstall -r requirements_streamlit.txt
```

### Port 8501 already in use
```bash
streamlit run app.py --server.port=8502
```

---

## 📚 Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Streamlit** | Interactive web frontend |
| **Sentence-Transformers** | Semantic embeddings (SBERT) |
| **Scikit-learn** | Clustering & dimensionality reduction |
| **PyTorch** | Deep learning backend |
| **Pandas/NumPy** | Data manipulation |
| **Matplotlib/Seaborn** | Visualization |
| **Transformers** | NLP model pipeline |

---

## 📈 Pipeline Overview

```
Raw Tweets
    ↓
Clean & Preprocess
    ↓
Generate Embeddings (SBERT)
    ↓
Dimensionality Reduction (PCA)
    ↓
Clustering
    ├─ KMeans (17 clusters)
    └─ DBSCAN (eps=0.8, min_samples=20)
    ↓
Label Generation (TF-IDF)
    ↓
Interactive Visualization (Streamlit)
```

---

## 🎓 Key Concepts

### Semantic Embeddings
Converting text to dense numerical vectors that capture meaning. Similar texts have similar embeddings.

### PCA (Principal Component Analysis)
Reduces high-dimensional embeddings while preserving 95% of information variance.

### KMeans
Partitions data into k clusters by minimizing within-cluster variance. Good for fixed cluster count.

### DBSCAN
Density-based clustering that finds clusters of arbitrary shape and identifies noise points.

### TF-IDF (Term Frequency-Inverse Document Frequency)
Extracts important keywords from each cluster for human-readable labels.

---

## 📊 Sample Results

**Airline Feedback Dataset (14,640 tweets)**
- **KMeans Clusters**: 17 topics identified
- **Silhouette Score**: 0.42 (reasonable clustering)
- **Top Clusters**: Flight delays, customer service, boarding, luggage, flight quality
- **Sentiment**: 42% positive, 38% negative, 20% neutral

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- [ ] LLM-based cluster summarization (GPT-4, Claude)
- [ ] HDBSCAN alternative
- [ ] Hierarchical clustering visualization
- [ ] Interactive cluster merging
- [ ] Export to PDF reports
- [ ] Real-time streaming data support

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👤 Author

**SiddhantSaha7**  
GitHub: [@SiddhantSaha7](https://github.com/SiddhantSaha7)

---

## 🙏 Acknowledgments

- SBERT models from [Sentence-Transformers](https://www.sbert.net/)
- Clustering algorithms from [Scikit-learn](https://scikit-learn.org/)
- Web framework [Streamlit](https://streamlit.io/)
- Airline feedback dataset (sample)

---

## 📞 Support

For issues, questions, or suggestions:
1. Open an issue on GitHub
2. Check existing issues for solutions
3. Provide detailed error messages and reproduction steps

---

## 🚀 Future Roadmap

- [ ] Multi-language support
- [ ] Real-time feedback ingestion
- [ ] Advanced sentiment analysis
- [ ] Trend detection over time
- [ ] API for model deployment
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)