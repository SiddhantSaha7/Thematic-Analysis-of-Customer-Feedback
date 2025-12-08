# Streamlit Frontend - Thematic Analysis of Customer Feedback

A fully interactive Streamlit web app for exploring and visualizing KMeans and DBSCAN clustering of airline customer feedback tweets.

## 🎯 Features

- **Data Loading**: Upload your own CSV or use the default dataset
- **Embeddings**: Generates sentence embeddings using SBERT (Sentence Transformers)
- **Dimensionality Reduction**: Applies PCA to reduce embedding dimensions while preserving 95% variance
- **KMeans Clustering**: 
  - Configurable number of clusters (2-30)
  - Silhouette and Davies-Bouldin scoring
  - Visualization on PCA components
  - Cluster distribution analysis
- **DBSCAN Clustering**:
  - Configurable EPS and min_samples
  - K-distance graph for EPS selection
  - Noise point identification
- **Cluster Analysis**:
  - Human-readable labels via TF-IDF keyword extraction
  - Sentiment distribution per cluster
  - Representative tweets (closest to centroid)
- **Export**: Download cluster analysis results and summaries as CSV

## 🚀 Quick Start

### Installation

1. Create a virtual environment (recommended):
```bash
cd /Users/arth/Desktop/Thematic-Analysis-of-Customer-Feedback
python3 -m venv venv_streamlit
source venv_streamlit/bin/activate  # On Windows: venv_streamlit\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements_streamlit.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

## 📊 Usage

1. **Configure Parameters** (Sidebar):
   - Select data source (default dataset or upload CSV)
   - Set KMeans cluster count (2-30)
   - Adjust DBSCAN EPS and min_samples

2. **Click "Run Analysis"** to start processing

3. **Explore Tabs**:
   - **📈 KMeans Overview**: Cluster distribution, PCA scatter plot, metrics
   - **🎯 Cluster Details**: Select individual cluster, view tweets, sentiment breakdown
   - **🔍 DBSCAN Analysis**: DBSCAN results, K-distance graph, noise analysis
   - **💾 Export Data**: Download results and summaries

## 📋 CSV Format

Expected columns in uploaded CSV:
- `text`: Original tweet text
- `airline_sentiment`: Sentiment label (positive, negative, neutral) [optional]
- Other columns are preserved and exported

## ⚙️ Performance Notes

- **First run**: Model downloads may take 1-2 minutes (SBERT ~200MB)
- **Caching**: Results are cached in-memory for fast interaction
- **GPU**: Automatically uses CUDA if available; falls back to CPU
- **Thread limits**: Automatically set to avoid kernel crashes during clustering

## 🔧 Customization

Edit `app.py` to customize:
- SBERT model: Change `'all-MiniLM-L6-v2'` to a larger/smaller model
- PCA variance threshold: Adjust `n_components=0.95`
- Default cluster count: Change `n_clusters=17`
- TF-IDF top terms: Modify cluster label extraction logic

## 📝 Example Workflow

1. Open the app
2. Select "Use default dataset" (or upload your tweets CSV)
3. Set n_clusters = 12, EPS = 0.7
4. Click "Run Analysis"
5. Navigate to "Cluster Details" and explore individual clusters
6. Use "Export Data" to download results for further analysis

## 🐛 Troubleshooting

**Error: "Default dataset not found"**
- Ensure `dataset/Tweets.csv` exists in the same directory as `app.py`

**Slow embeddings generation**
- First run downloads SBERT model (~200MB). Subsequent runs use cache.

**CUDA out of memory**
- Reduce dataset size or use CPU (automatic fallback if CUDA fails)

**ModuleNotFoundError**
- Ensure all dependencies installed: `pip install -r requirements_streamlit.txt`

## 📦 Project Structure

```
Thematic-Analysis-of-Customer-Feedback/
├── app.py                          # Main Streamlit app
├── requirements_streamlit.txt      # Dependencies
├── dataset/
│   └── Tweets.csv                  # Sample data
├── notebook/
│   └── CustomerAnalysis.ipynb      # Jupyter analysis notebook
└── README.md                        # Original project docs
```

## 📄 License

Part of the Thematic Analysis of Customer Feedback project.
