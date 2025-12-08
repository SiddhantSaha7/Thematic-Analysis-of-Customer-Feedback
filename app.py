"""
Streamlit Frontend for Thematic Analysis of Customer Feedback
Visualizes and explores KMeans and DBSCAN clustering results with GenAI labels.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG & SETUP
# ============================================================================

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set thread limits early
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Streamlit page config
st.set_page_config(
    page_title="Thematic Analysis - Customer Feedback",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CACHING & DATA LOADING
# ============================================================================

@st.cache_resource
def load_embedder():
    """Load SBERT embedder (cached once per session)."""
    return SentenceTransformer('all-MiniLM-L6-v2', device=device)

@st.cache_data
def load_data(file_path):
    """Load and clean tweet data."""
    df = pd.read_csv(file_path)
    
    def clean_tweet(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = " ".join([word for word in text.split() if not word.startswith('@')])
        return text.strip()
    
    df['cleaned_text'] = df['text'].apply(clean_tweet)
    df = df[df['cleaned_text'].str.len() > 5].reset_index(drop=True)
    return df

@st.cache_data
def generate_embeddings(texts, embedder):
    """Generate embeddings for tweets (cached)."""
    return embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

@st.cache_data
def apply_pca(embeddings, n_components=0.95):
    """Apply PCA to embeddings."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    embeddings_pca = pca.fit_transform(embeddings)
    return embeddings_pca, pca

@st.cache_data
def apply_kmeans(embeddings_pca, n_clusters=17):
    """Apply KMeans clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(embeddings_pca)
    sil_score = silhouette_score(embeddings_pca, labels)
    db_score = davies_bouldin_score(embeddings_pca, labels)
    return kmeans, labels, sil_score, db_score

@st.cache_data
def apply_dbscan(embeddings_pca, eps=0.8, min_samples=20):
    """Apply DBSCAN clustering."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings_pca)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    return dbscan, labels, n_clusters, n_noise

def generate_cluster_labels(df, kmeans_labels, embeddings_pca, kmeans, top_n=5):
    """Generate human-readable labels for clusters using TF-IDF."""
    cluster_labels = {}
    
    for cl in sorted(np.unique(kmeans_labels)):
        inds = np.where(kmeans_labels == cl)[0]
        if len(inds) == 0:
            cluster_labels[cl] = "(empty)"
            continue
        
        # Find top-n representative tweets closest to centroid
        try:
            centroid = kmeans.cluster_centers_[cl]
            emb_cluster = embeddings_pca[inds]
            dists = np.linalg.norm(emb_cluster - centroid, axis=1)
            closest_rel = np.argsort(dists)[:top_n]
            closest_idx = inds[closest_rel]
        except Exception:
            closest_idx = inds[:top_n]
        
        examples = df.loc[closest_idx, 'cleaned_text'].tolist()
        
        # Extract top TF-IDF terms as label
        try:
            vec = TfidfVectorizer(stop_words='english', max_features=100)
            X = vec.fit_transform(examples)
            scores = np.asarray(X.sum(axis=0)).ravel()
            terms = np.array(vec.get_feature_names_out())
            if len(terms) == 0:
                label = examples[0][:60]
            else:
                top_terms = terms[np.argsort(scores)[::-1][:6]]
                label = ', '.join(top_terms)
        except Exception:
            label = examples[0][:80]
        
        cluster_labels[cl] = label
    
    return cluster_labels

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.title("📊 Thematic Analysis of Customer Feedback")
    st.markdown("### Interactive Clustering & Topic Discovery for Airline Tweets")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload or default path
        data_source = st.radio("Select data source:", ["Use default dataset", "Upload CSV"])
        
        if data_source == "Use default dataset":
            file_path = "dataset/Tweets.csv"
        else:
            uploaded_file = st.file_uploader("Upload CSV with tweets", type=["csv"])
            if uploaded_file is None:
                st.info("Please upload a CSV file to proceed.")
                st.stop()
            # Save uploaded file temporarily
            file_path = uploaded_file
        
        # Clustering parameters
        st.subheader("KMeans Parameters")
        n_clusters = st.slider("Number of clusters", 2, 30, 17)
        
        st.subheader("DBSCAN Parameters")
        eps_value = st.slider("EPS (distance threshold)", 0.1, 2.0, 0.8, step=0.1)
        min_samples = st.slider("Min samples per cluster", 5, 50, 20)
        
        st.divider()
        if st.button("🚀 Run Analysis", key="run_analysis"):
            st.session_state.run_analysis = True
    
    # Load data
    try:
        df = load_data(file_path)
        st.success(f"✅ Loaded {len(df)} tweets")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Check if we should run analysis
    if not st.session_state.get("run_analysis", False):
        st.info("👈 Configure parameters in the sidebar and click 'Run Analysis' to begin.")
        st.stop()
    
    # Generate embeddings
    with st.spinner("Generating embeddings..."):
        embedder = load_embedder()
        embeddings = generate_embeddings(df['cleaned_text'].tolist(), embedder)
        st.success("✅ Embeddings generated")
    
    # Apply PCA
    with st.spinner("Applying PCA..."):
        embeddings_pca, pca = apply_pca(embeddings, n_components=0.95)
        st.success(f"✅ PCA reduced dimensions from {embeddings.shape[1]} to {embeddings_pca.shape[1]}")
    
    # Apply KMeans
    with st.spinner("Running KMeans clustering..."):
        kmeans, kmeans_labels, sil_score, db_score = apply_kmeans(embeddings_pca, n_clusters=n_clusters)
        df['kmeans_cluster'] = kmeans_labels
        st.success(f"✅ KMeans clustering completed (k={n_clusters})")
    
    # Apply DBSCAN
    with st.spinner("Running DBSCAN clustering..."):
        dbscan, dbscan_labels, n_clusters_db, n_noise = apply_dbscan(embeddings_pca, eps=eps_value, min_samples=min_samples)
        df['dbscan_cluster'] = dbscan_labels
        st.success(f"✅ DBSCAN clustering completed ({n_clusters_db} clusters, {n_noise} noise points)")
    
    # Generate cluster labels
    with st.spinner("Generating cluster labels..."):
        cluster_labels = generate_cluster_labels(df, kmeans_labels, embeddings_pca, kmeans, top_n=5)
        df['cluster_label'] = df['kmeans_cluster'].map(cluster_labels)
        st.success("✅ Cluster labels generated")
    
    st.divider()
    
    # Display results in tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 KMeans Overview", "🎯 Cluster Details", "🔍 DBSCAN Analysis", "💾 Export Data"]
    )
    
    # ========== TAB 1: KMEANS OVERVIEW ==========
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Number of Clusters", n_clusters)
        with col2:
            st.metric("Silhouette Score", f"{sil_score:.4f}")
        with col3:
            st.metric("Davies-Bouldin Score", f"{db_score:.4f}")
        with col4:
            st.metric("Total Tweets", len(df))
        
        st.subheader("Cluster Distribution")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart
            counts = df['kmeans_cluster'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(counts.index, counts.values, color='viridis', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Number of Tweets')
            ax.set_title(f'Distribution of {n_clusters} Clusters')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
        
        with col1:
            # PCA scatter plot
            fig, ax = plt.subplots(figsize=(12, 6))
            scatter = ax.scatter(
                embeddings_pca[:, 0], embeddings_pca[:, 1],
                c=kmeans_labels, cmap='tab20', s=30, alpha=0.6, edgecolor='black', linewidth=0.5
            )
            ax.set_xlabel('PCA Component 1')
            ax.set_ylabel('PCA Component 2')
            ax.set_title('Clusters Visualized on First 2 PCA Components')
            plt.colorbar(scatter, ax=ax, label='Cluster ID')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Cluster Sizes")
            cluster_sizes = df['kmeans_cluster'].value_counts().sort_index()
            st.dataframe(cluster_sizes, use_container_width=True)
    
    # ========== TAB 2: CLUSTER DETAILS ==========
    with tab2:
        st.subheader("Explore Cluster Details")
        selected_cluster = st.selectbox("Select a cluster to explore:", sorted(df['kmeans_cluster'].unique()))
        
        cluster_data = df[df['kmeans_cluster'] == selected_cluster]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cluster Size", len(cluster_data))
        with col2:
            st.metric("Cluster Label", cluster_labels.get(selected_cluster, "N/A")[:40])
        with col3:
            sentiment_dist = cluster_data['airline_sentiment'].value_counts()
            st.metric("Dominant Sentiment", sentiment_dist.index[0] if len(sentiment_dist) > 0 else "N/A")
        
        st.subheader("Representative Tweets")
        # Show top tweets closest to centroid
        centroid = kmeans.cluster_centers_[selected_cluster]
        inds = np.where(kmeans_labels == selected_cluster)[0]
        dists = np.linalg.norm(embeddings_pca[inds] - centroid, axis=1)
        closest_idx = inds[np.argsort(dists)[:10]]
        
        for i, idx in enumerate(closest_idx, 1):
            with st.expander(f"🐦 Tweet {i} (Sentiment: {df.loc[idx, 'airline_sentiment']})"):
                st.write(df.loc[idx, 'cleaned_text'])
        
        st.subheader("Sentiment Distribution in Cluster")
        sentiment_counts = cluster_data['airline_sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 4))
        sentiment_counts.plot(kind='bar', ax=ax, color=['red', 'gray', 'green'][:len(sentiment_counts)])
        ax.set_title(f"Sentiment Distribution - Cluster {selected_cluster}")
        ax.set_ylabel("Count")
        ax.set_xlabel("Sentiment")
        st.pyplot(fig)
    
    # ========== TAB 3: DBSCAN ANALYSIS ==========
    with tab3:
        st.subheader("DBSCAN Clustering Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Number of Clusters", n_clusters_db)
        with col2:
            st.metric("Noise Points", n_noise)
        with col3:
            noise_pct = (n_noise / len(df)) * 100
            st.metric("Noise %", f"{noise_pct:.1f}%")
        with col4:
            st.metric("Largest Cluster Size", df['dbscan_cluster'].value_counts().max() if len(df) > 0 else 0)
        
        st.subheader("DBSCAN Cluster Distribution")
        dbscan_counts = df['dbscan_cluster'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ['red' if idx == -1 else 'C0' for idx in dbscan_counts.index]
        ax.bar(dbscan_counts.index, dbscan_counts.values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Cluster ID (Red = Noise)')
        ax.set_ylabel('Number of Tweets')
        ax.set_title('DBSCAN Cluster Distribution')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        
        st.subheader("K-Distance Graph (EPS Selection)")
        k_neighbors = st.slider("K for k-distance graph:", 10, 100, 50)
        nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(embeddings_pca)
        distances, _ = nbrs.kneighbors(embeddings_pca)
        distance_desc = sorted(distances[:, k_neighbors - 1], reverse=True)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(distance_desc, linewidth=1)
        ax.axhline(y=eps_value, color='r', linestyle='--', label=f'Current EPS={eps_value}')
        ax.set_title('K-Distance Graph (Finding the Elbow for EPS)')
        ax.set_ylabel('Distance')
        ax.set_xlabel('Points sorted by distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # ========== TAB 4: EXPORT DATA ==========
    with tab4:
        st.subheader("Download Results")
        
        # Prepare export dataframe
        export_df = df[['text', 'cleaned_text', 'airline_sentiment', 'kmeans_cluster', 'cluster_label', 'dbscan_cluster']].copy()
        
        # CSV download
        csv_buffer = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results (CSV)",
            data=csv_buffer,
            file_name="cluster_analysis_results.csv",
            mime="text/csv",
        )
        
        # Cluster summary
        st.subheader("Cluster Summary")
        summary_data = []
        for cl in sorted(df['kmeans_cluster'].unique()):
            cluster_df = df[df['kmeans_cluster'] == cl]
            summary_data.append({
                'Cluster': cl,
                'Size': len(cluster_df),
                'Label': cluster_labels.get(cl, "N/A")[:50],
                'Sentiment': cluster_df['airline_sentiment'].mode()[0] if len(cluster_df) > 0 else 'N/A',
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Download summary
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Cluster Summary (CSV)",
            data=summary_csv,
            file_name="cluster_summary.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
