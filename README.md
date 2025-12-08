# Thematic-Analysis-of-Customer-Feedback
# Thematic Analysis of Customer Feedback with GenAI ✈️🤖


## 📌 Project Overview
This project aims to automate the discovery and labeling of themes in unlabeled customer feedback. By integrating **Transformer-based sentence embeddings (SBERT)** with **unsupervised clustering (KMeans/DBSCAN)** and **Generative AI**, we created a pipeline that not only groups similar customer complaints but also automatically generates human-readable titles and summaries for those groups.

This approach replaces manual topic modeling with a semantic, vector-based approach, allowing for the detection of granular issues (e.g., "Lost Luggage" vs. "Damaged Luggage") without human intervention.

## 🛠️ Methodology Pipeline

The project follows a linear unsupervised learning pipeline:

1.  **Data Ingestion:** Loading the **Twitter U.S. Airline Sentiment** dataset (14k tweets).
2.  **Preprocessing:** * Cleaning text (removing handles like `@United`, special characters).
    * Standardization (lowercasing).
3.  **Embeddings:** * Model: `all-MiniLM-L6-v2` (Sentence-BERT).
    * Output: 384-dimensional dense vectors.
4.  **Dimensionality Reduction:** * Technique: **PCA** (Principal Component Analysis).
    * Goal: Reduced dimensions to preserve 95% variance to stabilize clustering.
5.  **Clustering:**
    * **KMeans:** Used for forcing data into discrete topics ($k=17$).
    * **DBSCAN:** Used for density-based clustering to isolate noise ($eps=0.8$, $min\_samples=15$).
6.  **GenAI Labeling:**
    * Model: `sshleifer/distilbart-cnn-12-6`.
    * Technique: Summarizing representative samples from each cluster to generate a "Topic Title."
7  **Streamlit Frontend:**
    * Goal: Build Streamlit frontend for visualization.
---

## 📊 Key Findings

### Clustering Performance
* **KMeans:** Best performance at **k=17**. This allowed for granular separation of specific airline issues.
* **DBSCAN:** Tuned using the K-Distance "Elbow Method."
    * Optimal Epsilon ($eps$): **0.8**
    * Noise Handling: Successfully identified and removed ~20% of tweets as "noise" (irrelevant or gibberish text), resulting in cleaner topic definitions compared to KMeans.

### Discovered Themes
The GenAI pipeline successfully identified distinct categories such as:
* *Late Flight / Missed Connections*
* *Lost Luggage & Baggage Claims*
* *Customer Service Wait Times*
* *Booking & Website Errors*

## 💻 Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed.

### 1. Install Dependencies
Run the script ./install_requirements.sh