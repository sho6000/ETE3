import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64

class TextAnalysis:
    def __init__(self, df):
        self.df = df
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Extract unique tracks as topics
        self.topics = sorted(self.df['Track'].unique())
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def calculate_similarity(self, topic1, topic2):
        # Get feedback for each topic
        feedback1 = self.df[self.df['Track'] == topic1]['Feedback'].dropna()
        feedback2 = self.df[self.df['Track'] == topic2]['Feedback'].dropna()
        
        if len(feedback1) == 0 or len(feedback2) == 0:
            return 0.0
        
        # Preprocess feedback
        processed_feedback1 = [self.preprocess_text(text) for text in feedback1]
        processed_feedback2 = [self.preprocess_text(text) for text in feedback2]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        
        # Combine and transform feedback
        all_feedback = processed_feedback1 + processed_feedback2
        tfidf_matrix = vectorizer.fit_transform(all_feedback)
        
        # Calculate average similarity between all pairs
        similarities = []
        for i in range(len(processed_feedback1)):
            for j in range(len(processed_feedback2)):
                sim = cosine_similarity(
                    tfidf_matrix[i:i+1],
                    tfidf_matrix[len(processed_feedback1)+j:len(processed_feedback1)+j+1]
                )[0][0]
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def display(self):
        st.subheader("Text Analysis Options")
        
        # Analysis type selection
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Word Cloud", "Topic Similarity Analysis"]
        )
        
        if analysis_type == "Word Cloud":
            self.display_word_cloud()
        else:
            self.display_similarity_analysis()
    
    def display_word_cloud(self):
        st.subheader("Word Cloud Generation")
        
        # Track selection
        selected_track = st.selectbox(
            "Select Track",
            options=["All"] + list(self.df['Track'].unique())
        )
        
        # Filter data based on track
        if selected_track == "All":
            feedback_data = self.df['Feedback'].dropna()
        else:
            feedback_data = self.df[self.df['Track'] == selected_track]['Feedback'].dropna()
        
        if len(feedback_data) == 0:
            st.warning("No feedback data available for the selected track.")
            return
        
        # Generate word cloud
        text = ' '.join(feedback_data)
        processed_text = self.preprocess_text(text)
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            max_font_size=100,
            random_state=42,
            collocations=False
        ).generate(processed_text)
        
        # Display word cloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        # Display word frequencies
        st.subheader("Most Common Words")
        words = processed_text.split()
        word_freq = pd.Series(words).value_counts().head(10)
        
        fig = go.Figure(data=[
            go.Bar(
                x=word_freq.index,
                y=word_freq.values,
                marker_color='#1f77b4'
            )
        ])
        fig.update_layout(
            title="Top 10 Most Frequent Words",
            xaxis_title="Word",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def display_similarity_analysis(self):
        st.subheader("Topic Similarity Analysis")
        
        # Create similarity matrix
        similarity_matrix = np.zeros((len(self.topics), len(self.topics)))
        for i in range(len(self.topics)):
            for j in range(len(self.topics)):
                similarity_matrix[i, j] = self.calculate_similarity(self.topics[i], self.topics[j])
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=self.topics,
            y=self.topics,
            colorscale='Viridis',
            text=np.round(similarity_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False,
            colorbar=dict(
                title="Similarity Score",
                thickness=15,
                len=0.7,
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                outlinewidth=1,
                outlinecolor="black",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        ))
        
        fig.update_layout(
            title="Topic Similarity Matrix",
            xaxis_title="Topics",
            yaxis_title="Topics",
            height=600,
            width=800,
            margin=dict(l=100, r=100, t=100, b=100)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display topic relationships
        st.subheader("Topic Relationships")
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for i, topic in enumerate(self.topics):
            G.add_node(topic, size=20)
        
        # Add edges with similarity scores
        for i in range(len(self.topics)):
            for j in range(i+1, len(self.topics)):
                if similarity_matrix[i, j] > 0.3:  # Only show significant relationships
                    G.add_edge(self.topics[i], self.topics[j], 
                             weight=similarity_matrix[i, j])
        
        # Get node positions using spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Add edge positions
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        # Create node trace
        node_trace = go.Scatter(
            x=[],
            y=[],
            mode='markers+text',
            hoverinfo='text',
            text=[],
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=20,
                colorbar=dict(
                    title="Node Size",
                    thickness=15,
                    len=0.7,
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01,
                    outlinewidth=1,
                    outlinecolor="black",
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
            )
        )
        
        # Add node positions
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([f"Topic: {node}"])
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title="Topic Relationship Network",
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[dict(
                               text="",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0, y=0
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        st.plotly_chart(fig, use_container_width=True)

# Main Streamlit app
def main():
    st.title("Feedback Text Analysis")
    
    # Load data
    @st.cache_data
    def load_data():
        return pd.read_csv('poster_presentation_data.csv')
    
    df = load_data()
    
    # Create TextAnalysis instance
    text_analysis = TextAnalysis(df)
    
    # Display text analysis options
    text_analysis.display()

if __name__ == "__main__":
    main() 