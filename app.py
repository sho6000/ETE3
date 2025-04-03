import streamlit as st

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="National Poster Presentation Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from text_analysis import TextAnalysis
from image_processor import ImageProcessor
from dashboard import Dashboard

# Custom CSS for consistent styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    .stSlider {
        margin-bottom: 1rem;
    }
    .stTextInput {
        margin-bottom: 1rem;
    }
    .stMarkdown {
        margin-bottom: 1rem;
    }
    .stSubheader {
        color: #1f77b4;
        margin-top: 2rem;
    }
    .stHeader {
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_data
def load_data():
    return pd.read_csv('poster_presentation_data.csv')

def main():
    # Sidebar navigation
    
    
    # Add logo or branding
    st.sidebar.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2>National Poster Presentation</h2>
            <p style='color: #666;'>Analytics Dashboard</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation options
    page = st.sidebar.radio(
        "Select Section",
        ["Dashboard", "Text Analysis", "Image Gallery"],
        index=0
    )
    
    # Load data
    df = load_data()
    
    # Add footer with contact info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style='text-align: center; margin-top: 2rem;'>
            <p style='color: #666;'>Contact Support</p>
            <p style='font-size: 0.8rem;'>support@posterpresentation.com</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main content area
    if page == "Dashboard":
        st.title("📊 Analytics Dashboard")
        st.markdown("""
            <div style='background-color: #000000; padding: 1rem; border-radius: 4px; margin-bottom: 2rem;'>
                <p>Welcome to the National Poster Presentation Analytics Dashboard. 
                Use the filters below to explore participation patterns and trends.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Initialize and display dashboard
        dashboard = Dashboard(df)
        dashboard.display()
        
    elif page == "Text Analysis":
        st.title("📝 Feedback Analysis")
        st.markdown("""
            <div style='background-color: #000000; padding: 1rem; border-radius: 4px; margin-bottom: 2rem;'>
                <p>Analyze participant feedback using advanced text processing techniques. 
                Generate word clouds and explore feedback patterns.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Initialize and display text analysis
        text_analyzer = TextAnalysis(df)
        text_analyzer.display()
        
    else:  # Image Gallery
        st.title("🖼️ Image Gallery")
        st.markdown("""
            <div style='background-color: #000000; padding: 1rem; border-radius: 4px; margin-bottom: 2rem;'>
                <p>Browse and process event-related images. Apply filters, add watermarks, 
                and download processed images.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Initialize and display image processor
        image_processor = ImageProcessor(df)
        image_processor.display()

if __name__ == "__main__":
    main() 