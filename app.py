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
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    # Add logo or branding
    st.sidebar.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2>National Poster Presentation</h2>
            <p style='color: #666;'>Analytics Dashboard</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation options with emojis
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Dataset Overview", "📈 Participation Trends", "💬 Feedback Analysis", "🖼 Image Processing"],
        index=0
    )
    
    # Load data
    df = load_data()
    
    # Add footer with contact info
    st.sidebar.markdown("---")
    # st.sidebar.markdown("""
    #     <div style='text-align: center; margin-top: 2rem;'>
    #         <p style='color: #666;'>Contact Support</p>
    #         <p style='font-size: 0.8rem;'>support@posterpresentation.com</p>
    #     </div>
    # """, unsafe_allow_html=True)
    
    # Main content area
    if page == "🏠 Dataset Overview":
        st.title("📋 Dataset View")
        
        
        # Display dataset information
        st.subheader("Dataset Information")
        st.write(f"Total Rows: {len(df)}")
        st.write(f"Total Columns: {len(df.columns)}")
        st.write(f"Columns: {', '.join(df.columns)}")
        
        # Display first 20 rows
        st.subheader("First 20 Rows")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Display data types
        st.subheader("Data Types")
        st.dataframe(pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes
        }), use_container_width=True)
        
    elif page == "📈 Participation Trends":
        st.title("📊 Analytics Dashboard")
        
        
        # Initialize and display dashboard
        dashboard = Dashboard(df)
        dashboard.display()
        
    elif page == "💬 Feedback Analysis":
        st.title("📝 Feedback Analysis")
        
        
        # Initialize and display text analysis
        text_analyzer = TextAnalysis(df)
        text_analyzer.display()
        
    else:  # Image Processing
        st.title("🖼️ Image Gallery")
        
        
        # Initialize and display image processor
        image_processor = ImageProcessor(df)
        image_processor.display()

if __name__ == "__main__":
    main() 