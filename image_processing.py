import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
from datetime import datetime
import base64
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Poster Presentation Image Gallery",
    page_icon="🖼️",
    layout="wide"
)

# Create directories if they don't exist
def create_directories():
    # Create main uploads directory
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    
    # Create day-wise directories
    for day in range(1, 5):
        day_dir = f"uploads/day_{day}"
        if not os.path.exists(day_dir):
            os.makedirs(day_dir)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('poster_presentation_data.csv')

# Image processing functions
def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)

def add_watermark(image, text, position=(10, 10), font_size=30, color=(255, 255, 255, 128)):
    # Convert OpenCV image to PIL
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Create a copy of the image
    watermarked = image.copy()
    
    # Create a drawing context
    draw = ImageDraw.Draw(watermarked, 'RGBA')
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Add text watermark
    draw.text(position, text, font=font, fill=color)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(watermarked), cv2.COLOR_RGB2BGR)

# Function to get image files from a directory
def get_image_files(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []
    
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(directory, file))
    
    return image_files

# Function to display image with caption
def display_image_with_caption(image_path, caption):
    try:
        image = Image.open(image_path)
        st.image(image, caption=caption, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")

# Function to create a download link for processed images
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Main Streamlit app
def main():
    st.title("Poster Presentation Image Gallery")
    
    # Create necessary directories
    create_directories()
    
    # Load data
    df = load_data()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Image Gallery", "Image Processing"])
    
    if page == "Image Gallery":
        display_gallery(df)
    else:
        display_image_processor(df)

def display_gallery(df):
    st.header("Day-wise Image Gallery")
    
    # Day selection
    selected_day = st.selectbox("Select Day", [1, 2, 3, 4])
    
    # Get images for selected day
    day_dir = f"uploads/day_{selected_day}"
    image_files = get_image_files(day_dir)
    
    if not image_files:
        st.info(f"No images available for Day {selected_day}. Upload some images in the Image Processing section.")
    else:
        # Display images in a grid
        cols = st.columns(3)
        for i, img_path in enumerate(image_files):
            with cols[i % 3]:
                # Extract filename without extension
                filename = os.path.basename(img_path)
                name_without_ext = os.path.splitext(filename)[0]
                
                # Try to find matching track from filename
                track = "Unknown Track"
                for t in df['Track'].unique():
                    if t.lower() in name_without_ext.lower():
                        track = t
                        break
                
                # Display image with caption
                display_image_with_caption(img_path, f"Day {selected_day} - {track}")

def display_image_processor(df):
    st.header("Image Processing")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "gif", "bmp"])
    
    if uploaded_file is not None:
        # Display original image
        st.subheader("Original Image")
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Original Image", use_container_width=True)
        
        # Convert to OpenCV format for processing
        opencv_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        
        # Processing options
        st.subheader("Processing Options")
        
        # Day selection for saving
        selected_day = st.selectbox("Select Day for Saving", [1, 2, 3, 4])
        
        # Track selection for caption
        selected_track = st.selectbox("Select Track", df['Track'].unique())
        
        # Initialize session state for processed image
        if 'processed_image' not in st.session_state:
            st.session_state.processed_image = opencv_image
            st.session_state.filter_applied = False
            st.session_state.watermark_applied = False
            st.session_state.filter_type = None
        
        # Processing filters
        st.subheader("Apply Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Apply Grayscale"):
                st.session_state.processed_image = apply_grayscale(opencv_image)
                st.session_state.filter_applied = True
                st.session_state.filter_type = "grayscale"
                st.image(st.session_state.processed_image, caption="Grayscale", use_container_width=True)
        
        with col2:
            if st.button("Apply Blur"):
                st.session_state.processed_image = apply_blur(opencv_image)
                st.session_state.filter_applied = True
                st.session_state.filter_type = "blur"
                st.image(st.session_state.processed_image, caption="Blurred", use_container_width=True)
        
        with col3:
            if st.button("Apply Edge Detection"):
                st.session_state.processed_image = apply_edge_detection(opencv_image)
                st.session_state.filter_applied = True
                st.session_state.filter_type = "edge"
                st.image(st.session_state.processed_image, caption="Edge Detection", use_container_width=True)
        
        # Watermark options
        st.subheader("Add Watermark")
        watermark_text = st.text_input("Watermark Text", "National Poster Presentation")
        watermark_position = st.slider("Watermark Position (X)", 10, 500, 10)
        watermark_opacity = st.slider("Watermark Opacity", 0, 255, 128)
        
        if st.button("Apply Watermark"):
            # Apply watermark to the current processed image
            watermarked_image = add_watermark(
                st.session_state.processed_image, 
                watermark_text, 
                position=(watermark_position, 10), 
                color=(255, 255, 255, watermark_opacity)
            )
            st.session_state.processed_image = watermarked_image
            st.session_state.watermark_applied = True
            st.image(watermarked_image, caption="With Watermark", use_container_width=True)
        
        # Display current processed image
        if st.session_state.filter_applied or st.session_state.watermark_applied:
            st.subheader("Current Processed Image")
            st.image(st.session_state.processed_image, caption="Current Processed Image", use_container_width=True)
        
        # Save and download options
        st.subheader("Save and Download")
        
        # Save button
        if st.button("Save Processed Image"):
            save_processed_image(st.session_state.processed_image, selected_day, selected_track, 
                               f"{st.session_state.filter_type if st.session_state.filter_applied else 'original'}_watermarked" if st.session_state.watermark_applied else 
                               st.session_state.filter_type if st.session_state.filter_applied else "original")
        
        # Download button
        if st.session_state.filter_applied or st.session_state.watermark_applied:
            st.subheader("Download Processed Image")
            # Convert to PIL for download
            if len(st.session_state.processed_image.shape) == 2:  # Grayscale
                processed_pil = Image.fromarray(st.session_state.processed_image)
            else:
                processed_pil = Image.fromarray(cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB))
            
            # Create filename based on applied filters
            filename = f"{selected_track}_"
            if st.session_state.filter_applied:
                filename += f"{st.session_state.filter_type}_"
            if st.session_state.watermark_applied:
                filename += "watermarked_"
            filename += f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            
            st.markdown(get_image_download_link(processed_pil, filename, "Download Processed Image"), unsafe_allow_html=True)

def save_processed_image(image, day, track, suffix):
    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{track}_{suffix}_{timestamp}.jpg"
    filepath = os.path.join(f"uploads/day_{day}", filename)
    
    # Save image
    cv2.imwrite(filepath, image)
    st.success(f"Image saved as {filename}")

if __name__ == "__main__":
    main() 