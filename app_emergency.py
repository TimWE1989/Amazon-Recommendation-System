import streamlit as st
import pandas as pd
import numpy as np
import os
import gdown
import re
import random
from html import unescape

# Set page configuration
st.set_page_config(
    page_title="Product Recommendation System (Emergency Mode)",
    page_icon="🛒",
    layout="wide"
)

# Create a data directory
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Initialize session state for user tracking
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{np.random.randint(10000, 99999)}"
    
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
    
if 'current_product' not in st.session_state:
    st.session_state.current_product = None

if 'products_loaded' not in st.session_state:
    st.session_state.products_loaded = False

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .product-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        transition: transform 0.3s, box-shadow 0.3s;
        height: 100%;
        display: flex;
        flex-direction: column;
        background-color: white;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .product-title {
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 8px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .product-price {
        color: #e63946;
        font-weight: bold;
        font-size: 1.2em;
        margin: 8px 0;
    }
    .product-description {
        color: #555;
        font-size: 0.9em;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        flex-grow: 1;
    }
    .product-detail-header {
        margin-bottom: 30px;
    }
    .product-detail-title {
        font-size: 2em;
        margin-bottom: 10px;
    }
    .product-detail-price {
        color: #e63946;
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .product-detail-description {
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 30px;
    }
    .main-title {
        text-align: center;
        color: #1E3A8A;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to download from Google Drive
def download_from_gdrive(file_id, output_path):
    """Download a file from Google Drive."""
    if os.path.exists(output_path):
        return output_path
    
    with st.spinner(f"Downloading file to {output_path}..."):
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
            
            if os.path.exists(output_path):
                st.success(f"Successfully downloaded file to {output_path}")
                return output_path
            else:
                st.error(f"Failed to download file to {output_path}")
                return None
        except Exception as e:
            st.error(f"Error downloading file: {e}")
            return None

# Load data only once
@st.cache_data
def load_data(max_products=500):
    meta_path = os.path.join(data_dir, "out_Meta.json")
    reviews_path = os.path.join(data_dir, "out_Appliances5.json")
    
    # Download files if they don't exist
    if not os.path.exists(meta_path):
        meta_file_id = "1H4V_ZimSgxnM2oD3zyUsVVh278JeC_g1"
        meta_path = download_from_gdrive(meta_file_id, meta_path)
        
    if not os.path.exists(reviews_path):
        reviews_file_id = "1eqHSIaqMdo9c8dcUnMVSIEw28H6uAbZv"
        reviews_path = download_from_gdrive(reviews_file_id, reviews_path)
    
    if not meta_path or not reviews_path:
        return None, None
    
    try:
        # Load a limited number of products to save memory
        meta_df = pd.read_json(meta_path, lines=True, nrows=max_products)
        
        # Basic preprocessing
        if 'price' in meta_df.columns:
            meta_df['price'] = pd.to_numeric(meta_df['price'], errors='coerce')
        
        # Clean up categories if they exist
        if 'category' in meta_df.columns:
            meta_df['category'] = meta_df['category'].apply(
                lambda x: unescape(str(x)).replace('&amp;','&').strip()
            )
        
        # Load some reviews, but only for products we have
        reviews_df = pd.read_json(reviews_path, lines=True, nrows=1000)
        reviews_df = reviews_df[reviews_df['asin'].isin(meta_df['asin'])]
        
        return meta_df, reviews_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Simple recommendation function
def get_simple_recommendations(meta_df, product_asin=None, n=6):
    if product_asin is None:
        # Return random products
        return meta_df.sample(min(n, len(meta_df)))
    
    # Find the product
    product = meta_df[meta_df['asin'] == product_asin]
    if product.empty:
        return meta_df.sample(min(n, len(meta_df)))
    
    # Try to find similar products by category
    if 'category' in meta_df.columns:
        category = product['category'].iloc[0]
        same_category = meta_df[(meta_df['category'] == category) & 
                                (meta_df['asin'] != product_asin)]
        if len(same_category) >= n:
            return same_category.sample(min(n, len(same_category)))
    
    # Try by brand if available
    if 'brand' in meta_df.columns:
        brand = product['brand'].iloc[0]
        same_brand = meta_df[(meta_df['brand'] == brand) & 
                             (meta_df['asin'] != product_asin)]
        if len(same_brand) >= n:
            return same_brand.sample(min(n, len(same_brand)))
    
    # Fallback to random
    return meta_df[meta_df['asin'] != product_asin].sample(min(n, len(meta_df)-1))

# Function to create a product card
def create_product_card(product, col, index):
    asin = product.get('asin', 'Unknown')
    title = product.get('title', 'Unknown Product')
    price = product.get('price', 'Price unavailable')
    description = product.get('description', 'No description available')
    
    # Format price
    if isinstance(price, (int, float)) and not pd.isna(price):
        price_display = f"${float(price):.2f}"
    else:
        price_display = "Price unavailable"
    
    # Truncate description if too long
    if isinstance(description, str) and len(description) > 100:
        desc_display = description[:100] + "..."
    else:
        desc_display = description
    
    # Create card
    with col:
        st.markdown(f"""
        <div class="product-card">
            <div class="product-title">{title}</div>
            <div class="product-price">{price_display}</div>
            <div class="product-description">{desc_display}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add button
        button_key = f"view_{asin}_{index}"
        if st.button(f"View Details", key=button_key):
            st.session_state.current_product = asin
            st.session_state.current_page = 'product_detail'
            st.rerun()

# Show all products
def show_all_products(meta_df):
    st.markdown("<h1 class='main-title'>All Products</h1>", unsafe_allow_html=True)
    
    # Get a sample of products
    if len(meta_df) > 30:
        products = meta_df.sample(30)
    else:
        products = meta_df
    
    # Display in grid (3 columns)
    cols = st.columns(3)
    for i, (_, product) in enumerate(products.iterrows()):
        create_product_card(product.to_dict(), cols[i % 3], f"all_{i}")

# Show product details
def show_product_detail(meta_df, reviews_df):
    product = meta_df[meta_df['asin'] == st.session_state.current_product]
    
    if product.empty:
        st.error("Product not found!")
        return
    
    # Get product info
    asin = product['asin'].iloc[0]
    title = product.get('title', 'Unknown Product').iloc[0]
    price = product.get('price', 0).iloc[0]
    description = product.get('description', 'No description available').iloc[0] 
    
    # Format price
    if isinstance(price, (int, float)) and not pd.isna(price):
        price_display = f"${float(price):.2f}"
    else:
        price_display = "Price unavailable"
    
    # Create two columns for image and details
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://via.placeholder.com/400x400?text=Product+Image", use_column_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="product-detail-header">
            <div class="product-detail-title">{title}</div>
            <div class="product-detail-price">{price_display}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Product description
    st.markdown("### Product Description")
    st.markdown(f"""
    <div class="product-detail-description">
        {description}
    </div>
    """, unsafe_allow_html=True)
    
    # Show reviews if available
    product_reviews = reviews_df[reviews_df['asin'] == asin] if reviews_df is not None else pd.DataFrame()
    
    if not product_reviews.empty:
        st.markdown("### Customer Reviews")
        for _, review in product_reviews.head(3).iterrows():
            reviewer = review.get('reviewerName', 'Anonymous')
            rating = review.get('overall', 0)
            review_text = review.get('reviewText', '')
            
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <div style="font-weight: bold;">{reviewer}</div>
                    <div>{'⭐' * int(rating)}</div>
                </div>
                <div>{review_text}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Similar products
    st.markdown("### You May Also Like")
    recommendations = get_simple_recommendations(meta_df, asin, 6)
    
    cols = st.columns(3)
    for i, (_, product) in enumerate(recommendations.iterrows()):
        create_product_card(product.to_dict(), cols[i % 3], f"rec_{i}")
    
    # Back button
    if st.button("← Back to Products"):
        st.session_state.current_page = 'home'
        st.rerun()

# Main function
def main():
    # Load CSS
    load_css()
    
    # Create sidebar
    with st.sidebar:
        st.title("Navigation")
        
        if st.button("All Products"):
            st.session_state.current_page = 'home'
            st.rerun()
        
        # Settings
        st.markdown("---")
        st.subheader("Settings")
        
        max_products = st.slider("Max products to load", 
                               min_value=100, 
                               max_value=5000, 
                               value=500,
                               step=100)
        
        if st.button("Apply Settings"):
            st.cache_data.clear()
            st.session_state.products_loaded = False
            st.rerun()
        
        st.markdown("---")
        st.info("EMERGENCY MODE: Using simplified recommendation system.")
        st.write(f"User ID: {st.session_state.user_id}")
    
    # Load data
    if not st.session_state.products_loaded:
        with st.spinner("Loading product data..."):
            meta_df, reviews_df = load_data(max_products)
            if meta_df is None:
                st.error("Failed to load data. Please check your data files.")
                return
            
            st.session_state.meta_df = meta_df
            st.session_state.reviews_df = reviews_df
            st.session_state.products_loaded = True
    else:
        meta_df = st.session_state.meta_df
        reviews_df = st.session_state.reviews_df
    
    # Main content
    if st.session_state.current_page == 'home':
        show_all_products(meta_df)
    elif st.session_state.current_page == 'product_detail':
        show_product_detail(meta_df, reviews_df)

if __name__ == "__main__":
    main()