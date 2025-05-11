import streamlit as st
import pandas as pd
import numpy as np
import os
import random

# Create data directory if it doesn't exist
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="Product Recommendation System (Emergency Mode)",
    page_icon="🛒",
    layout="wide"
)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{np.random.randint(10000, 99999)}"

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
    
if 'current_product' not in st.session_state:
    st.session_state.current_product = None

if 'current_view' not in st.session_state:
    st.session_state.current_view = 'all_products'

if 'product_cache' not in st.session_state:
    st.session_state.product_cache = None

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
    </style>
    """, unsafe_allow_html=True)

# Simplified function to load data
@st.cache_data
def load_data(max_products=1000):
    meta_path = os.path.join(data_dir, "out_Meta.json")
    reviews_path = os.path.join(data_dir, "out_Appliances5.json")
    
    try:
        # Check if files exist
        if not os.path.exists(meta_path) or not os.path.exists(reviews_path):
            st.error("Required data files not found.")
            return None, None
            
        # Count lines in meta file to determine sampling
        with open(meta_path, 'r') as f:
            for i, _ in enumerate(f):
                if i >= max_products:
                    break
            total_lines = min(i + 1, max_products)
        
        # Load only a sample of the meta data
        meta_df = pd.read_json(meta_path, lines=True, nrows=total_lines)
        
        # Load a small sample of reviews
        reviews_df = pd.read_json(reviews_path, lines=True, nrows=1000)
        
        # Filter reviews to match products in meta_df
        reviews_df = reviews_df[reviews_df['asin'].isin(meta_df['asin'])]
        
        return meta_df, reviews_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Simple function to get product recommendations
def get_simple_recommendations(meta_df, product_asin=None, n=6):
    if product_asin is None:
        # Return random products
        return meta_df.sample(min(n, len(meta_df)))
    
    # Find the product
    product = meta_df[meta_df['asin'] == product_asin]
    if product.empty:
        return meta_df.sample(min(n, len(meta_df)))
    
    # Try to find products with the same category
    if 'category' in meta_df.columns:
        category = product['category'].iloc[0]
        same_category = meta_df[(meta_df['category'] == category) & (meta_df['asin'] != product_asin)]
        if len(same_category) >= n:
            return same_category.sample(min(n, len(same_category)))
    
    # Fallback to random selection
    other_products = meta_df[meta_df['asin'] != product_asin]
    return other_products.sample(min(n, len(other_products)))

# Function to create a product card
def create_product_card(product, col, index):
    asin = product.get('asin', 'Unknown')
    title = product.get('title', 'Unknown Product')
    price = product.get('price', 'Price unavailable')
    description = product.get('description', 'No description available')
    
    # Format price
    if isinstance(price, (int, float)):
        price_display = f"${float(price):.2f}"
    else:
        price_display = price if isinstance(price, str) else "Price unavailable"
    
    # Truncate description
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

# Function to show all products
def show_all_products(meta_df, limit=30):
    st.markdown("<h1 style='text-align: center;'>All Products</h1>", unsafe_allow_html=True)
    
    # Get a sample of products
    products = meta_df.sample(min(limit, len(meta_df)))
    
    # Display in grid
    cols = st.columns(3)
    for i, (_, product) in enumerate(products.iterrows()):
        create_product_card(product.to_dict(), cols[i % 3], f"all_{i}")

# Function to show product details
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
    if isinstance(price, (int, float)):
        price_display = f"${float(price):.2f}"
    else:
        price_display = "Price unavailable"
    
    # Create two columns for info
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://via.placeholder.com/300x300?text=Product+Image", use_column_width=True)
    
    with col2:
        st.title(title)
        st.markdown(f"<div style='font-size: 24px; color: #e63946; font-weight: bold;'>{price_display}</div>", unsafe_allow_html=True)
        st.write("Product ID:", asin)
    
    # Product description
    st.markdown("### Product Description")
    st.markdown(f"""
    <div style='padding: 20px; background-color: #f8f9fa; border-radius: 8px; margin-bottom: 30px;'>
        {description}
    </div>
    """, unsafe_allow_html=True)
    
    # Reviews
    st.markdown("### Customer Reviews")
    product_reviews = reviews_df[reviews_df['asin'] == asin]
    
    if product_reviews.empty:
        st.info("No reviews available for this product.")
    else:
        for i, (_, review) in enumerate(product_reviews.head(3).iterrows()):
            reviewer_name = review.get('reviewerName', 'Anonymous')
            review_text = review.get('reviewText', 'No review text')
            rating = review.get('overall', 0)
            
            st.markdown(f"""
            <div style='border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                    <div style='font-weight: bold;'>{reviewer_name}</div>
                    <div>{'⭐' * int(rating)}</div>
                </div>
                <div>{review_text}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("### You May Also Like")
    recommendations = get_simple_recommendations(meta_df, asin, 6)
    
    # Display in grid
    cols = st.columns(3)
    for i, (_, rec) in enumerate(recommendations.iterrows()):
        create_product_card(rec.to_dict(), cols[i % 3], f"rec_{i}")
    
    # Back button
    if st.button("← Back to Products"):
        st.session_state.current_page = 'home'
        st.rerun()

# Main app function
def main():
    # Load CSS
    load_css()
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        
        if st.button("All Products"):
            st.session_state.current_page = 'home'
            st.rerun()
        
        st.info("Emergency Mode Active: Using simplified recommendation system due to memory constraints.")
        
        # Sample size slider
        sample_size = st.slider("Max products to load", 
                               min_value=100, 
                               max_value=5000, 
                               value=1000,
                               step=100,
                               help="Lower values use less memory")
        
        if st.button("Apply"):
            # Clear cache to reload with new settings
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.write(f"User ID: {st.session_state.user_id}")
    
    # Check if we've already loaded data
    if st.session_state.product_cache is None:
        with st.spinner("Loading product data..."):
            meta_df, reviews_df = load_data(max_products=1000)
            if meta_df is None:
                st.error("Failed to load data. Please check your data files.")
                return
            
            # Cache the data
            st.session_state.product_cache = (meta_df, reviews_df)
    else:
        meta_df, reviews_df = st.session_state.product_cache
    
    # Main content
    if st.session_state.current_page == 'home':
        show_all_products(meta_df)
    elif st.session_state.current_page == 'product_detail':
        show_product_detail(meta_df, reviews_df)

if __name__ == "__main__":
    main()