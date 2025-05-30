import traceback
import streamlit as st
import gdown
import pandas as pd
import numpy as np
import re
from html import unescape
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from numba import jit
from collections import defaultdict
import nltk_setup
import os
import json
import time

def optimize_memory(df):
    """Reduce memory usage of a DataFrame while being conservative with categoricals."""
    result_df = df.copy()
    
    # List of columns that shouldn't be converted to categorical
    # because they might need new values added later
    skip_categorical = ['main_cat', 'primary_category', 'brand', 'category', 'fine_category']
    
    for col in result_df.columns:
        # Skip columns with unhashable types like lists or dicts
        try:
            # Try to access first element to check type
            if len(result_df) > 0:
                first_val = result_df[col].iloc[0]
                # Skip if the value is unhashable (list, dict, etc.)
                if isinstance(first_val, (list, dict, set)):
                    continue
            
            if result_df[col].dtype == 'object':
                # Skip columns that might need new categories added
                if col in skip_categorical:
                    continue
                    
                # Try to get unique values, but handle unhashable types
                try:
                    unique_count = len(result_df[col].unique())
                    if unique_count / len(result_df) < 0.2:  # Even more conservative: only if less than 20% unique
                        result_df[col] = result_df[col].astype('category')
                except TypeError:
                    # If unhashable type, skip this column
                    continue
            elif result_df[col].dtype == 'float64':
                result_df[col] = pd.to_numeric(result_df[col], downcast='float')
            elif result_df[col].dtype == 'int64':
                result_df[col] = pd.to_numeric(result_df[col], downcast='integer')
        except Exception:
            # If any error occurs, just skip this column
            continue
            
    return result_df

# New function to download data from Google Drive
def download_from_gdrive(file_id, output_path):
    """Download a file from Google Drive using its file ID."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(output_path):
        st.info(f"Using cached file: {output_path}")
        return output_path
    
    # Show download status
    with st.spinner(f"Downloading file to {output_path}..."):
        try:
            # Create the Google Drive URL
            url = f"https://drive.google.com/uc?id={file_id}"
            
            # Download the file
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

# Create a data directory
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Google Drive file IDs (from your URLs)
meta_file_id = "1H4V_ZimSgxnM2oD3zyUsVVh278JeC_g1"
reviews_file_id = "1eqHSIaqMdo9c8dcUnMVSIEw28H6uAbZv"

# Set page configuration
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛒",
    layout="wide"
)

# Download datasets from Google Drive
meta_path = download_from_gdrive(meta_file_id, os.path.join(data_dir, "out_Meta.json"))
reviews_path = download_from_gdrive(reviews_file_id, os.path.join(data_dir, "out_Appliances5.json"))

if not meta_path or not reviews_path:
    st.error("Failed to download required datasets")
    st.stop()

# Load the datasets
try:
    with st.spinner("Loading datasets..."):
        df1 = pd.read_json(meta_path, lines=True)
        df2 = pd.read_json(reviews_path, lines=True)
        st.success(f"Successfully loaded {len(df1)} products and {len(df2)} reviews")
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()

# Add this function to your code
def ensure_data_files():
    """Ensure data files are downloaded before attempting to load them."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    meta_path = os.path.join(data_dir, "out_Meta.json")
    reviews_path = os.path.join(data_dir, "out_Appliances5.json")
    
    # Download meta data if needed
    if not os.path.exists(meta_path):
        with st.spinner("Downloading meta data..."):
            meta_file_id = "1H4V_ZimSgxnM2oD3zyUsVVh278JeC_g1"
            download_from_gdrive(meta_file_id, meta_path)
    
    # Download reviews if needed
    if not os.path.exists(reviews_path):
        with st.spinner("Downloading reviews data..."):
            reviews_file_id = "1eqHSIaqMdo9c8dcUnMVSIEw28H6uAbZv"
            download_from_gdrive(reviews_file_id, reviews_path)
    
    # Check if files exist after download attempts
    meta_exists = os.path.exists(meta_path)
    reviews_exist = os.path.exists(reviews_path)
    
    if not meta_exists or not reviews_exist:
        st.error("Failed to download required data files.")
        if not meta_exists:
            st.error(f"Meta data file not found at {meta_path}")
        if not reviews_exist:
            st.error(f"Reviews file not found at {reviews_path}")
        return False
    
    return True

# Add the download function if you don't already have it
def download_from_gdrive(file_id, output_path):
    """Download a file from Google Drive using its file ID."""
    if os.path.exists(output_path):
        st.info(f"File already exists at {output_path}. Using cached version.")
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

# Initialize session state for user tracking
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{np.random.randint(10000, 99999)}"
    
if 'clicked_products' not in st.session_state:
    st.session_state.clicked_products = []
    
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
    
if 'current_product' not in st.session_state:
    st.session_state.current_product = None

# Class definitions in proper dependency order
class DataPreprocessor:
    """Class to handle data preprocessing for all recommendation algorithms."""
    
    def __init__(self, meta_df, reviews_df):
        """Initialize with metadata and reviews dataframes."""
        self.meta_df = meta_df
        self.reviews_df = reviews_df
        
    def clean_category(self):
        """Clean and normalize category information."""
        df = self.meta_df.copy()
        
        # Check if 'category' column exists, use 'categories' if needed
        category_col = 'category' if 'category' in df.columns else 'categories'
        
        if category_col in df.columns:
            df[category_col] = df[category_col].apply(lambda x: unescape(str(x)).replace('&amp;','&').strip())
            
            # Extract finest category
            def fine_cat(t):
                parts = re.split(r',|\s>\s', str(t))
                last = parts[-1].strip()
                return last.split('&')[0].strip()
                
            df['fine_category'] = df[category_col].apply(fine_cat)
            
            # Handle main_cat column carefully
            if 'main_cat' in df.columns:
                # If already converted to category, handle differently
                if pd.api.types.is_categorical_dtype(df['main_cat']):
                    # Add 'Unknown' to categories if not already there
                    if 'Unknown' not in df['main_cat'].cat.categories:
                        df['main_cat'] = df['main_cat'].cat.add_categories(['Unknown'])
                    df['main_cat'] = df['main_cat'].fillna('Unknown')
                else:
                    # Normal fillna if not categorical
                    df['main_cat'] = df['main_cat'].fillna('Unknown')
            else:
                # Create main_cat column if it doesn't exist
                df['main_cat'] = 'Unknown'
            
            # Convert to title case if possible
            if not pd.api.types.is_categorical_dtype(df['main_cat']):
                df['main_cat'] = df['main_cat'].str.title()
            
            conds = [
                df['fine_category'].str.contains('Refrigerator|Freezer|Ice Maker', case=False, na=False),
                df['fine_category'].str.contains('Washer|Dryer', case=False, na=False),
                df['fine_category'].str.contains('Dishwasher', case=False, na=False),
                df['fine_category'].str.contains('Range|Cooktop', case=False, na=False),
            ]
            choices = ['Refrigeration', 'Laundry', 'Dishwashing', 'Cooking']
            df['primary_category'] = np.select(conds, choices, default=df['main_cat'])
            
            # Handle categorical conversion for primary_category
            if pd.api.types.is_categorical_dtype(df['primary_category']):
                # Add new categories if they don't exist
                new_categories = set(['Refrigeration', 'Laundry', 'Dishwashing', 'Cooking', 'Home Appliances']) - set(df['primary_category'].cat.categories)
                if new_categories:
                    df['primary_category'] = df['primary_category'].cat.add_categories(list(new_categories))
            
            # Now do the replacement
            df['primary_category'] = df['primary_category'].replace({
                'Amazon Home': 'Home Appliances',
                'Tools & Home Improvement': 'Home Appliances'
            })
        else:
            # Fallback if no category information
            df['primary_category'] = 'Unknown'
            df['fine_category'] = 'Unknown'
        
        self.meta_df = df
        return df
    
    def preprocess_text(self, text):
        """Clean and normalize text data."""
        try:
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
            
            text = unescape(str(text)).lower()
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text)
            tokens = text.split()
            tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
            return ' '.join(tokens)
        except:
            # Fallback if NLTK resources aren't available
            text = unescape(str(text)).lower()
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text)
            return text
    
    def process_text_fields(self):
        """Process text fields in metadata."""
        fields = ['title', 'description', 'feature']
        
        # Brand might be in a different field
        brand_field = 'brand' if 'brand' in self.meta_df.columns else None
        if brand_field:
            fields.append(brand_field)
        
        for f in fields:
            if f in self.meta_df.columns:
                self.meta_df[f'clean_{f}'] = self.meta_df[f].fillna('').apply(self.preprocess_text)
            else:
                self.meta_df[f'clean_{f}'] = ''
        
        # Combine text fields
        text_fields = []
        if 'clean_title' in self.meta_df.columns:
            text_fields.append(self.meta_df['clean_title'])
        if 'clean_description' in self.meta_df.columns:
            text_fields.append(self.meta_df['clean_description'])
        if 'clean_feature' in self.meta_df.columns:
            text_fields.append(self.meta_df['clean_feature'])
        if 'primary_category' in self.meta_df.columns:
            text_fields.append(self.meta_df['primary_category'])
            
        # Combine available fields
        if text_fields:
            self.meta_df['combined_text'] = pd.concat(text_fields, axis=1).apply(
                lambda x: ' '.join(x.dropna().astype(str)), axis=1
            )
        else:
            # Fallback: use title or create empty string
            self.meta_df['combined_text'] = self.meta_df.get('title', '').fillna('')
        
        # Encode categorical features
        if 'primary_category' in self.meta_df.columns:
            self.meta_df['category_encoded'] = LabelEncoder().fit_transform(self.meta_df['primary_category'])
        
        if brand_field:
            self.meta_df['brand_encoded'] = LabelEncoder().fit_transform(self.meta_df[brand_field].fillna('Unknown'))
    
    def process_reviews(self):
        """Process review data for sentiment analysis."""
        if self.reviews_df is not None:
            # Convert asin to string for consistent joining
            self.reviews_df['asin'] = self.reviews_df['asin'].astype(str)
            
            # Add sentiment analysis
            try:
                self.reviews_df['sentiment'] = self.reviews_df['reviewText'].fillna('').apply(
                    lambda t: TextBlob(t).sentiment.polarity
                )
            except:
                # Fallback if TextBlob fails
                self.reviews_df['sentiment'] = 0
            
            # Calculate average sentiment per product
            avg_sent = self.reviews_df.groupby('asin')['sentiment'] \
                          .mean().reset_index(name='avg_sentiment')
            avg_sent['asin'] = avg_sent['asin'].astype(str)
            
            # Merge sentiment data with metadata
            self.meta_df['asin'] = self.meta_df['asin'].astype(str)
            self.meta_df = self.meta_df.merge(avg_sent, on='asin', how='left')
            self.meta_df['avg_sentiment'] = self.meta_df['avg_sentiment'].fillna(0)
    
    def clean_reviews_for_cf(self):
        """Clean review data for collaborative filtering."""
        if self.reviews_df is None:
            return None
            
        # Clean essential columns
        clean_df = self.reviews_df.dropna(subset=['reviewerID', 'asin', 'overall'])
        
        # Handle vote column if it exists
        if 'vote' in clean_df.columns:
            try:
                if clean_df['vote'].dtype == 'object':
                    clean_df['vote'] = pd.to_numeric(clean_df['vote'].str.replace(',', ''), errors='coerce')
                else:
                    clean_df['vote'] = pd.to_numeric(clean_df['vote'], errors='coerce')
                clean_df['vote'] = clean_df['vote'].fillna(0)
            except:
                clean_df['vote'] = 1
        else:
            clean_df['vote'] = 1
        
        # Filter users with too few ratings
        user_counts = clean_df['reviewerID'].value_counts()
        active_users = user_counts[user_counts >= 3].index
        filtered_df = clean_df[clean_df['reviewerID'].isin(active_users)]
        
        return filtered_df
    
    def analyze_related_products(self):
        """Process related products data for association rules."""
        if 'related' in self.meta_df.columns:
            # Extract 'also_buy' from related products field
            def extract_also_buy(related):
                if isinstance(related, dict) and 'also_buy' in related:
                    return related['also_buy']
                elif isinstance(related, str):
                    try:
                        rel_dict = eval(related.replace('true', 'True').replace('false', 'False'))
                        if isinstance(rel_dict, dict) and 'also_buy' in rel_dict:
                            return rel_dict['also_buy']
                    except:
                        pass
                return []
                
            self.meta_df['also_buy'] = self.meta_df['related'].apply(extract_also_buy)
            
    def preprocess_all(self):
        """Run all preprocessing steps."""
        with st.spinner("Preprocessing data..."):
            self.clean_category()
            self.process_text_fields()
            self.process_reviews()
            self.analyze_related_products()
            clean_reviews = self.clean_reviews_for_cf()
            return self.meta_df, clean_reviews

class TextSimilarityRecommender:
    """Class for text-based similarity recommendations."""
    
    def __init__(self, meta_df):
        """Initialize with processed metadata."""
        self.meta_df = meta_df
        self.tfidf_matrix = None
        self.tfidf_array = None
        self.is_initialized = False
        
    def initialize(self):
        """Generate TF-IDF matrix for text similarity."""
        if 'combined_text' not in self.meta_df.columns:
            st.warning("Missing 'combined_text' column. Run preprocessing first.")
            return
            
        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.meta_df['combined_text'])
        
        if not isinstance(self.tfidf_matrix, csr_matrix):
            self.tfidf_matrix = csr_matrix(self.tfidf_matrix)
            
        self.tfidf_array = self.tfidf_matrix.toarray().astype(np.float32)
        self.is_initialized = True
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _cosine_sim_numba(qv, mat):
        """Fast cosine similarity calculation using Numba."""
        qn = np.sqrt(np.sum(qv**2))
        res = np.zeros(mat.shape[0], dtype=np.float32)
        for i in range(mat.shape[0]):
            dot = np.dot(mat[i], qv)
            denom = max(np.sqrt(np.sum(mat[i]**2))*qn, 1e-8)
            res[i] = dot/denom
        return res
    
    def get_recommendations(self, asin, top_n=5, popular_fallback=None):
        """Get recommendations based on text similarity."""
        if not self.is_initialized:
            self.initialize()
            
        # Find index of the product
        indices = self.meta_df.index[self.meta_df['asin'] == asin].tolist()
        if not indices:
            return popular_fallback() if popular_fallback else pd.DataFrame()
            
        idx = indices[0]
        
        # Calculate text similarity
        text_sim = self._cosine_sim_numba(self.tfidf_array[idx], self.tfidf_array)
        
        # Get top similar items (excluding the input item)
        similar_indices = np.argsort(-text_sim)
        similar_indices = [i for i in similar_indices if i != idx][:top_n]
        
        # Return required columns with scores
        item_info = ['asin', 'title', 'description', 'price', 'brand', 'primary_category']
        available_columns = [col for col in item_info if col in self.meta_df.columns]
        result = self.meta_df.iloc[similar_indices][available_columns].copy()
        result['similarity_score'] = text_sim[similar_indices]
        return result

class FrequentItemsetRecommender:
    """Class for association rule mining and frequent itemset recommendations."""
    
    def __init__(self, meta_df):
        """Initialize with processed metadata."""
        self.meta_df = meta_df
        
    def get_recommendations(self, asin, top_n=5, popular_fallback=None):
        """Get recommendations based on association rules and frequent itemsets."""
        # Find index of the product
        indices = self.meta_df.index[self.meta_df['asin'] == asin].tolist()
        
        # Check for related products data
        has_also_buy = 'also_buy' in self.meta_df.columns
        has_related = 'related' in self.meta_df.columns
        
        if not indices or (not has_also_buy and not has_related):
            return popular_fallback() if popular_fallback else pd.DataFrame()
            
        idx = indices[0]
        
        # Get co-purchased items - try 'also_buy' first
        co_purchased = []
        
        if has_also_buy:
            also_buy = self.meta_df.at[idx, 'also_buy']
            
            if isinstance(also_buy, list) or (isinstance(also_buy, str) and also_buy):
                if isinstance(also_buy, str):
                    also_buy = re.split(r'[,\s]+', also_buy.strip())
                    
                co_purchased = self.meta_df[self.meta_df['asin'].isin(also_buy)]
        
        # If not enough co-purchased items and 'related' exists, try using it
        if len(co_purchased) < top_n and has_related:
            related = self.meta_df.at[idx, 'related']
            
            if isinstance(related, dict) and 'also_bought' in related:
                also_bought = related['also_bought']
                if also_bought and isinstance(also_bought, list):
                    additional = self.meta_df[self.meta_df['asin'].isin(also_bought)]
                    if not co_purchased.empty:
                        co_purchased = pd.concat([co_purchased, additional])
                    else:
                        co_purchased = additional
            
            # Try 'bought_together' as well
            if isinstance(related, dict) and 'bought_together' in related:
                bought_together = related['bought_together']
                if bought_together and isinstance(bought_together, list):
                    additional = self.meta_df[self.meta_df['asin'].isin(bought_together)]
                    if not co_purchased.empty:
                        co_purchased = pd.concat([co_purchased, additional])
                    else:
                        co_purchased = additional
        
        if not isinstance(co_purchased, pd.DataFrame):
            co_purchased = pd.DataFrame()
            
        if not co_purchased.empty and len(co_purchased) >= top_n:
            # Get all required columns with scores
            item_info = ['asin', 'title', 'description', 'price', 'brand', 'primary_category']
            available_columns = [col for col in item_info if col in self.meta_df.columns]
            result = co_purchased.head(top_n)[available_columns].copy()
            result['co_purchase_score'] = 1.0  # All co-purchased items get a score of 1.0
            return result
        else:
            # Fill with similar category items if not enough co-purchased items
            if 'primary_category' in self.meta_df.columns:
                category = self.meta_df.at[idx, 'primary_category']
                similar_cat = self.meta_df[self.meta_df['primary_category'] == category]
                similar_cat = similar_cat[~similar_cat['asin'].isin([asin] + list(co_purchased['asin'] if not co_purchased.empty else []))]
                
                # Combine co-purchased and category items
                item_info = ['asin', 'title', 'description', 'price', 'brand', 'primary_category']
                available_columns = [col for col in item_info if col in self.meta_df.columns]
                
                if not co_purchased.empty:
                    recommendations = pd.concat([co_purchased, similar_cat]).head(top_n)
                    recommendations = recommendations[available_columns].copy()
                    # Add scores - co-purchased get 1.0, category similarity gets 0.7
                    recommendations['co_purchase_score'] = 0.7
                    recommendations.loc[recommendations['asin'].isin(co_purchased['asin']), 'co_purchase_score'] = 1.0
                    return recommendations
                else:
                    result = similar_cat.head(top_n)[available_columns].copy()
                    result['co_purchase_score'] = 0.7
                    return result
            else:
                # Fallback to popular items
                return popular_fallback() if popular_fallback else pd.DataFrame()

class ClusteringRecommender:
    """Class for clustering-based recommendations."""
    
    def __init__(self, meta_df):
        """Initialize with processed metadata."""
        self.meta_df = meta_df
        
    def get_recommendations(self, asin, top_n=5, popular_fallback=None):
        """Get recommendations based on product clusters."""
        # Find index of the product
        indices = self.meta_df.index[self.meta_df['asin'] == asin].tolist()
        if not indices:
            return popular_fallback() if popular_fallback else pd.DataFrame()
            
        idx = indices[0]
        
        # Find similar items by category and brand
        has_category = 'primary_category' in self.meta_df.columns
        has_brand = 'brand' in self.meta_df.columns
        
        if not has_category and not has_brand:
            return popular_fallback() if popular_fallback else pd.DataFrame()
        
        # Get item info
        category = self.meta_df.at[idx, 'primary_category'] if has_category else None
        brand = self.meta_df.at[idx, 'brand'] if has_brand else None
        
        # Calculate clustering scores
        scores = []
        
        # First try to find items with same brand and category
        if has_category and has_brand and category and brand:
            same_brand_cat = self.meta_df[
                (self.meta_df['primary_category'] == category) & 
                (self.meta_df['brand'] == brand) &
                (self.meta_df['asin'] != asin)
            ]
            
            if len(same_brand_cat) >= top_n:
                # Return required columns with scores
                item_info = ['asin', 'title', 'description', 'price', 'brand', 'primary_category']
                available_columns = [col for col in item_info if col in self.meta_df.columns]
                result = same_brand_cat.head(top_n)[available_columns].copy()
                result['clustering_score'] = 1.0  # Same brand + category gets highest score
                return result
        else:
            same_brand_cat = pd.DataFrame()
        
        # If not enough with brand+category, try just category
        if has_category and category:
            same_cat = self.meta_df[
                (self.meta_df['primary_category'] == category) & 
                (self.meta_df['asin'] != asin)
            ]
            
            # Remove items we already have from brand+category
            if not same_brand_cat.empty:
                same_cat = same_cat[~same_cat['asin'].isin(same_brand_cat['asin'])]
                
            if len(same_brand_cat) + len(same_cat) >= top_n:
                # Return required columns with scores
                item_info = ['asin', 'title', 'description', 'price', 'brand', 'primary_category']
                available_columns = [col for col in item_info if col in self.meta_df.columns]
                recommendations = pd.concat([same_brand_cat, same_cat]).head(top_n)[available_columns].copy()
                # Add scores - brand+category gets 1.0, just category gets 0.8
                recommendations['clustering_score'] = 0.8
                if not same_brand_cat.empty:
                    recommendations.loc[recommendations['asin'].isin(same_brand_cat['asin']), 'clustering_score'] = 1.0
                return recommendations
        else:
            same_cat = pd.DataFrame()
        
        # If not enough with category, try just brand
        if has_brand and brand and (len(same_brand_cat) + len(same_cat) < top_n):
            same_brand = self.meta_df[
                (self.meta_df['brand'] == brand) &
                (self.meta_df['asin'] != asin)
            ]
            
            # Remove items we already have
            same_brand = same_brand[
                (~same_brand['asin'].isin(same_brand_cat['asin'] if not same_brand_cat.empty else [])) &
                (~same_brand['asin'].isin(same_cat['asin'] if not same_cat.empty else []))
            ]
            
            # Combine all recommendations
            recommendations = pd.concat([
                same_brand_cat if not same_brand_cat.empty else pd.DataFrame(),
                same_cat if not same_cat.empty else pd.DataFrame(),
                same_brand
            ]).head(top_n)
            
            # If we have enough recommendations, return them
            if len(recommendations) >= top_n:
                # Return required columns with scores
                item_info = ['asin', 'title', 'description', 'price', 'brand', 'primary_category']
                available_columns = [col for col in item_info if col in self.meta_df.columns]
                recommendations = recommendations[available_columns].copy()
                # Add scores - brand+category gets 1.0, category gets 0.8, brand gets 0.6
                recommendations['clustering_score'] = 0.6
                if not same_cat.empty:
                    recommendations.loc[recommendations['asin'].isin(same_cat['asin']), 'clustering_score'] = 0.8
                if not same_brand_cat.empty:
                    recommendations.loc[recommendations['asin'].isin(same_brand_cat['asin']), 'clustering_score'] = 1.0
                return recommendations
        
        # If still not enough, add popular items
        if popular_fallback:
            # Calculate how many more items we need
            if not same_brand_cat.empty and not same_cat.empty:
                needed = top_n - len(same_brand_cat) - len(same_cat)
            elif not same_brand_cat.empty:
                needed = top_n - len(same_brand_cat)
            elif not same_cat.empty:
                needed = top_n - len(same_cat)
            else:
                needed = top_n
                
            popular = popular_fallback(needed)
            
            # Remove duplicates
            if not same_brand_cat.empty or not same_cat.empty:
                existing_asins = []
                if not same_brand_cat.empty:
                    existing_asins.extend(same_brand_cat['asin'].tolist())
                if not same_cat.empty:
                    existing_asins.extend(same_cat['asin'].tolist())
                    
                popular = popular[~popular['asin'].isin(existing_asins)]
                popular = popular[popular['asin'] != asin]
                
                # Combine with existing recommendations
                dfs_to_concat = []
                if not same_brand_cat.empty:
                    dfs_to_concat.append(same_brand_cat)
                if not same_cat.empty:
                    dfs_to_concat.append(same_cat)
                dfs_to_concat.append(popular)
                
                recommendations = pd.concat(dfs_to_concat).head(top_n)
            else:
                recommendations = popular
                
            # Return required columns with scores
            item_info = ['asin', 'title', 'description', 'price', 'brand', 'primary_category']
            available_columns = [col for col in item_info if col in recommendations.columns]
            result = recommendations[available_columns].copy()
            # Add scores
            result['clustering_score'] = 0.4  # Popular items get lowest score
            if 'asin' in result.columns:
                if not same_cat.empty:
                    result.loc[result['asin'].isin(same_cat['asin']), 'clustering_score'] = 0.8
                if not same_brand_cat.empty:
                    result.loc[result['asin'].isin(same_brand_cat['asin']), 'clustering_score'] = 1.0
            return result
        
        # If we get here, we don't have enough recommendations and no fallback
        if not same_brand_cat.empty or not same_cat.empty:
            dfs_to_concat = []
            if not same_brand_cat.empty:
                dfs_to_concat.append(same_brand_cat)
            if not same_cat.empty:
                dfs_to_concat.append(same_cat)
                
            recommendations = pd.concat(dfs_to_concat)
            item_info = ['asin', 'title', 'description', 'price', 'brand', 'primary_category']
            available_columns = [col for col in item_info if col in self.meta_df.columns]
            result = recommendations[available_columns].copy()
            # Add scores
            result['clustering_score'] = 0.8
            if not same_brand_cat.empty:
                result.loc[result['asin'].isin(same_brand_cat['asin']), 'clustering_score'] = 1.0
            return result
        else:
            # Return empty DataFrame with correct columns
            result = pd.DataFrame(columns=['asin', 'title', 'description', 'price', 'brand', 'primary_category'])
            result['clustering_score'] = []
            return result

class InfluentialProductsRecommender:
    """Class for recommendations based on product influence."""
    
    def __init__(self, meta_df):
        """Initialize with processed metadata."""
        self.meta_df = meta_df
        self.is_initialized = False
        
    def initialize(self):
        """Prepare influence data."""
        if 'also_buy' not in self.meta_df.columns:
            st.warning("No co-purchase data available for influence analysis")
            return
            
        # Count how many times each product appears in others' also_buy lists
        influence_count = defaultdict(int)
        
        for _, row in self.meta_df.iterrows():
            also_buy = row['also_buy']
            if isinstance(also_buy, list) or (isinstance(also_buy, str) and also_buy):
                if isinstance(also_buy, str):
                    also_buy = re.split(r'[,\s]+', also_buy.strip())
                for item in also_buy:
                    influence_count[item] += 1
        
        # Add influence score to metadata
        self.meta_df['influence_score'] = self.meta_df['asin'].map(
            lambda x: influence_count.get(x, 0)
        )
        
        self.is_initialized = True
    
    def get_recommendations(self, asin, top_n=5, popular_fallback=None):
        """Get recommendations based on product influence."""
        if not self.is_initialized:
            self.initialize()
            
        if 'influence_score' not in self.meta_df.columns:
            return popular_fallback() if popular_fallback else pd.DataFrame()
        
        # Find index of the product
        indices = self.meta_df.index[self.meta_df['asin'] == asin].tolist()
        if not indices:
            return popular_fallback() if popular_fallback else pd.DataFrame()
            
        idx = indices[0]
        
        # Get category of the target product
        category = self.meta_df.at[idx, 'primary_category']
        
        # Find influential products in the same category
        influential = self.meta_df[
            (self.meta_df['primary_category'] == category) & 
            (self.meta_df['asin'] != asin)
        ].sort_values('influence_score', ascending=False)
        
        if len(influential) >= top_n:
            item_info = ['asin', 'title', 'description', 'price', 'brand', 'primary_category']
            available_columns = [col for col in item_info if col in self.meta_df.columns]
            result = influential.head(top_n)[available_columns].copy()
            # Normalize influence scores to 0-1 range for consistent scoring
            if 'influence_score' in influential.columns:
                max_score = self.meta_df['influence_score'].max()
                # Get influence scores for selected items
                selected_asins = result['asin'].values
                influence_scores = self.meta_df[self.meta_df['asin'].isin(selected_asins)].set_index('asin')['influence_score']
                result['influence_score_normalized'] = result['asin'].map(influence_scores) / max_score if max_score > 0 else 0
            return result
        
        # If not enough, add generally influential products
        other_influential = self.meta_df[
            (self.meta_df['primary_category'] != category) & 
            (self.meta_df['asin'] != asin)
        ].sort_values('influence_score', ascending=False)
        
        recommendations = pd.concat([influential, other_influential]).head(top_n)
        item_info = ['asin', 'title', 'description', 'price', 'brand', 'primary_category']
        available_columns = [col for col in item_info if col in self.meta_df.columns]
        result = recommendations[available_columns].copy()
        # Normalize influence scores
        if 'influence_score' in recommendations.columns:
            max_score = self.meta_df['influence_score'].max()
            # Get influence scores for selected items
            selected_asins = result['asin'].values
            influence_scores = self.meta_df[self.meta_df['asin'].isin(selected_asins)].set_index('asin')['influence_score']
            result['influence_score_normalized'] = result['asin'].map(influence_scores) / max_score if max_score > 0 else 0
        return result

class CollaborativeFilteringRecommender:
    """Class for collaborative filtering recommendations."""
    
    def __init__(self, meta_df, reviews_df):
        """Initialize with processed metadata and reviews."""
        self.meta_df = meta_df
        self.reviews_df = reviews_df
        self.cf_model = None
        self.cf_matrix = None
        self.cf_mappers = None
        self.is_initialized = False
        
    def initialize(self):
        """Prepare and train the collaborative filtering model."""
        if self.reviews_df is None:
            st.warning("No review data available for collaborative filtering")
            return
            
        try:
            # Verify required columns
            required_cols = ['reviewerID', 'asin', 'overall']
            if not all(col in self.reviews_df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in self.reviews_df.columns]
                st.warning(f"Missing required columns for CF: {missing}")
                return
                
            # Create user-item matrix for CF
            user_ids = self.reviews_df['reviewerID'].unique()
            item_ids = self.reviews_df['asin'].unique()
            
            user_mapper = {user: i for i, user in enumerate(user_ids)}
            item_mapper = {item: i for i, item in enumerate(item_ids)}
            
            user_inv_mapper = {i: user for user, i in user_mapper.items()}
            item_inv_mapper = {i: item for item, i in item_mapper.items()}
            
            user_indices = self.reviews_df['reviewerID'].map(user_mapper).values
            item_indices = self.reviews_df['asin'].map(item_mapper).values
            ratings = self.reviews_df['overall'].values
            
            user_item_matrix = csr_matrix((ratings, (user_indices, item_indices)), 
                                         shape=(len(user_ids), len(item_ids)))
            
            # Train SVD model
            max_components = min(user_item_matrix.shape[0], user_item_matrix.shape[1]) - 1
            n_components = min(50, max_components)
            
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            latent_matrix = svd.fit_transform(user_item_matrix)
            
            self.cf_model = {
                'svd': svd,
                'user_features': latent_matrix,
                'item_features': svd.components_.T
            }
            
            self.cf_matrix = user_item_matrix
            self.cf_mappers = (user_mapper, item_mapper, user_inv_mapper, item_inv_mapper)
            
            self.is_initialized = True
            
        except Exception as e:
            st.error(f"Error in collaborative filtering initialization: {e}")
    
    def get_recommendations(self, user_id, top_n=5, popular_fallback=None):
        """Get collaborative filtering recommendations for a user."""
        if not self.is_initialized:
            self.initialize()
            
        if self.cf_model is None or self.cf_matrix is None:
            return popular_fallback() if popular_fallback else pd.DataFrame()
            
        user_mapper, item_mapper, user_inv_mapper, item_inv_mapper = self.cf_mappers
        
        # If user not in training data, return popular items
        if user_id not in user_mapper:
            return popular_fallback() if popular_fallback else pd.DataFrame()
        
        user_idx = user_mapper[user_id]
        
        # Get items the user has already rated
        user_row = self.cf_matrix[user_idx].toarray().flatten()
        rated_indices = np.where(user_row > 0)[0]
        rated_items = {item_inv_mapper[idx] for idx in rated_indices}
        
        # Get all possible items
        all_items = set(item_inv_mapper.values())
        unrated_items = list(all_items - rated_items)
        
        # Get user features
        svd = self.cf_model['svd']
        user_features = self.cf_model['user_features'][user_idx]
        
        # Predict ratings for unrated items
        predictions = []
        for item_id in unrated_items[:min(500, len(unrated_items))]:
            try:
                item_idx = item_mapper[item_id]
                item_vec = svd.components_[:, item_idx]
                pred_rating = np.dot(user_features, item_vec)
                pred_rating = max(1, min(5, pred_rating))
                
                if pred_rating >= 4.0:  # Only consider well-predicted items
                    predictions.append((item_id, pred_rating))
            except:
                continue
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommended_items = [item_id for item_id, _ in predictions[:top_n]]
        
        # If no predictions meet threshold, use top predictions regardless of score
        if not recommended_items and predictions:
            recommended_items = [item_id for item_id, _ in predictions[:top_n]]
        
        # If still no recommendations, use popular items
        if not recommended_items:
            return popular_fallback() if popular_fallback else pd.DataFrame()
        
        # Get metadata for recommended items
        item_info = ['asin', 'title', 'description', 'price', 'brand', 'primary_category']
        available_columns = [col for col in item_info if col in self.meta_df.columns]
        
        # Filter for recommended items and select required columns
        recommendations = self.meta_df[self.meta_df['asin'].isin(recommended_items)]
        if not recommendations.empty:
            result = recommendations[available_columns].copy()
            # Add CF scores
            cf_scores = {}
            for item_id, rating in predictions:
                if item_id in recommended_items:
                    cf_scores[item_id] = (rating - 1) / 4  # Normalize to 0-1 scale
            result['cf_score'] = result['asin'].map(cf_scores)
            return result
        else:
            return popular_fallback() if popular_fallback else pd.DataFrame()

class HybridRecommender:
    """Class for hybrid recommendations combining multiple algorithms."""
    
    def __init__(self, meta_df, reviews_df=None):
        """Initialize with processed metadata and reviews."""
        self.meta_df = meta_df
        self.reviews_df = reviews_df
        self.text_recommender = None
        self.cf_recommender = None
        self.itemset_recommender = None
        self.cluster_recommender = None
        self.influence_recommender = None
        
    def initialize(self):
        """Initialize all component recommenders."""
        self.text_recommender = TextSimilarityRecommender(self.meta_df)
        self.text_recommender.initialize()
        
        self.itemset_recommender = FrequentItemsetRecommender(self.meta_df)
        
        self.cluster_recommender = ClusteringRecommender(self.meta_df)
        
        self.influence_recommender = InfluentialProductsRecommender(self.meta_df)
        self.influence_recommender.initialize()
        
        if self.reviews_df is not None:
            self.cf_recommender = CollaborativeFilteringRecommender(self.meta_df, self.reviews_df)
            self.cf_recommender.initialize()
    
    def weighted_cf_prediction(self, user_id, asin):
        """Get collaborative filtering prediction for a specific user and item."""
        if not self.cf_recommender or not self.cf_recommender.is_initialized:
            return 0.0
            
        user_mapper, item_mapper, _, _ = self.cf_recommender.cf_mappers
        
        # Check if user and item are in training data
        if user_id not in user_mapper or asin not in item_mapper:
            return 0.0
            
        user_idx = user_mapper[user_id]
        item_idx = item_mapper[asin]
        
        # Get prediction using SVD
        svd = self.cf_recommender.cf_model['svd']
        user_features = self.cf_recommender.cf_model['user_features'][user_idx]
        item_vec = svd.components_[:, item_idx]
        
        # Calculate predicted rating
        pred_rating = np.dot(user_features, item_vec)
        
        # Normalize to [0,1] scale (1-5 rating scale becomes 0-1)
        normalized_rating = (max(1, min(5, pred_rating)) - 1) / 4.0
        
        return normalized_rating
    
    def get_enhanced_recommendations(self, asin, top_n=5, user_id=None,
                                    w_text=0.4, w_cat=0.15, w_co=0.15, w_sent=0.1, w_cf=0.2,
                                    popular_fallback=None):
        """Get hybrid recommendations with weighted components."""
        # Find index of the product
        indices = self.meta_df.index[self.meta_df['asin'] == asin].tolist()
        if not indices:
            return popular_fallback() if popular_fallback else pd.DataFrame()
            
        idx = indices[0]
        
        # Adjust weights if CF is not available
        cf_available = user_id and self.cf_recommender and self.cf_recommender.is_initialized
        if not cf_available:
            # Redistribute CF weight to other components
            w_text += w_cf * 0.5
            w_cat += w_cf * 0.2
            w_co += w_cf * 0.2
            w_sent += w_cf * 0.1
            w_cf = 0.0
        
        # Calculate text similarity if available
        if self.text_recommender and self.text_recommender.is_initialized:
            text_sim = self.text_recommender._cosine_sim_numba(
                self.text_recommender.tfidf_array[idx], 
                self.text_recommender.tfidf_array
            )
        else:
            text_sim = np.zeros(len(self.meta_df))
        
        # Calculate category similarity
        if 'category_encoded' in self.meta_df.columns:
            cat_sim = (self.meta_df['category_encoded'].values == 
                      self.meta_df.at[idx, 'category_encoded']).astype(np.float32)
        else:
            cat_sim = np.zeros(len(self.meta_df))
        
        # Calculate co-purchase similarity
        co_sim = np.zeros(len(self.meta_df), dtype=np.float32)
        
        # Try also_buy if available
        if 'also_buy' in self.meta_df.columns:
            rel = self.meta_df.at[idx, 'also_buy']
            if isinstance(rel, list) or (isinstance(rel, str) and rel):
                if isinstance(rel, str):
                    rel = re.split(r'[,\s]+', rel.strip())
                inds = self.meta_df[self.meta_df['asin'].isin(rel)].index
                co_sim[inds] = 1.0
        
        # Try related if available
        if 'related' in self.meta_df.columns:
            rel = self.meta_df.at[idx, 'related']
            related_asins = []
            
            # Extract various related product types
            if isinstance(rel, dict):
                for key in ['also_bought', 'bought_together', 'also_viewed']:
                    if key in rel and isinstance(rel[key], list):
                        related_asins.extend(rel[key])
            
            if related_asins:
                inds = self.meta_df[self.meta_df['asin'].isin(related_asins)].index
                co_sim[inds] = 1.0
        
        # Calculate sentiment similarity
        if 'avg_sentiment' in self.meta_df.columns:
            sent = (self.meta_df['avg_sentiment'].values + 1) / 2  # Normalize to [0,1]
        else:
            sent = np.zeros(len(self.meta_df))
        
        # Calculate CF scores if applicable
        cf_sim = np.zeros(len(self.meta_df), dtype=np.float32)
        if cf_available and user_id:
            for i, product_asin in enumerate(self.meta_df['asin'].values):
                cf_sim[i] = self.weighted_cf_prediction(user_id, product_asin)
        
        # Combine scores
        score = w_text*text_sim + w_cat*cat_sim + w_co*co_sim + w_sent*sent + w_cf*cf_sim
        
        # Get top recommendations
        top = np.argpartition(-score, top_n+1)[:top_n+1]
        top = [i for i in top if i != idx][:top_n]
        
        # Get all required columns
        item_info = ['asin', 'title', 'description', 'price', 'brand', 'primary_category']
        available_columns = [col for col in item_info if col in self.meta_df.columns]
        
        result = self.meta_df.iloc[top][available_columns].copy()
        # Add hybrid score
        result['hybrid_score'] = score[top]
        
        return result
    
    def get_full_hybrid_recommendations(self, user_id, asin, top_n=10, popular_fallback=None):
        """Get comprehensive hybrid recommendations using all available algorithms."""
        # Recommendations from each algorithm (2 each)
        n_each = 2
        all_recs = []
        
        # Collect recommendations from each algorithm
        if self.cf_recommender and self.cf_recommender.is_initialized:
            cf_recs = self.cf_recommender.get_recommendations(user_id, top_n=n_each)
            if not cf_recs.empty:
                all_recs.append(cf_recs)
        
        if self.text_recommender and self.text_recommender.is_initialized:
            text_recs = self.text_recommender.get_recommendations(
                asin, top_n=n_each, popular_fallback=popular_fallback
            )
            if not text_recs.empty:
                all_recs.append(text_recs)
        
        if self.itemset_recommender:
            freq_recs = self.itemset_recommender.get_recommendations(
                asin, top_n=n_each, popular_fallback=popular_fallback
            )
            if not freq_recs.empty:
                all_recs.append(freq_recs)
        
        if self.cluster_recommender:
            cluster_recs = self.cluster_recommender.get_recommendations(
                asin, top_n=n_each, popular_fallback=popular_fallback
            )
            if not cluster_recs.empty:
                all_recs.append(cluster_recs)
        
        if self.influence_recommender and self.influence_recommender.is_initialized:
            infl_recs = self.influence_recommender.get_recommendations(
                asin, top_n=n_each, popular_fallback=popular_fallback
            )
            if not infl_recs.empty:
                all_recs.append(infl_recs)
        
        # Combine and deduplicate
        if all_recs:
            combined_recs = pd.concat(all_recs)
            combined_recs = combined_recs.drop_duplicates(subset=['asin'])
        else:
            combined_recs = pd.DataFrame()
        
        # If not enough, supplement with enhanced hybrid recommendations
        if len(combined_recs) < top_n:
            hybrid_recs = self.get_enhanced_recommendations(
                asin, top_n=top_n-len(combined_recs), user_id=user_id, popular_fallback=popular_fallback
            )
            # Deduplicate
            if not hybrid_recs.empty:
                hybrid_recs = hybrid_recs[~hybrid_recs['asin'].isin(combined_recs['asin'])]
                combined_recs = pd.concat([combined_recs, hybrid_recs])
        
        # Return top recommendations
        return combined_recs.head(top_n)

class PopularityRecommender:
    """Class for popularity-based recommendations."""
    
    def __init__(self, meta_df, reviews_df=None):
        """Initialize with metadata and reviews."""
        self.meta_df = meta_df
        self.reviews_df = reviews_df
        self.popular_cache = None
        
    def get_recommendations(self, top_n=10):
        """Get popular items based on review count and rating."""
        if self.popular_cache is not None and len(self.popular_cache) >= top_n:
            return self.popular_cache.head(top_n)
            
        if self.reviews_df is not None:
            # Group by asin and calculate metrics
            popular = self.reviews_df.groupby('asin').agg({
                'overall': ['count', 'mean']
            })
            popular.columns = ['count', 'mean']
            
            # Filter items with enough reviews and sort by popularity score
            popular = popular[popular['count'] > 5]
            popular['score'] = popular['mean'] * np.log1p(popular['count'])
            popular = popular.sort_values('score', ascending=False)
            
            # Get top N popular items
            top_asins = list(popular.index[:min(len(popular), 100)])  # Cache up to 100
            
            # Return metadata for these items with required columns
            item_info = ['asin', 'title', 'description', 'price', 'brand', 'primary_category']
            available_columns = [col for col in item_info if col in self.meta_df.columns]
            self.popular_cache = self.meta_df[self.meta_df['asin'].isin(top_asins)][available_columns]
            # Sort by score to maintain order
            popular_dict = popular.sort_values('score', ascending=False)['score'].to_dict()
            self.popular_cache['popularity_score'] = self.popular_cache['asin'].map(popular_dict)
            self.popular_cache = self.popular_cache.sort_values('popularity_score', ascending=False)
            return self.popular_cache.head(top_n)
        else:
            # If no review data, return random items
            item_info = ['asin', 'title', 'description', 'price', 'brand', 'primary_category']
            available_columns = [col for col in item_info if col in self.meta_df.columns]
            self.popular_cache = self.meta_df.sample(min(100, len(self.meta_df)))[available_columns]
            return self.popular_cache.head(top_n)

class UserBehaviorHandler:
    """Class to handle user click behavior and personalized recommendations."""
    
    def __init__(self):
        """Initialize with empty clicks dictionary."""
        self.user_clicks = defaultdict(list)
    
    def record_click(self, user_id, asin):
        """Record user click for future recommendations."""
        self.user_clicks[user_id].append(asin)
        # Keep only the last 10 clicks per user
        self.user_clicks[user_id] = self.user_clicks[user_id][-10:]
        return self.user_clicks[user_id]
    
    def get_click_history(self, user_id):
        """Get user's click history."""
        return self.user_clicks.get(user_id, [])
    
    def clear_history(self, user_id):
        """Clear user's click history."""
        if user_id in self.user_clicks:
            del self.user_clicks[user_id]

class RecommendationSystem:
    """Main recommendation system class that integrates all recommenders."""
    
    def __init__(self, meta_df=None, reviews_df=None):
        """Initialize the recommendation system."""
        self.meta_df = meta_df
        self.reviews_df = reviews_df
        self.preprocessor = None
        self.text_recommender = None
        self.itemset_recommender = None
        self.cluster_recommender = None
        self.influence_recommender = None
        self.cf_recommender = None
        self.hybrid_recommender = None
        self.popularity_recommender = None
        self.user_behavior = UserBehaviorHandler()
        self.is_initialized = False
    
    def load_data(self, meta_path, reviews_path):
        """Load data from files."""
        print(f"Loading data from {meta_path} and {reviews_path}...")
        try:
            self.meta_df = pd.read_json(meta_path, lines=True)
            self.reviews_df = pd.read_json(reviews_path, lines=True)
            print(f"Loaded {len(self.meta_df)} products and {len(self.reviews_df)} reviews")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def initialize(self):
        """Initialize all components of the recommendation system."""
        if self.meta_df is None:
            st.error("Error: No metadata available. Load data first.")
            return False
        
        with st.spinner("Initializing recommendation system..."):
            # Preprocess data
            self.preprocessor = DataPreprocessor(self.meta_df, self.reviews_df)
            processed_meta, processed_reviews = self.preprocessor.preprocess_all()
            
            # Initialize recommenders
            self.text_recommender = TextSimilarityRecommender(processed_meta)
            self.text_recommender.initialize()
            
            self.itemset_recommender = FrequentItemsetRecommender(processed_meta)
            
            self.cluster_recommender = ClusteringRecommender(processed_meta)
            
            self.influence_recommender = InfluentialProductsRecommender(processed_meta)
            self.influence_recommender.initialize()
            
            if processed_reviews is not None:
                self.cf_recommender = CollaborativeFilteringRecommender(processed_meta, processed_reviews)
                self.cf_recommender.initialize()
            
            self.hybrid_recommender = HybridRecommender(processed_meta, processed_reviews)
            self.hybrid_recommender.initialize()
            
            self.popularity_recommender = PopularityRecommender(processed_meta, self.reviews_df)
            
            self.is_initialized = True
            return True
    
    def get_popular_items(self, top_n=10):
        """Get popular items based on review count and rating."""
        if not self.is_initialized:
            if not self.initialize():
                return pd.DataFrame()
        
        recommendations = self.popularity_recommender.get_recommendations(top_n)
        return self.format_recommendations(recommendations)
    
    def record_user_click(self, user_id, asin):
        """Record user click for future recommendations."""
        return self.user_behavior.record_click(user_id, asin)
    
    def get_recommendations_for_user(self, user_id, top_n=10):
        """Get personalized recommendations based on user history."""
        if not self.is_initialized:
            if not self.initialize():
                return self.get_popular_items(top_n)
        
        # Get user's click history
        clicked_items = self.user_behavior.get_click_history(user_id)
        
        # If no click history, return popular items
        if not clicked_items:
            return self.get_popular_items(top_n)
        
        # Use the last clicked item for recommendations
        last_item = clicked_items[-1]
        
        # Use full hybrid recommendations
        recommendations = self.hybrid_recommender.get_full_hybrid_recommendations(
            user_id, last_item, top_n=top_n+len(clicked_items),
            popular_fallback=lambda n: self.popularity_recommender.get_recommendations(n)
        )
        
        # Exclude items the user has already clicked
        final_recs = recommendations[~recommendations['asin'].isin(clicked_items)]
        
        # If we don't have enough recommendations, add popular items
        if len(final_recs) < top_n:
            popular_recs = self.get_popular_items(top_n - len(final_recs))
            # Remove duplicates
            popular_recs = popular_recs[~popular_recs['asin'].isin(final_recs['asin'])]
            popular_recs = popular_recs[~popular_recs['asin'].isin(clicked_items)]
            final_recs = pd.concat([final_recs, popular_recs])
        
        return self.format_recommendations(final_recs.head(top_n))
    
    def get_recommendations_by_algorithm(self, asin, user_id=None, top_n=5, algorithm="hybrid"):
        """Get recommendations using a specific algorithm with scores."""
        if not self.is_initialized:
            if not self.initialize():
                return self.get_popular_items(top_n)
        
        # Popular items fallback function
        popular_fallback = lambda n: self.popularity_recommender.get_recommendations(n)
        
        recommendations = None
        
        if algorithm == "text_similarity":
            recommendations = self.text_recommender.get_recommendations(asin, top_n, popular_fallback)
        
        elif algorithm == "frequent_itemset":
            recommendations = self.itemset_recommender.get_recommendations(asin, top_n, popular_fallback)
        
        elif algorithm == "clustering":
            recommendations = self.cluster_recommender.get_recommendations(asin, top_n, popular_fallback)
        
        elif algorithm == "influential":
            recommendations = self.influence_recommender.get_recommendations(asin, top_n, popular_fallback)
        
        elif algorithm == "collaborative_filtering" and user_id:
            if self.cf_recommender:
                recommendations = self.cf_recommender.get_recommendations(user_id, top_n, popular_fallback)
            else:
                recommendations = popular_fallback(top_n)
        
        elif algorithm == "hybrid":
            recommendations = self.hybrid_recommender.get_enhanced_recommendations(
                asin, top_n, user_id=user_id, popular_fallback=popular_fallback
            )
        
        else:
            recommendations = popular_fallback(top_n)
        
        return self.format_recommendations(recommendations)
    
    def format_recommendations(self, recommendations):
        """Format recommendation results with required information."""
        if recommendations is None or recommendations.empty:
            return pd.DataFrame(columns=['asin', 'title', 'description', 'price'])
        
        # Ensure we have all required columns
        required_columns = ['asin', 'title', 'description', 'price']
        existing_columns = [col for col in required_columns if col in recommendations.columns]
        
        # If price is missing, try to get it from the original metadata
        if 'price' not in existing_columns and 'price' in self.meta_df.columns:
            recommendations = recommendations.merge(
                self.meta_df[['asin', 'price']],
                on='asin', how='left'
            )
            existing_columns = [col for col in required_columns if col in recommendations.columns]
        
        # If description is missing, try to get it from the original metadata
        if 'description' not in existing_columns and 'description' in self.meta_df.columns:
            recommendations = recommendations.merge(
                self.meta_df[['asin', 'description']],
                on='asin', how='left'
            )
            existing_columns = [col for col in required_columns if col in recommendations.columns]
        
        # Create missing columns with placeholder values
        for col in required_columns:
            if col not in existing_columns:
                recommendations[col] = "Not available"
        
        # Format price as string with currency
        if 'price' in recommendations.columns:
            recommendations['price'] = recommendations['price'].apply(
                lambda x: f"${float(x):.2f}" if pd.notnull(x) and str(x).replace('.', '').isdigit() else "Price not available"
            )
        
        # Truncate description if it's too long
        if 'description' in recommendations.columns:
            recommendations['description'] = recommendations['description'].apply(
                lambda x: str(x)[:200] + "..." if isinstance(x, str) and len(str(x)) > 200 else x
            )
        
        # Reorder columns and select only required ones
        return recommendations[required_columns + [col for col in recommendations.columns 
                                                  if col not in required_columns]]

# Custom CSS for the app
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
    .product-image {
        width: 100%;
        height: 150px;
        object-fit: contain;
        margin-bottom: 15px;
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
    .product-detail-image {
        max-width: 100%;
        max-height: 400px;
        object-fit: contain;
        margin-bottom: 20px;
    }
    .review-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: white;
    }
    .review-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .reviewer-name {
        font-weight: bold;
    }
    .review-summary {
        font-style: italic;
        color: #555;
        margin-bottom: 10px;
    }
    .review-text {
        color: #333;
    }
    .review-rating {
        color: #f39c12;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .review-vote {
        color: #888;
        font-size: 0.9em;
        text-align: right;
        margin-top: 5px;
    }
    .view-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 14px;
        margin: 4px 2px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .view-button:hover {
        background-color: #45a049;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    .main-title {
        text-align: center;
        color: #1E3A8A;
        margin-bottom: 30px;
    }
    .sidebar-nav {
        padding: 15px;
    }
    .algorithm-score {
        background-color: #f0f8ff;
        border-radius: 4px;
        padding: 5px 10px;
        margin: 5px 0;
        font-size: 0.9em;
    }
    .score-yes {
        color: #28a745;
    }
    .score-no {
        color: #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to optimize dataframe memory
def optimize_dataframe(df):
    """Optimize dataframe for memory efficiency."""
    for col in df.columns:
        # Convert object columns with low cardinality to category
        if df[col].dtype == 'object':
            num_unique = df[col].nunique()
            num_total = len(df)
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
        
        # Downcast numeric columns
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df

# Add this function to subsample the data for faster processing
def subsample_data(meta_df, reviews_df, sample_pct=0.1):
    """Take a smaller sample of the data for faster processing."""
    if sample_pct >= 1.0:
        return meta_df, reviews_df
    
    # Sample meta data
    meta_sample = meta_df.sample(frac=sample_pct, random_state=42)
    
    # Only keep reviews for products in the sample
    reviews_sample = reviews_df[reviews_df['asin'].isin(meta_sample['asin'])]
    
    return meta_sample, reviews_sample

# Initialize the recommendation system
@st.cache_resource(hash_funcs={pd.DataFrame: lambda _: None})
def load_recommendation_system():
    try:
        # Define data directory and file paths
        data_dir = "data"
        meta_path = os.path.join(data_dir, "out_Meta.json")
        reviews_path = os.path.join(data_dir, "out_Appliances5.json")
        
        # Check if files exist
        if not os.path.exists(meta_path) or not os.path.exists(reviews_path):
            st.error(f"Data files not found at {meta_path} or {reviews_path}")
            st.info("Make sure you've downloaded the files first.")
            
            # Display the current working directory and available files for debugging
            cwd = os.getcwd()
            st.write(f"Current working directory: {cwd}")
            
            if os.path.exists(data_dir):
                files = os.listdir(data_dir)
                st.write(f"Files in {data_dir}: {files}")
            else:
                st.warning(f"Data directory '{data_dir}' does not exist")
                
            # Try alternate paths
            alt_meta_path = "out_Meta.json"
            alt_reviews_path = "out_Appliances5.json"
            
            if os.path.exists(alt_meta_path) and os.path.exists(alt_reviews_path):
                st.info("Found files in the root directory, using those instead.")
                meta_path = alt_meta_path
                reviews_path = alt_reviews_path
            else:
                st.stop()
        
        with st.spinner("Loading data..."):
            try:
                # Load data with proper paths
                meta_df = pd.read_json(meta_path, lines=True)
                reviews_df = pd.read_json(reviews_path, lines=True)
                
                # Get sample sizes from session state if available
                use_sample = st.session_state.get('use_sample', False)
                sample_size = st.session_state.get('sample_size', 1.0)
                
                # Sample data if enabled
                if use_sample and sample_size < 1.0:
                    sample_count = int(len(meta_df) * sample_size)
                    st.info(f"Sampling {sample_count} products ({sample_size:.0%} of total)")
                    meta_df = meta_df.sample(sample_count)
                    # Only keep reviews for sampled products
                    reviews_df = reviews_df[reviews_df['asin'].isin(meta_df['asin'])]
                
                # Display basic info
                st.info(f"Loaded {len(meta_df)} products and {len(reviews_df)} reviews")
                
                # Initialize the recommendation system BEFORE memory optimization
                # This avoids categorical conversion issues during preprocessing
                rec_system = RecommendationSystem(meta_df, reviews_df)
                # Initialize the preprocessor first
                rec_system.preprocessor = DataPreprocessor(meta_df, reviews_df)
                # Preprocess the data
                with st.spinner("Preprocessing data..."):
                    processed_meta, processed_reviews = rec_system.preprocessor.preprocess_all()
                
                # NOW apply memory optimization to the processed data
                with st.spinner("Optimizing memory..."):
                    processed_meta = optimize_memory(processed_meta)
                    if processed_reviews is not None:
                        processed_reviews = optimize_memory(processed_reviews)
                    
                    # Show memory usage info
                    meta_memory = processed_meta.memory_usage(deep=True).sum() / (1024 * 1024)
                    reviews_memory = processed_reviews.memory_usage(deep=True).sum() / (1024 * 1024) if processed_reviews is not None else 0
                    st.info(f"Memory usage: Meta data {meta_memory:.2f} MB, Reviews {reviews_memory:.2f} MB")
                
                # Update the recommendation system with optimized data
                rec_system.meta_df = processed_meta
                rec_system.reviews_df = processed_reviews
                rec_system.preprocessor.meta_df = processed_meta
                rec_system.preprocessor.reviews_df = processed_reviews
                
            except Exception as e:
                st.error(f"Error loading or processing data: {e}")
                st.code(traceback.format_exc())
                
                # Try emergency data loading with minimal features
                st.warning("Attempting to load with minimal features...")
                meta_df = pd.read_json(meta_path, lines=True, nrows=500)  # Load only 500 products
                
                # Only load essential columns if possible
                if isinstance(meta_df, pd.DataFrame) and len(meta_df) > 0:
                    essential_cols = ['asin', 'title', 'price', 'description']
                    meta_df = meta_df[[col for col in essential_cols if col in meta_df.columns]]
                    
                    # Load minimal reviews
                    reviews_df = pd.read_json(reviews_path, lines=True, nrows=100)
                    reviews_df = reviews_df[reviews_df['asin'].isin(meta_df['asin'])]
                else:
                    st.error("Could not load data even in emergency mode.")
                    st.stop()
                
                # Create minimal recommendation system
                rec_system = RecommendationSystem(meta_df, reviews_df)
        
        # Initialize the recommendation components
        with st.spinner("Initializing recommendation system..."):
            try:
                # Initialize all the recommender components
                rec_system.text_recommender = TextSimilarityRecommender(rec_system.meta_df)
                rec_system.text_recommender.initialize()
                
                rec_system.itemset_recommender = FrequentItemsetRecommender(rec_system.meta_df)
                
                rec_system.cluster_recommender = ClusteringRecommender(rec_system.meta_df)
                
                rec_system.influence_recommender = InfluentialProductsRecommender(rec_system.meta_df)
                rec_system.influence_recommender.initialize()
                
                if rec_system.reviews_df is not None:
                    rec_system.cf_recommender = CollaborativeFilteringRecommender(rec_system.meta_df, rec_system.reviews_df)
                    rec_system.cf_recommender.initialize()
                
                rec_system.hybrid_recommender = HybridRecommender(rec_system.meta_df, rec_system.reviews_df)
                rec_system.hybrid_recommender.initialize()
                
                rec_system.popularity_recommender = PopularityRecommender(rec_system.meta_df, rec_system.reviews_df)
                
                rec_system.is_initialized = True
                
            except Exception as e:
                st.error(f"Error initializing recommendation system: {e}")
                st.code(traceback.format_exc())
                
                # Create a minimal recommendation system
                st.warning("Creating minimal recommendation system...")
                
                # Define a minimal class to replace the full system
                class MinimalRecommendationSystem:
                    def __init__(self, meta_df, reviews_df):
                        self.meta_df = meta_df
                        self.reviews_df = reviews_df
                        self.is_initialized = True
                    
                    def get_popular_items(self, top_n=10):
                        # Simple random selection as fallback
                        return self.meta_df.sample(min(top_n, len(self.meta_df)))
                    
                    def get_recommendations_by_algorithm(self, asin, user_id=None, top_n=5, algorithm=None):
                        # Simple category-based recommendation as fallback
                        product = self.meta_df[self.meta_df['asin'] == asin]
                        if product.empty:
                            return self.get_popular_items(top_n)
                        
                        # Try to use category if available
                        if 'category' in self.meta_df.columns:
                            try:
                                category = product['category'].iloc[0]
                                same_category = self.meta_df[
                                    (self.meta_df['category'] == category) & 
                                    (self.meta_df['asin'] != asin)
                                ]
                                if len(same_category) >= top_n:
                                    return same_category.sample(min(top_n, len(same_category)))
                            except:
                                pass
                        
                        # Fallback to random
                        return self.meta_df[self.meta_df['asin'] != asin].sample(
                            min(top_n, len(self.meta_df)-1)
                        )
                
                rec_system = MinimalRecommendationSystem(rec_system.meta_df, rec_system.reviews_df)
            
        return rec_system
    except Exception as e:
        st.error(f"Fatal error in recommendation system: {e}")
        st.code(traceback.format_exc())
        st.stop()

def create_product_card(product, container, index):
    """
    Create a product card for display
    
    Parameters:
    - product: Dict with product info
    - container: Either a column object from st.columns() or st itself
    - index: Unique index for button keys
    """
    asin = product.get('asin', 'Unknown')
    title = product.get('title', 'Unknown Product')
    price = product.get('price', 'Price unavailable')
    description = product.get('description', 'No description available')
    image_url = product.get('imageURL', None)
    
    # Check if image_url is a pandas Series or array and extract scalar value if needed
    if hasattr(image_url, '__iter__') and not isinstance(image_url, (str, list)):
        # Convert Series/array to scalar
        if len(image_url) > 0:
            image_url = image_url.iloc[0] if hasattr(image_url, 'iloc') else image_url[0]
        else:
            image_url = None
    
    # Use Amazon logo as fallback instead of placeholder
    if image_url is None or (isinstance(image_url, (float, int)) and pd.isna(image_url)):
        image_url = AMAZON_LOGO_URL
        
    # Multiple URLs - take the first one
    if isinstance(image_url, list) and len(image_url) > 0:
        image_url = image_url[0]
    
    # Truncate title and description if too long
    if isinstance(title, str) and len(title) > 50:
        title_display = title[:50] + "..."
    else:
        title_display = title
        
    if isinstance(description, str) and len(description) > 100:
        desc_display = description[:100] + "..."
    else:
        desc_display = description
    
    # Format price correctly
    if isinstance(price, (int, float)):
        price_display = f"${float(price):.2f}"
    else:
        price_display = price if isinstance(price, str) else "Price unavailable"
    
    # Create the card
    container.markdown(f"""
    <div class="product-card">
        <img src="{image_url}" class="product-image" alt="{title_display}">
        <div class="product-title">{title_display}</div>
        <div class="product-price">{price_display}</div>
        <div class="product-description">{desc_display}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a button to view product details
    # Use a unique key by adding an index parameter
    button_key = f"view_{asin}_{index}"
    if container.button(f"View Details", key=button_key):
        # Record click for recommendations
        rec_system = load_recommendation_system()
        rec_system.record_user_click(st.session_state.user_id, asin)
        
        # Store current product in session state and change page
        st.session_state.current_product = asin
        st.session_state.current_page = 'product_detail'
        st.rerun()



def show_product_detail():
    rec_system = load_recommendation_system()
    product = rec_system.meta_df[rec_system.meta_df['asin'] == st.session_state.current_product]
    
    if product.empty:
        st.error("Product not found!")
        return
    
    # Extract product information
    asin = product['asin'].iloc[0]
    title = product['title'].iloc[0] if 'title' in product.columns else 'Unknown Product'
    description = product['description'].iloc[0] if 'description' in product.columns else 'No description available'
    
    # Get as dictionary first to match homepage behavior
    product_dict = product.iloc[0].to_dict()
    price = product_dict.get('price', 'Price unavailable')
    
    # Handle price the same way as in create_product_card function
    if isinstance(price, (int, float)):
        price_display = f"${float(price):.2f}"
    else:
        try:
            # Try to convert string to float (handles "$123.45" format)
            if isinstance(price, str) and '$' in price:
                price = float(price.replace('$', '').replace(',', ''))
                price_display = f"${price:.2f}"
            else:
                price_display = price if isinstance(price, str) else "Price unavailable"
        except (ValueError, TypeError):
            price_display = "Price unavailable"
    
    # Image handling - use Amazon logo as fallback
    if 'imageURL' in product.columns:
        image_url = product['imageURL'].iloc[0]
    else:
        image_url = None
    
    # Handle multiple images or missing image - using Amazon logo instead of placeholder
    if image_url is None or (hasattr(image_url, '__iter__') and not isinstance(image_url, str) and len(image_url) == 0):
        AMAZON_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
        image_url = AMAZON_LOGO_URL
    elif isinstance(image_url, list) and len(image_url) > 0:
        image_url = image_url[0]
    
    # Create two columns for image and details
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image_url, use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="product-detail-header">
            <div class="product-detail-title">{title}</div>
            <div class="product-detail-price">{price_display}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Product Description")
    st.markdown(f"""
    <div class="product-detail-description">
        {description}
    </div>
    """, unsafe_allow_html=True)
    
    # Show product reviews below description
    show_product_reviews(asin)
    
    # Display Recommendation Algorithm Analysis for the current product
    st.markdown("## Recommendation Algorithm Analysis")
    st.markdown("This analysis shows how different recommendation algorithms would score this product:")
    
    # Define the algorithms to display
    algorithms = [
        ("text_similarity", "Text Similarity Analysis", 
         "Evaluates similarity between product descriptions, titles, and features using TF-IDF vectorization and cosine similarity."),
        ("frequent_itemset", "Frequent Itemset Mining", 
         "Analyzes if this product appears in frequent purchase patterns and association rules."),
        ("clustering", "Clustering Analysis", 
         "Evaluates this product's membership in product clusters based on category and brand."),
        ("influential", "Influential Products", 
         "Measures this product's influence score based on how frequently it appears in recommendation lists."),
        ("collaborative_filtering", "Rating Prediction & Collaborative Filtering", 
         "Predicts how users might rate this product based on similar users' preferences."),
        ("hybrid", "Hybrid Recommendation Score", 
         "Combines all algorithm scores with appropriate weights to provide an overall recommendation score.")
    ]
    
    # Create a table for algorithm scores
    table_data = []
    
    # Get a reference product to use for checking if current product is recommended
    reference_products = rec_system.get_popular_items(1)
    if not reference_products.empty:
        reference_asin = reference_products.iloc[0]['asin']
    else:
        # Fallback to first product if no popular items
        reference_asin = rec_system.meta_df.iloc[0]['asin'] if not rec_system.meta_df.empty else None
    
    # For each algorithm, check if it would recommend the current product
    for algo_id, algo_name, algo_desc in algorithms:
        with st.expander(f"{algo_name}", expanded=False):
            st.markdown(f"**{algo_desc}**")
            
            # Check if algorithm recommends this product
            score = "none"
            recommendation_status = "Not recommended by this algorithm"
            
            if algo_id == "text_similarity" and rec_system.text_recommender and rec_system.text_recommender.is_initialized:
                # For text similarity, check similarity between current product and itself (should be 1.0)
                # or with other similar products
                if hasattr(rec_system.text_recommender, 'tfidf_array'):
                    # Get the index of current product
                    idx = rec_system.meta_df.index[rec_system.meta_df['asin'] == asin].tolist()
                    if idx:
                        # Calculate similarity with itself (should be 1.0 for exact match)
                        similarity = rec_system.text_recommender._cosine_sim_numba(
                            rec_system.text_recommender.tfidf_array[idx[0]], 
                            rec_system.text_recommender.tfidf_array[idx[0]].reshape(1, -1)
                        )[0]
                        score = f"{similarity:.3f}"
                        recommendation_status = "Maximum text similarity (self-reference)"
                
            elif algo_id == "frequent_itemset":
                # Check if product appears in any "also_buy" lists
                if 'also_buy' in rec_system.meta_df.columns:
                    # Count how many times this product appears in other products' also_buy lists
                    count = 0
                    for also_buy_list in rec_system.meta_df['also_buy']:
                        if isinstance(also_buy_list, list) and asin in also_buy_list:
                            count += 1
                        elif isinstance(also_buy_list, str) and asin in also_buy_list:
                            count += 1
                    
                    if count > 0:
                        score = f"{count} co-purchases"
                        recommendation_status = f"Appears in {count} product's co-purchase lists"
            
            elif algo_id == "clustering":
                # Check product's category and brand alignment
                # Access the values correctly as scalar values, not Series
                product_category = product['primary_category'].iloc[0] if 'primary_category' in product.columns else None
                product_brand = product['brand'].iloc[0] if 'brand' in product.columns else None
                
                if product_category is not None or product_brand is not None:
                    cluster_score = 0
                    if product_category and product_category != 'Unknown':
                        cluster_score += 0.5
                    if product_brand and product_brand != 'Unknown':
                        cluster_score += 0.5
                    
                    score = f"{cluster_score:.1f}"
                    recommendation_status = f"Category and brand similarity score: {score}"
            
            elif algo_id == "influential":
                # Check product's influence score if available
                if 'influence_score' in rec_system.meta_df.columns:
                    influence = rec_system.meta_df.loc[rec_system.meta_df['asin'] == asin, 'influence_score']
                    if not influence.empty:
                        influence_value = influence.iloc[0]
                        # Normalize to 0-1 scale
                        max_influence = rec_system.meta_df['influence_score'].max()
                        if max_influence > 0:
                            normalized = influence_value / max_influence
                            score = f"{normalized:.3f}"
                            recommendation_status = f"Relative influence: {score}"
                        else:
                            score = "0.000"
                            recommendation_status = "No influence detected"
            
            elif algo_id == "collaborative_filtering" and rec_system.cf_recommender and rec_system.cf_recommender.is_initialized:
                # Get user's rating prediction for this product
                if st.session_state.user_id in rec_system.cf_recommender.cf_mappers[0] and asin in rec_system.cf_recommender.cf_mappers[1]:
                    user_idx = rec_system.cf_recommender.cf_mappers[0][st.session_state.user_id]
                    item_idx = rec_system.cf_recommender.cf_mappers[1][asin]
                    
                    # Calculate predicted rating
                    try:
                        svd = rec_system.cf_recommender.cf_model['svd']
                        user_features = rec_system.cf_recommender.cf_model['user_features'][user_idx]
                        item_vec = svd.components_[:, item_idx]
                        pred_rating = np.dot(user_features, item_vec)
                        pred_rating = max(1, min(5, pred_rating))
                        
                        score = f"{pred_rating:.1f}/5"
                        recommendation_status = f"Predicted rating: {score}"
                    except:
                        pass
            
            elif algo_id == "hybrid" and rec_system.hybrid_recommender:
                # Calculate hybrid score for this product
                # This is simplified - actual hybrid scoring would compare with other products
                hybrid_score = 0.0
                score_components = []
                
                # Text similarity component (40%)
                if rec_system.text_recommender and rec_system.text_recommender.is_initialized:
                    idx = rec_system.meta_df.index[rec_system.meta_df['asin'] == asin].tolist()
                    if idx:
                        text_sim = 1.0  # Self-similarity is 1.0
                        hybrid_score += 0.4 * text_sim
                        score_components.append(f"Text: {text_sim:.2f}")
                
                # Category similarity (15%)
                if 'primary_category' in rec_system.meta_df.columns:
                    cat_score = 1.0  # Self-category match is 1.0
                    hybrid_score += 0.15 * cat_score
                    score_components.append(f"Category: {cat_score:.2f}")
                
                # Co-purchase similarity (15%)
                if 'also_buy' in rec_system.meta_df.columns:
                    # Simplified - check if product has co-purchases
                    also_buy = product.get('also_buy', [])
                    co_score = 1.0 if (isinstance(also_buy, list) and len(also_buy) > 0) else 0.0
                    hybrid_score += 0.15 * co_score
                    score_components.append(f"Co-purchase: {co_score:.2f}")
                
                # Sentiment component (10%)
                if 'avg_sentiment' in rec_system.meta_df.columns:
                    sentiment = rec_system.meta_df.loc[rec_system.meta_df['asin'] == asin, 'avg_sentiment']
                    if not sentiment.empty:
                        sent_score = (sentiment.iloc[0] + 1) / 2  # Normalize to [0,1]
                        hybrid_score += 0.1 * sent_score
                        score_components.append(f"Sentiment: {sent_score:.2f}")
                
                # CF component (20%)
                if rec_system.cf_recommender and rec_system.cf_recommender.is_initialized:
                    if st.session_state.user_id in rec_system.cf_recommender.cf_mappers[0] and asin in rec_system.cf_recommender.cf_mappers[1]:
                        try:
                            user_idx = rec_system.cf_recommender.cf_mappers[0][st.session_state.user_id]
                            item_idx = rec_system.cf_recommender.cf_mappers[1][asin]
                            
                            svd = rec_system.cf_recommender.cf_model['svd']
                            user_features = rec_system.cf_recommender.cf_model['user_features'][user_idx]
                            item_vec = svd.components_[:, item_idx]
                            pred_rating = np.dot(user_features, item_vec)
                            pred_rating = max(1, min(5, pred_rating))
                            
                            cf_score = (pred_rating - 1) / 4  # Normalize to [0,1]
                            hybrid_score += 0.2 * cf_score
                            score_components.append(f"CF: {cf_score:.2f}")
                        except:
                            pass
                
                if hybrid_score > 0:
                    score = f"{hybrid_score:.3f}"
                    recommendation_status = "Combined score from all algorithms: " + ", ".join(score_components)
            
            # Display the recommendation score and status
            if score != "none":
                st.markdown(f"""
                <div class="algorithm-score">
                    <span class="score-yes">✓ Score: {score}</span><br>
                    {recommendation_status}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="algorithm-score">
                    <span class="score-no">✗ Score: none</span><br>
                    This algorithm does not recommend this product
                </div>
                """, unsafe_allow_html=True)
            
            # Add to table data
            table_data.append([algo_name, score])
    
    # Back button
    if st.button("← Back to Products"):
        st.session_state.current_page = 'home'
        st.rerun()

# Function to display product reviews
def show_product_reviews(asin):
    rec_system = load_recommendation_system()
    reviews = rec_system.reviews_df[rec_system.reviews_df['asin'] == asin]
    
    if reviews.empty:
        st.info("No reviews available for this product.")
        return
    
    st.markdown("### Customer Reviews")
    
    # Sort reviews by votes (if available) or date
    if 'vote' in reviews.columns:
        try:
            # Convert vote to numeric, handling strings like "1,234"
            reviews['vote_numeric'] = reviews['vote'].apply(
                lambda x: pd.to_numeric(str(x).replace(',', ''), errors='coerce')
            )
            reviews = reviews.sort_values('vote_numeric', ascending=False).reset_index(drop=True)
        except:
            pass
    
    # Display up to 5 reviews
    for i, (_, review) in enumerate(reviews.head(5).iterrows()):
        reviewer_name = review.get('reviewerName', 'Anonymous')
        review_text = review.get('reviewText', 'No review text')
        summary = review.get('summary', '')
        vote = review.get('vote', 0)
        overall = review.get('overall', 0)
        
        # Format vote
        if isinstance(vote, (int, float)):
            vote_display = f"{int(vote)} helpful vote{'s' if vote != 1 else ''}"
        elif isinstance(vote, str) and vote:
            vote_display = f"{vote} helpful vote{'s' if vote != '1' else ''}"
        else:
            vote_display = ""
        
        # Create star rating display
        stars = "⭐" * int(overall) if overall else ""
        
        st.markdown(f"""
        <div class="review-card">
            <div class="review-header">
                <div class="reviewer-name">{reviewer_name}</div>
                <div class="review-rating">{stars} ({overall}/5)</div>
            </div>
            <div class="review-summary">{summary}</div>
            <div class="review-text">{review_text}</div>
            <div class="review-vote">{vote_display}</div>
        </div>
        """, unsafe_allow_html=True)



# Function to display all products page with pagination and popularity sorting
def show_all_products():
    rec_system = load_recommendation_system()
    
    st.markdown("<h1 class='main-title'>All Products</h1>", unsafe_allow_html=True)
    
    # Initialize pagination in session state if not already present
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 0
    
    # Get all products and calculate popularity scores
    all_products = rec_system.meta_df.copy()
    
    # Calculate popularity scores based on reviews
    if rec_system.reviews_df is not None:
        # Group by asin and calculate popularity metrics
        popularity_metrics = rec_system.reviews_df.groupby('asin').agg({
            'overall': ['count', 'mean'],
            'vote': 'sum'
        })
        
        # Flatten column names
        popularity_metrics.columns = ['review_count', 'avg_rating', 'total_votes']
        
        # Calculate popularity score (combining rating, review count, and votes)
        popularity_metrics['popularity_score'] = (
            popularity_metrics['avg_rating'] * np.log1p(popularity_metrics['review_count']) +
            np.log1p(popularity_metrics['total_votes'])
        )
        
        # Merge with product metadata
        all_products = all_products.merge(popularity_metrics, on='asin', how='left')
        all_products['popularity_score'] = all_products['popularity_score'].fillna(0)
        
        # Sort by popularity score in descending order
        all_products = all_products.sort_values('popularity_score', ascending=False)
    else:
        # If no review data, just keep the original order
        all_products['popularity_score'] = 0
    
    # Calculate total pages
    items_per_page = 10
    total_items = len(all_products)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    # Create pagination controls
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("← Previous", disabled=(st.session_state.page_number <= 0)):
            st.session_state.page_number -= 1
            st.rerun()
            
    with col2:
        # Display current page info
        st.markdown(f"""
        <div style="text-align: center;">
            <p>Page {st.session_state.page_number + 1} of {total_pages}</p>
            <p><small>Total Products: {total_items}</small></p>
            <p><small>Sorted by popularity</small></p>
        </div>
        """, unsafe_allow_html=True)
            
    with col3:
        if st.button("Next →", disabled=(st.session_state.page_number >= total_pages - 1)):
            st.session_state.page_number += 1
            st.rerun()
    
    # Get products for current page
    start_idx = st.session_state.page_number * items_per_page
    end_idx = start_idx + items_per_page
    
    # Get slice of products for current page
    page_products = all_products.iloc[start_idx:end_idx].reset_index(drop=True)
    
    # Display products in a grid (3 columns)
    if not page_products.empty:
        # Create rows with 3 columns each (for better responsive design)
        for i in range(0, len(page_products), 3):
            cols = st.columns(3)
            # Display up to 3 products in this row
            for j in range(3):
                if i + j < len(page_products):
                    product = page_products.iloc[i + j]
                    # Add unique index to make keys unique
                    create_product_card(product.to_dict(), cols[j], f"all_{start_idx + i + j}")
    else:
        st.info("No products available.")

# Function to display recommended products
def show_recommended_products():
    rec_system = load_recommendation_system()
    
    st.markdown("<h1 class='main-title'>Recommended For You</h1>", unsafe_allow_html=True)
    
    # Get user's click history
    clicked_items = rec_system.user_behavior.get_click_history(st.session_state.user_id)
    
    # If no click history, prompt user to browse products first
    if not clicked_items:
        st.info("Start by exploring our products to get personalized recommendations!")
        return
    
    # Get personalized recommendations
    recommendations = rec_system.get_recommendations_for_user(st.session_state.user_id, 30)
    
    # Display recommendations in a grid (3 columns)
    if not recommendations.empty:
        cols = st.columns(3)
        for i, (_, product) in enumerate(recommendations.iterrows()):
            # Add index to make keys unique
            create_product_card(product.to_dict(), cols[i % 3], f"rec_user_{i}")
    else:
        st.info("No recommendations available. Try browsing more products!")

# Main app
def main():
    # Define a fallback image URL for products without images
    global AMAZON_LOGO_URL
    AMAZON_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
    
    # Load CSS
    load_css()
    
    # In the main() function, where you initialize other session state variables
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 0
    
    # Initialize session state
    if 'user_id' not in st.session_state:
        st.session_state.user_id = f"user_{np.random.randint(10000, 99999)}"
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
        
    if 'current_product' not in st.session_state:
        st.session_state.current_product = None
    
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'all_products'
    
    # Create sidebar navigation
    with st.sidebar:
        st.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)
        st.title("Shopping Navigation")
        
        # Sidebar buttons
        if st.button("All Products", key="sidebar_all"):
            st.session_state.current_view = 'all_products'
            st.session_state.current_page = 'home'
            st.rerun()
        
        if st.button("Recommended Products", key="sidebar_rec"):
            st.session_state.current_view = 'recommended_products'
            st.session_state.current_page = 'home'
            st.rerun()
        
        # Add settings UI (if you implemented it)
        if 'add_settings_to_sidebar' in globals():
            add_settings_to_sidebar()
        
        # Show user ID for demonstration purposes
        st.markdown("---")
        st.markdown(f"User ID: {st.session_state.user_id}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Ensure data files are downloaded before proceeding
    if not ensure_data_files():
        return
    
    # Main content
    try:
        if st.session_state.current_page == 'home':
            # Load recommendation system
            rec_system = load_recommendation_system()
            
            if st.session_state.current_view == 'all_products':
                show_all_products()
            else:  # 'recommended_products'
                show_recommended_products()
        elif st.session_state.current_page == 'product_detail':
            # Load recommendation system
            rec_system = load_recommendation_system()
            show_product_detail()
    except Exception as e:
        st.error(f"An error occurred while running the app: {e}")
        st.code(traceback.format_exc())
        
        # Offer to switch to emergency mode
        st.warning("Your app encountered an error. Would you like to try the emergency mode?")
        
        if st.button("Try Emergency Mode"):
            # This will only work if you've created app_emergency.py
            st.info("Please deploy app_emergency.py as your main file in Streamlit Cloud to use emergency mode.")

if __name__ == "__main__":
    # Load all required classes before running app
    # For a real implementation, import your classes here or in the appropriate location
    
    # Run the app
    main()

import nltk
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    pass