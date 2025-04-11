import pandas as pd
import openai
import requests
import time
import re
import json
import os
import logging
import argparse
import concurrent.futures
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# === COMMAND LINE ARGUMENTS ===
parser = argparse.ArgumentParser(description='SEO Content Generator for Fragrances using GPT-4o')
parser.add_argument('--start', type=int, default=0, help='Start index (0-based)')
parser.add_argument('--batch', type=int, default=None, help='Batch size')
parser.add_argument('--threads', type=int, default=1, help='Number of concurrent threads')
parser.add_argument('--file', type=str, default="fragrance_products.xlsx", help='Input Excel file')
parser.add_argument('--output', type=str, default="seo_outputs", help='Output directory')
args = parser.parse_args()

# === CONFIGURATION ===
# Try to get API keys from environment variables, fall back to hardcoded values
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
WC_API_URL = os.getenv("WC_API_URL", "https://xsellpoint.com/wp-json/wc/v3/products/")
WC_CONSUMER_KEY = os.getenv("WC_CONSUMER_KEY", "ck_ad9577c47151c4fa50ca6ee85dd2a58d2d6e6e79")
WC_CONSUMER_SECRET = os.getenv("WC_CONSUMER_SECRET", "cs_dc38dd138ae05842942ed84ff64a207a4b41b109")

# WordPress Application Password credentials 
WP_USERNAME = os.getenv("WP_USERNAME", "lolotam")
WP_APP_PASSWORD = os.getenv("WP_APP_PASSWORD", "8qSB vJDo s1sJ ZJkt kYVD P66W")

# Input and output configuration
INPUT_FILE = args.file
OUTPUT_DIR = args.output
MAX_RETRIES = 3  # Maximum number of retries for API calls

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === SETUP LOGGING ===
log_file = os.path.join(OUTPUT_DIR, f"seo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def log(message):
    logging.info(message)

# === INIT OpenAI Client ===
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# === LOAD EXCEL ===
try:
    df = pd.read_excel(INPUT_FILE)
    log(f"Loaded {len(df)} products from {INPUT_FILE}")
except Exception as e:
    log(f"Error loading Excel file: {e}")
    exit(1)

# === RETRY MECHANISM ===
def retry_with_backoff(func, *args, **kwargs):
    """Execute a function with exponential backoff retry logic"""
    max_attempts = MAX_RETRIES
    attempt = 0
    
    while attempt < max_attempts:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt == max_attempts:
                raise
            
            # Calculate backoff time
            delay = min(30, 3 * (2 ** attempt))  # Cap at 30 seconds
            log(f"⚠️ Attempt {attempt} failed: {str(e)}. Retrying in {delay}s...")
            time.sleep(delay)

# === ENHANCED PROMPT TEMPLATE ===
def build_prompt(name, url):
    return f"""
You are an expert eCommerce SEO product description writer specializing in fragrance content optimization. Your task is to write a high-converting, SEO-optimized product description for this fragrance. Follow these instructions precisely.

📌 Product Information
Fragrance Name: {name}
Product Link: {url}
Competitor Websites for Research:
- https://www.fragrantica.com
- https://klinq.com
- https://www.brandatt.com
- http://tatayab.com
- https://fragrancekw.com
- https://perfumeskuwait.com
- https://en.bloomingdales.com.kw

✅ Instructions:

1. Keyword Optimization
- Research and identify high-search-volume keywords relevant to this fragrance.
- Use these keywords naturally throughout the content in <strong> tags.

2. Long Product Description (300+ words)
Create a compelling, HTML-formatted product description that includes:
- The Focus Keyword at the beginning of the content
- The Focus Keyword used multiple times throughout
- The Focus Keyword in H2, H3, or H4 subheadings
- A properly formatted HTML table for Product Info (Size, Gender, Product Type, Concentration, Brand)
- A properly formatted HTML table for Fragrance Notes (Top, Heart, Base)
- A list of Key Features (bulleted or paragraph style)
- A short history/background about this perfume or brand
- One frequently searched question with a detailed answer
- Emotional language with appropriate emojis (🌸, 💫, 🌿, 🔥, 💎, ✨)
- Six hyperlinked words (3 external to perfume databases, 3 internal from this list):
  -https://xsellpoint.com/product-category/new-arrival/
  -https://xsellpoint.com/product-category/oriental-fragrance/arabic-perfume/
  -https://xsellpoint.com/product-category/best-sellers/
  -https://xsellpoint.com/product/damou-al-dahab-edp-100ml/
  -https://xsellpoint.com/product-category/shop-by-brand/brand-international/estee-lauder/
  -https://xsellpoint.com/product-category/shop-by-brand/brand-international/jean-paul-gaultier/
  -https://xsellpoint.com/product-category/shop-by-brand/brand-international/cartier/
  -https://xsellpoint.com/product-category/shop-by-brand/brand-international/nishane/
  -https://xsellpoint.com/product-category/shop-by-brand/brand-international/xerjoff/
  -https://xsellpoint.com/product-category/shop-by-brand/brand-international/narciso-rodriguez/

IMPORTANT: You MUST format your response with EXACTLY these section headings:

🔹 Product Description (HTML Format):
[Your HTML-formatted product description as specified above]

🔹 Short Description (Max 50 words):
[A punchy, enticing summary that captures the fragrance's essence and highlights main notes]

🔹 SEO Title (Max 60 characters):
[Title with Focus Keyword, under 60 characters, with a power word, sentiment, and number]

🔹 Meta Description (Max 155 characters):
[Active voice description with Focus Keyword and clear call to action]

🔹 Alt Text for Product Images:
[Descriptive, keyword-rich alt text using the product title]

🔹 Image Title:
[Full product title]

🔹 Image Caption:
[Short, elegant caption fitting the tone of luxury fragrances]

🔹 Image Description:
[Brief 1-2 sentence description using product title and main keywords]

🔹 SEO Tags (6 High-Search Keywords):
[EXACTLY 6 high-volume keywords separated by commas]

🔹 Focus Keywords:
[4 high-search-volume keywords relevant to the fragrance, separated by commas]

DO NOT skip any of these sections. DO NOT add any explanations or additional sections.
"""

# === VALIDATE SEO KEYWORDS ===
def validate_keywords(keywords_str):
    """Validate SEO keywords for quality and reasonableness"""
    if not keywords_str:
        log("⚠️ Empty keywords string")
        return False
    
    keywords = [k.strip() for k in keywords_str.split(',')]
    
    # Check for minimum number of keywords
    if len(keywords) < 3:
        log(f"⚠️ Too few keywords: {len(keywords)}")
        return False
    
    # Check for minimum length
    if any(len(k) < 3 for k in keywords):
        log(f"⚠️ Some keywords are too short")
        return False
    
    # Check for excessive repetition
    if len(set(keywords)) < len(keywords) * 0.8:
        log(f"⚠️ Too many duplicate keywords")
        return False
    
    # Check for keyword phrases that are too long
    if any(len(k.split()) > 7 for k in keywords):
        log(f"⚠️ Some keyword phrases are too long")
        return False
    
    return True

# === PARSE GPT-4o RESPONSE ===
def parse_gpt4o_output(text):
    fields = {}
    
    # Define all the sections we want to extract
    sections = [
        ("🔹 Product Description (HTML Format):", "description"),
        ("🔹 Short Description (Max 50 words):", "short_description"),
        ("🔹 SEO Title (Max 60 characters):", "seo_title"),
        ("🔹 Meta Description (Max 155 characters):", "meta_description"),
        ("🔹 Alt Text for Product Images:", "alt_text"),
        ("🔹 Image Title:", "image_title"),
        ("🔹 Image Caption:", "image_caption"),
        ("🔹 Image Description:", "image_description"),
        ("🔹 SEO Tags (6 High-Search Keywords):", "seo_tags"),
        ("🔹 Focus Keywords:", "focus_keywords")
    ]
    
    # Extract each section
    for i, (section_header, field_name) in enumerate(sections):
        if section_header not in text:
            log(f"Warning: Could not find section '{section_header}' in response")
            fields[field_name] = ""
            continue
            
        start_idx = text.find(section_header) + len(section_header)
        
        # Find the start of the next section (if any)
        end_idx = len(text)
        if i < len(sections) - 1:
            next_section = sections[i+1][0]
            if next_section in text[start_idx:]:
                end_idx = text.find(next_section, start_idx)
        
        # Extract the content
        content = text[start_idx:end_idx].strip()
        
        # Clean up any code blocks
        if "```html" in content:
            content = re.sub(r"```html\s*", "", content)
            content = re.sub(r"\s*```", "", content)
        elif "```" in content:
            content = re.sub(r"```\s*", "", content)
            content = re.sub(r"\s*```", "", content)
        
        fields[field_name] = content
    
    # Log what was found and what's missing
    log(f"Extracted {len([v for v in fields.values() if v])} non-empty fields")
    
    missing = [field for field, value in fields.items() if not value]
    if missing:
        log(f"Missing fields: {', '.join(missing)}")
    
    # Validate keywords
    if "seo_tags" in fields and fields["seo_tags"]:
        if not validate_keywords(fields["seo_tags"]):
            log(f"⚠️ SEO tags failed validation: {fields['seo_tags']}")
    
    if "focus_keywords" in fields and fields["focus_keywords"]:
        if not validate_keywords(fields["focus_keywords"]):
            log(f"⚠️ Focus keywords failed validation: {fields['focus_keywords']}")
    
    return fields

# === SAVE OUTPUT TO FILE ===
def save_content_to_file(product_id, product_name, content):
    filename = os.path.join(OUTPUT_DIR, f"product_{product_id}_{product_name.replace(' ', '_')}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)
    log(f"📁 Saved content to {filename}")
    return filename

# === UPDATE PRODUCT PERMALINK ===
def update_product_permalink(product_id, content):
    """Update the product's permalink to include the focus keyword"""
    # Get the focus keyword
    focus_keywords = content.get("focus_keywords", "")
    if not focus_keywords:
        log(f"⚠️ No focus keywords found, cannot update permalink")
        return False
    
    # Get the first focus keyword
    primary_keyword = focus_keywords.split(",")[0].strip() if "," in focus_keywords else focus_keywords.strip()
    if not primary_keyword:
        log(f"⚠️ Empty primary keyword, cannot update permalink")
        return False
    
    # Convert it to a slug format (lowercase, hyphens instead of spaces)
    keyword_slug = primary_keyword.lower().replace(" ", "-")
    # Remove any special characters
    keyword_slug = re.sub(r'[^a-z0-9\-]', '', keyword_slug)
    
    log(f"Generated permalink slug from primary keyword: {keyword_slug}")
    
    # Prepare the data for the API request
    data = {
        "slug": keyword_slug
    }
    
    # Update the product's permalink
    try:
        def update_permalink():
            response = requests.put(
                WC_API_URL + str(product_id),
                auth=(WC_CONSUMER_KEY, WC_CONSUMER_SECRET),
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response
        
        response = retry_with_backoff(update_permalink)
        
        if response.status_code in [200, 201]:
            log(f"✅ Successfully updated product permalink to include focus keyword")
            return True
        else:
            log(f"❌ Failed to update product permalink: Status {response.status_code}")
            return False
    except Exception as e:
        log(f"❌ Exception during permalink update: {e}")
        return False

# === UPDATE PRODUCT IMAGE METADATA DIRECTLY ===
def update_image_metadata(image_id, content):
    """Update the image metadata directly using WordPress REST API"""
    
    # Check if we have the necessary image metadata
    if not any([content.get("alt_text"), content.get("image_title"), 
                content.get("image_caption"), content.get("image_description")]):
        log(f"⚠️ No image metadata found to update")
        return False
    
    # Prepare the image metadata
    image_data = {}
    
    if content.get("alt_text"):
        image_data["alt_text"] = content["alt_text"]
    
    if content.get("image_title"):
        image_data["title"] = {"rendered": content["image_title"]}
    
    if content.get("image_caption"):
        image_data["caption"] = {"rendered": content["image_caption"]}
    
    if content.get("image_description"):
        image_data["description"] = {"rendered": content["image_description"]}
    
    # Add metadata using the WP REST API format
    media_data = image_data.copy()
    
    # Add acf fields for direct access if ACF plugin is used
    media_data["acf"] = {
        "alt_text": content.get("alt_text", ""),
        "image_title": content.get("image_title", ""),
        "image_caption": content.get("image_caption", ""),
        "image_description": content.get("image_description", "")
    }
    
    # Use WordPress REST API to update the image
    wp_api_url = "https://xsellpoint.com/wp-json/wp/v2/media/"
    
    # Headers for JSON content
    headers = {
        "Content-Type": "application/json"
    }
    
    # Try using WordPress Application Password authentication
    try:
        def update_with_app_password():
            # Use WordPress Application Password authentication
            from requests.auth import HTTPBasicAuth
            auth = HTTPBasicAuth(WP_USERNAME, WP_APP_PASSWORD)
            
            response = requests.post(
                f"{wp_api_url}{image_id}",
                auth=auth,
                json=media_data,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response
        
        response = retry_with_backoff(update_with_app_password)
        
        if response.status_code in [200, 201]:
            log(f"✅ Successfully updated image metadata via WordPress Application Password")
            return True
        else:
            log(f"⚠️ Failed with WordPress Application Password: Status {response.status_code}")
    except Exception as e:
        log(f"⚠️ Error with WordPress Application Password: {str(e)}")
    
    # Try WooCommerce authentication as a fallback
    try:
        def update_with_wc_auth():
            response = requests.post(
                f"{wp_api_url}{image_id}",
                auth=(WC_CONSUMER_KEY, WC_CONSUMER_SECRET),
                json=media_data,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response
        
        response = retry_with_backoff(update_with_wc_auth)
        
        if response.status_code in [200, 201]:
            log(f"✅ Successfully updated image metadata via WooCommerce auth")
            return True
        else:
            log(f"⚠️ Failed with WooCommerce auth: Status {response.status_code}")
    except Exception as e:
        log(f"⚠️ Error with WooCommerce auth: {str(e)}")
    
    # Try a direct update through a custom endpoint if available
    try:
        custom_endpoint = "https://xsellpoint.com/wp-json/custom/v1/update-attachment-meta"
        custom_data = {
            "attachment_id": image_id,
            "alt_text": content.get("alt_text", ""),
            "title": content.get("image_title", ""),
            "caption": content.get("image_caption", ""),
            "description": content.get("image_description", ""),
            "api_key": WC_CONSUMER_SECRET  # For authentication
        }
        
        def update_with_custom_endpoint():
            custom_response = requests.post(
                custom_endpoint,
                json=custom_data,
                timeout=30
            )
            custom_response.raise_for_status()
            return custom_response
        
        custom_response = retry_with_backoff(update_with_custom_endpoint)
        
        if custom_response.status_code in [200, 201]:
            log(f"✅ Successfully updated image metadata using custom endpoint")
            return True
        else:
            log(f"❌ All methods failed to update image metadata")
            return False
    except Exception as ce:
        log(f"❌ Error with custom endpoint: {str(ce)}")
        log(f"❌ All methods failed to update image metadata")
        return False

# === UPDATE PRODUCT IN WOOCOMMERCE ===
def update_woocommerce_product(product_id, content):
    # Prepare tags from the comma-separated string
    tag_list = []
    if content.get("seo_tags"):
        tag_names = [tag.strip() for tag in content["seo_tags"].split(",")]
        tag_list = [{"name": tag} for tag in tag_names if tag]
    
    # Create meta data for SEO
    meta_data = [
        {"key": "rank_math_title", "value": content.get("seo_title", "")},
        {"key": "rank_math_description", "value": content.get("meta_description", "")},
        {"key": "rank_math_focus_keyword", "value": content.get("focus_keywords", "")},
    ]
    
    # Add direct image metadata keys that WooCommerce might recognize
    if content.get("alt_text"):
        meta_data.append({"key": "_wp_attachment_image_alt", "value": content.get("alt_text", "")})
        meta_data.append({"key": "_woocommerce_product_image_alt", "value": content.get("alt_text", "")})
        meta_data.append({"key": "alt_text", "value": content.get("alt_text", "")})
        meta_data.append({"key": "_aioseo_image_alt", "value": content.get("alt_text", "")})
        
    if content.get("image_title"):
        meta_data.append({"key": "_wp_attachment_image_title", "value": content.get("image_title", "")})
        meta_data.append({"key": "_woocommerce_product_image_title", "value": content.get("image_title", "")})
        meta_data.append({"key": "image_title", "value": content.get("image_title", "")})
        meta_data.append({"key": "_aioseo_image_title", "value": content.get("image_title", "")})
        
    if content.get("image_caption"):
        meta_data.append({"key": "_wp_attachment_image_caption", "value": content.get("image_caption", "")})
        meta_data.append({"key": "_woocommerce_product_image_caption", "value": content.get("image_caption", "")})
        meta_data.append({"key": "image_caption", "value": content.get("image_caption", "")})
        meta_data.append({"key": "_aioseo_image_caption", "value": content.get("image_caption", "")})
        
    if content.get("image_description"):
        meta_data.append({"key": "_wp_attachment_image_description", "value": content.get("image_description", "")})
        meta_data.append({"key": "_woocommerce_product_image_description", "value": content.get("image_description", "")})
        meta_data.append({"key": "image_description", "value": content.get("image_description", "")})
        meta_data.append({"key": "_aioseo_image_description", "value": content.get("image_description", "")})
    
    # Create data payload for WooCommerce API
    data = {
        "description": content.get("description", ""),
        "short_description": content.get("short_description", ""),
        "tags": tag_list,
        "meta_data": meta_data,
        "images": []
    }
    
    # First, let's try to get the current product to see the image IDs
    try:
        def get_product_data():
            get_response = requests.get(
                WC_API_URL + str(product_id),
                auth=(WC_CONSUMER_KEY, WC_CONSUMER_SECRET),
                timeout=30
            )
            get_response.raise_for_status()
            return get_response
        
        get_response = retry_with_backoff(get_product_data)
        
        if get_response.status_code == 200:
            product_data = get_response.json()
            
            # Check if there are images
            if product_data.get("images") and len(product_data["images"]) > 0:
                # Get the first image ID
                first_image_id = product_data["images"][0].get("id")
                
                if first_image_id:
                    log(f"Found existing image ID: {first_image_id}")
                    
                    # Add the image with updated metadata
                    data["images"].append({
                        "id": first_image_id,
                        "alt": content.get("alt_text", ""),
                        "name": content.get("image_title", ""),
                        "src": product_data["images"][0].get("src", ""),
                        "title": content.get("image_title", ""),
                        "caption": content.get("image_caption", ""),
                        "description": content.get("image_description", "")
                    })
                    
                    # Try to update the image metadata directly
                    update_image_metadata(first_image_id, content)
                else:
                    log("No image ID found in product data")
            else:
                log("No images found for this product")
        else:
            log(f"⚠️ Could not retrieve product data: {get_response.status_code}")
    except Exception as e:
        log(f"⚠️ Error retrieving product data: {str(e)}")
    
    # Now update the product with our data
    try:
        def update_product():
            log(f"Updating product {product_id} with WooCommerce API")
            response = requests.put(
                WC_API_URL + str(product_id),
                auth=(WC_CONSUMER_KEY, WC_CONSUMER_SECRET),
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response
        
        response = retry_with_backoff(update_product)
        
        if response.status_code in [200, 201]:
            log(f"✅ Updated product {product_id} successfully")
            log(f"  - Updated with {len(tag_list)} tags")
            log(f"  - Updated with {len(meta_data)} meta fields")
            log(f"  - Updated with {len(data['images'])} images")
            
            # Also update the permalink
            update_product_permalink(product_id, content)
            
            # Store the successful response data
            success_file = os.path.join(OUTPUT_DIR, f"success_{product_id}.json") 
            with open(success_file, "w", encoding="utf-8") as f:
                json.dump(response.json(), f, indent=2, ensure_ascii=False)
            log(f"  - Saved API response to {success_file}")
            
            return True, response.json()
        else:
            log(f"❌ Failed to update product {product_id}: Status {response.status_code}")
            log(f"Response: {response.text[:500]}...")
            
            # Save the failed API response for debugging
            error_file = os.path.join(OUTPUT_DIR, f"api_error_{product_id}.json")
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(f"Status: {response.status_code}\n\n")
                f.write(response.text)
            log(f"  - Saved error response to {error_file}")
            
            # Also save what we tried to send
            data_file = os.path.join(OUTPUT_DIR, f"api_request_{product_id}.json")
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            log(f"  - Saved request data to {data_file}")
            
            return False, response.text
    except Exception as e:
        log(f"❌ Exception during API request: {e}")
        
        # Save what we tried to send even in case of exception
        data_file = os.path.join(OUTPUT_DIR, f"api_request_exception_{product_id}.json")
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log(f"  - Saved request data to {data_file}")
        
        return False, str(e)

# === PROCESS SINGLE PRODUCT ===
def process_product(idx, row, total_products):
    pid = row['product id']
    name = row['Perfume/Product Name']
    url = row['Product URL']

    log(f"🔄 [{idx+1}/{total_products}] Processing: {name} (ID: {pid}) - {(idx+1)/total_products*100:.1f}% complete")

    try:
        # Generate content with GPT-4o
        prompt = build_prompt(name, url)
        log(f"📤 Sending request to GPT-4o for {name}")
        
        # Save prompt for debugging
        prompt_file = os.path.join(OUTPUT_DIR, f"prompt_{pid}.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        def get_gpt4o_response():
            # Using GPT-4o instead of Claude
            return client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert eCommerce SEO content writer specializing in fragrance products."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.7
            )
        
        response = retry_with_backoff(get_gpt4o_response)
        
        # Extract the response text
        result = response.choices[0].message.content
        log(f"📥 Received response from GPT-4o ({len(result)} chars)")
        
        # Save the raw response
        response_file = os.path.join(OUTPUT_DIR, f"response_{pid}.txt")
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(result)
        log(f"📄 Saved raw GPT-4o response to {response_file}")
        
        # Parse the content
        content = parse_gpt4o_output(result)
        
        # Validate content
        required_fields = ["description", "short_description", "seo_title", "meta_description", 
                         "alt_text", "image_title", "image_caption", "image_description", 
                         "seo_tags", "focus_keywords"]
        
        missing_fields = [field for field in required_fields if not content.get(field)]
        
        if missing_fields:
            log(f"⚠️ Missing required fields for {name}: {', '.join(missing_fields)}")
            # We'll still try to update with what we have
            
        # Save content to file
        save_content_to_file(pid, name, content)
        
        # Update WooCommerce
        success, response_data = update_woocommerce_product(pid, content)
        
        return success
        
    except Exception as e:
        log(f"❌ Error processing {name} (ID: {pid}): {str(e)}")
        
        # Try to save whatever we have from GPT-4o's response
        try:
            error_file = os.path.join(OUTPUT_DIR, f"exception_{pid}_{name.replace(' ', '_')}.txt")
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(f"Error: {str(e)}\n\n")
                if 'result' in locals():
                    f.write(result)
                else:
                    f.write("No response received from GPT-4o")
            log(f"📄 Saved error details to {error_file}")
        except:
            log("⚠️ Could not save error details")
        
        return False

# === PROCESS PRODUCTS (WITH THREADING) ===
def process_products(start_index=0, batch_size=None, num_threads=1):
    total_products = len(df)
    
    if batch_size is None:
        end_index = total_products
    else:
        end_index = min(start_index + batch_size, total_products)
    
    products_to_process = df.iloc[start_index:end_index]
    products_count = len(products_to_process)
    
    log(f"🚀 Starting SEO content generation for products {start_index+1}-{end_index} of {total_products}")
    
    successful = 0
    failed = 0
    
    # When using just one thread, process sequentially
    if num_threads == 1:
        for idx, (_, row) in enumerate(products_to_process.iterrows()):
            abs_idx = idx + start_index
            if process_product(abs_idx, row, total_products):
                successful += 1
            else:
                failed += 1
            # Small delay between products
            time.sleep(2)
    else:
        # Process in parallel with multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Create a list of future tasks
            future_to_idx = {
                executor.submit(process_product, idx + start_index, row, total_products): idx + start_index 
                for idx, (_, row) in enumerate(products_to_process.iterrows())
            }
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    success = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                except Exception as exc:
                    log(f"❌ Product at index {idx} generated an exception: {exc}")
                    failed += 1
    
    log(f"✅ Completed batch processing: {successful} successful, {failed} failed")
    return successful, failed

# === GENERATE SUMMARY REPORT ===
def generate_summary_report(successful, failed, start_time):
    """Generate a summary report of the SEO run"""
    end_time = datetime.now()
    duration = end_time - start_time
    total = successful + failed
    
    # Create summary report
    report = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": duration.total_seconds(),
        "duration_formatted": str(duration),
        "total_products": total,
        "successful": successful,
        "failed": failed,
        "success_rate": f"{(successful/total*100) if total > 0 else 0:.2f}%",
        "configuration": {
            "input_file": INPUT_FILE,
            "output_dir": OUTPUT_DIR,
            "threads": args.threads,
            "batch_size": args.batch,
            "start_index": args.start
        }
    }
    
    # Save report to file
    report_file = os.path.join(OUTPUT_DIR, f"summary_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    
    # Log summary
    log("\n" + "="*50)
    log("📊 SEO GENERATION SUMMARY")
    log("="*50)
    log(f"Start Time: {report['start_time']}")
    log(f"End Time:   {report['end_time']}")
    log(f"Duration:   {report['duration_formatted']}")
    log(f"Products:   {total} total, {successful} successful, {failed} failed")
    log(f"Success Rate: {report['success_rate']}")
    log("="*50)
    log(f"Full report saved to: {report_file}")
    
    return report

# === MAIN EXECUTION ===
if __name__ == "__main__":
    try:
        start_time = datetime.now()
        log("🚀 Starting Fragrance SEO Content Generator with GPT-4o")
        log(f"📊 Found {len(df)} products in Excel file")
        log(f"⚙️ Configuration: Start Index={args.start}, Batch Size={args.batch}, Threads={args.threads}")
        
        successful, failed = process_products(
            start_index=args.start, 
            batch_size=args.batch,
            num_threads=args.threads
        )
        
        generate_summary_report(successful, failed, start_time)
        
    except Exception as e:
        log(f"🔴 Critical error: {str(e)}")
        import traceback
        log(traceback.format_exc())