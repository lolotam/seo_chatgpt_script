import sys
import os
import pandas as pd
import requests
import time
import re
import json
import threading
import queue
from datetime import datetime
import logging
import openai
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, 
                           QFileDialog, QProgressBar, QComboBox, QGroupBox, QRadioButton,
                           QMessageBox, QScrollArea, QGridLayout, QTableWidget, QTableWidgetItem,
                           QHeaderView, QCheckBox, QSpinBox, QSplitter)
from PyQt5.QtGui import QFont, QIcon, QPixmap, QTextCursor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

# Configure logging with UTF-8 encoding to handle special characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"seo_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def safe_log_message(message):
    """Convert Unicode symbols to ASCII alternatives for Windows compatibility"""
    if isinstance(message, str):
        message = message.replace("✅", "[SUCCESS]")
        message = message.replace("❌", "[ERROR]")
        message = message.replace("⚠️", "[WARNING]")
    return message

class LogHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        # Make sure the message is Windows-compatible
        msg = safe_log_message(msg)
        self.text_widget.append(msg)
        self.text_widget.moveCursor(QTextCursor.End)

class WorkerThread(QThread):
    update_progress = pyqtSignal(int)
    update_status = pyqtSignal(str)
    finished_product = pyqtSignal(dict)
    finished_all = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue = queue.Queue()
        self.is_running = True
        self.api_config = {}
        self.model_config = {}
        self.total_products = 0
        self.processed_products = 0

    def add_task(self, task_type, data, api_config, model_config):
        self.api_config = api_config
        self.model_config = model_config
        self.queue.put((task_type, data))
        self.total_products = len(data) if isinstance(data, list) else 1
        self.processed_products = 0
        
    def stop(self):
        self.is_running = False
        
    def run(self):
        while self.is_running:
            try:
                if not self.queue.empty():
                    task_type, data = self.queue.get(block=False)
                    
                    if task_type == "bulk":
                        self.process_bulk(data)
                    elif task_type == "single":
                        self.process_single(data)
                        
                    self.finished_all.emit()
                else:
                    time.sleep(0.1)  # Sleep to reduce CPU usage
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                self.update_status.emit(f"[ERROR] Error in worker thread: {str(e)}")
                logger.error(f"Error in worker thread: {str(e)}")
    
    def process_bulk(self, products):
        for i, product in enumerate(products):
            try:
                pid = product.get('product_id', 'N/A')
                name = product.get('name', 'Unknown')
                url = product.get('url', '')
                
                self.update_status.emit(f"Processing ({i+1}/{len(products)}): {name}")
                
                seo_content = self.generate_seo_content(name, url)
                if seo_content:
                    if self.api_config.get('update_woocommerce', False):
                        status, response = self.update_woocommerce(pid, seo_content)
                        result_status = f"[SUCCESS] Updated in WooCommerce" if status in [200, 201] else f"[ERROR] Update failed: {status}"
                    else:
                        result_status = "[SUCCESS] Generated content (WooCommerce update disabled)"
                    
                    result = {
                        'product_id': pid,
                        'name': name,
                        'content': seo_content,
                        'status': result_status
                    }
                    self.finished_product.emit(result)
                
                self.processed_products += 1
                progress = int((self.processed_products / self.total_products) * 100)
                self.update_progress.emit(progress)
                
                # Wait between requests to avoid rate limits
                time.sleep(self.model_config.get('request_delay', 2))
                
            except Exception as e:
                self.update_status.emit(f"[ERROR] Error processing {name}: {str(e)}")
                logger.error(f"Error processing {name}: {str(e)}")
    
    def process_single(self, product):
        try:
            pid = product.get('product_id', 'N/A')
            name = product.get('name', '')
            url = product.get('url', '')
            
            self.update_status.emit(f"Processing single product: {name}")
            
            seo_content = self.generate_seo_content(name, url)
            if seo_content:
                if self.api_config.get('update_woocommerce', False):
                    status, response = self.update_woocommerce(pid, seo_content)
                    result_status = f"[SUCCESS] Updated in WooCommerce" if status in [200, 201] else f"[ERROR] Update failed: {status}"
                else:
                    result_status = "[SUCCESS] Generated content (WooCommerce update disabled)"
                
                result = {
                    'product_id': pid,
                    'name': name,
                    'content': seo_content,
                    'status': result_status
                }
                self.finished_product.emit(result)
            
            self.update_progress.emit(100)
            
        except Exception as e:
            self.update_status.emit(f"[ERROR] Error processing {name}: {str(e)}")
            logger.error(f"Error processing {name}: {str(e)}")
    
    def generate_seo_content(self, name, url):
        self.update_status.emit(f"Generating SEO content for: {name}")
        
        prompt = self.build_prompt(name, url)
        
        # Using GPT-4o model
        return self.generate_with_gpt(prompt)
    
    def generate_with_gpt(self, prompt):
        try:
            api_key = self.model_config.get('openai_api_key', '')
            if not api_key:
                self.update_status.emit("[ERROR] OpenAI API key not configured")
                return None
                
            client = openai.OpenAI(api_key=api_key)
            
            self.update_status.emit("Sending request to GPT-4o...")
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert eCommerce SEO content writer specializing in fragrance products."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.7
            )
            
            result = response.choices[0].message.content
            
            # Parse GPT's response
            return self.parse_gpt_output(result)
            
        except Exception as e:
            self.update_status.emit(f"[ERROR] Error generating content with GPT-4o: {str(e)}")
            logger.error(f"Error generating content with GPT-4o: {str(e)}")
            return None
    
    def build_prompt(self, name, url):
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

    def parse_gpt_output(self, text):
        """Parse GPT-4o's response into structured fields."""
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
                self.update_status.emit(f"Warning: Could not find section '{section_header}' in response")
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
        
        return fields
    
    def update_woocommerce(self, product_id, content):
        """Update a WooCommerce product with the generated SEO content."""
        try:
            wc_url = self.api_config.get('wc_api_url', '')
            wc_key = self.api_config.get('wc_consumer_key', '')
            wc_secret = self.api_config.get('wc_consumer_secret', '')
            
            if not all([wc_url, wc_key, wc_secret]):
                self.update_status.emit("[ERROR] WooCommerce API credentials not configured")
                return 400, {"error": "WooCommerce API credentials not configured"}
            
            # Fix URL formatting by removing any spaces and ensuring proper format
            wc_url = wc_url.strip().replace(" ", "")
            
            # Ensure URL ends with slash
            if not wc_url.endswith('/'):
                wc_url += '/'
                
            # Validate URL format
            if not wc_url.startswith('http'):
                self.update_status.emit("[ERROR] Invalid WooCommerce API URL format. Must start with http:// or https://")
                return 400, {"error": "Invalid URL format"}
            
            # Process tags - split by commas and clean up
            tags_raw = content.get("seo_tags", "")
            if isinstance(tags_raw, str):
                # Remove any markdown formatting and split by commas
                tags_clean = re.sub(r'\*\*|\*', '', tags_raw)
                tags_list = [tag.strip() for tag in tags_clean.split(',') if tag.strip()]
            else:
                tags_list = []
            
            # Prepare the data for WooCommerce API
            data = {
                "description": content.get("description", ""),
                "short_description": content.get("short_description", ""),
                "tags": [{"name": tag} for tag in tags_list],
                "meta_data": [
                    {"key": "rank_math_title", "value": content.get("seo_title", "")},
                    {"key": "rank_math_description", "value": content.get("meta_description", "")},
                    {"key": "rank_math_focus_keyword", "value": content.get("focus_keywords", "")},
                    {"key": "_wp_attachment_image_alt", "value": content.get("alt_text", "")},
                    {"key": "image_title", "value": content.get("image_title", "")},
                    {"key": "image_caption", "value": content.get("image_caption", "")},
                    {"key": "image_description", "value": content.get("image_description", "")}
                ]
            }
            
            # Check for images data and add if found
            try:
                # Try to get the current product to see if it has images
                get_response = requests.get(
                    wc_url + str(product_id),
                    auth=(wc_key, wc_secret),
                    timeout=30
                )
                
                if get_response.status_code == 200:
                    product_data = get_response.json()
                    
                    # Check if there are images
                    if product_data.get("images") and len(product_data["images"]) > 0:
                        # Get the first image ID
                        first_image_id = product_data["images"][0].get("id")
                        
                        if first_image_id:
                            self.update_status.emit(f"Found image ID: {first_image_id} for product")
                            
                            # Add the image with updated metadata
                            data["images"] = [{
                                "id": first_image_id,
                                "alt": content.get("alt_text", ""),
                                "name": content.get("image_title", ""),
                                "title": content.get("image_title", ""),
                                "caption": content.get("image_caption", ""),
                                "description": content.get("image_description", "")
                            }]
            except Exception as img_e:
                self.update_status.emit(f"[WARNING] Error getting product images: {str(img_e)}")
                # Continue without image data
            
            # Make sure the product ID is valid
            if not product_id or product_id == 'N/A':
                self.update_status.emit("[WARNING] No valid product ID, skipping WooCommerce update")
                return 400, {"error": "No valid product ID"}

            try:
                api_url = f"{wc_url}{product_id}"
                self.update_status.emit(f"Sending update to WooCommerce: {api_url}")
                response = requests.put(
                    api_url,
                    auth=(wc_key, wc_secret),
                    json=data,
                    timeout=30
                )
                
                if response.status_code not in [200, 201]:
                    self.update_status.emit(f"[ERROR] API Error: {response.status_code} - {response.text[:300]}")
                else:
                    # Try to update the permalink based on focus keyword
                    self.update_product_permalink(product_id, content)
                    
                return response.status_code, response.json()
            except requests.exceptions.RequestException as e:
                self.update_status.emit(f"[ERROR] Request failed: {str(e)}")
                return 500, {"error": str(e)}
                
        except Exception as e:
            self.update_status.emit(f"[ERROR] Error updating product: {str(e)}")
            return 500, {"error": str(e)}
    
    def update_product_permalink(self, product_id, content):
        """Update the product's permalink to include the focus keyword"""
        try:
            # Get the focus keyword
            focus_keywords = content.get("focus_keywords", "")
            if not focus_keywords:
                self.update_status.emit(f"⚠️ No focus keywords found, cannot update permalink")
                return False
            
            # Get the first focus keyword
            primary_keyword = focus_keywords.split(",")[0].strip() if "," in focus_keywords else focus_keywords.strip()
            if not primary_keyword:
                self.update_status.emit(f"⚠️ Empty primary keyword, cannot update permalink")
                return False
            
            # Convert it to a slug format (lowercase, hyphens instead of spaces)
            keyword_slug = primary_keyword.lower().replace(" ", "-")
            # Remove any special characters
            keyword_slug = re.sub(r'[^a-z0-9\-]', '', keyword_slug)
            
            self.update_status.emit(f"Generated permalink slug from primary keyword: {keyword_slug}")
            
            # Prepare the data for the API request
            data = {
                "slug": keyword_slug
            }
            
            # Get WooCommerce credentials
            wc_url = self.api_config.get('wc_api_url', '')
            wc_key = self.api_config.get('wc_consumer_key', '')
            wc_secret = self.api_config.get('wc_consumer_secret', '')
            
            # Update the product's permalink
            response = requests.put(
                f"{wc_url}{product_id}",
                auth=(wc_key, wc_secret),
                json=data,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                self.update_status.emit(f"✅ Successfully updated product permalink to include focus keyword")
                return True
            else:
                self.update_status.emit(f"❌ Failed to update product permalink: Status {response.status_code}")
                return False
        except Exception as e:
            self.update_status.emit(f"❌ Exception during permalink update: {e}")
            return False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Fragrance SEO Content Generator")
        self.setGeometry(100, 100, 1200, 800)
        
        self.results = []
        
        # Initialize worker thread
        self.worker = WorkerThread()
        self.worker.update_progress.connect(self.update_progress)
        self.worker.update_status.connect(self.update_status)
        self.worker.finished_product.connect(self.add_result)
        self.worker.finished_all.connect(self.processing_complete)
        self.worker.start()
        
        # Load saved settings if available
        self.settings = self.load_settings()
        
        self.init_ui()
    
    def init_ui(self):
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Add tabs
        tab_single = QWidget()
        tab_bulk = QWidget()
        tab_settings = QWidget()
        tab_results = QWidget()
        
        tab_widget.addTab(tab_single, "Single Product")
        tab_widget.addTab(tab_bulk, "Bulk Processing")
        tab_widget.addTab(tab_settings, "Settings")
        tab_widget.addTab(tab_results, "Results")
        
        # Setup each tab
        self.setup_single_tab(tab_single)
        self.setup_bulk_tab(tab_bulk)
        self.setup_settings_tab(tab_settings)
        self.setup_results_tab(tab_results)
        
        main_layout.addWidget(tab_widget)
        
        # Status bar at the bottom
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        status_layout.addWidget(self.status_label, 7)
        status_layout.addWidget(self.progress_bar, 3)
        
        main_layout.addLayout(status_layout)
        
        # Load settings to UI
        self.apply_saved_settings()
    
    def setup_single_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Product info group
        product_group = QGroupBox("Product Information")
        product_layout = QGridLayout()
        
        product_layout.addWidget(QLabel("Product Name:"), 0, 0)
        self.single_name_input = QLineEdit()
        product_layout.addWidget(self.single_name_input, 0, 1)
        
        product_layout.addWidget(QLabel("Product URL:"), 1, 0)
        self.single_url_input = QLineEdit()
        product_layout.addWidget(self.single_url_input, 1, 1)
        
        product_layout.addWidget(QLabel("Product ID (for WooCommerce):"), 2, 0)
        self.single_id_input = QLineEdit()
        product_layout.addWidget(self.single_id_input, 2, 1)
        
        product_group.setLayout(product_layout)
        layout.addWidget(product_group)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        self.single_generate_btn = QPushButton("Generate SEO Content")
        self.single_generate_btn.clicked.connect(self.process_single_product)
        btn_layout.addWidget(self.single_generate_btn)
        
        self.single_clear_btn = QPushButton("Clear Fields")
        self.single_clear_btn.clicked.connect(self.clear_single_fields)
        btn_layout.addWidget(self.single_clear_btn)
        
        layout.addLayout(btn_layout)
        
        # Console output
        layout.addWidget(QLabel("Status:"))
        self.single_console = QTextEdit()
        self.single_console.setReadOnly(True)
        layout.addWidget(self.single_console)
        
        # Connect log handler
        log_handler = LogHandler(self.single_console)
        logger.addHandler(log_handler)
    
    def setup_bulk_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # File selection group
        file_group = QGroupBox("Excel File")
        file_layout = QHBoxLayout()
        
        self.file_path_input = QLineEdit()
        self.file_path_input.setReadOnly(True)
        file_layout.addWidget(self.file_path_input, 7)
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_excel_file)
        file_layout.addWidget(self.browse_btn, 1)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Excel preview
        preview_group = QGroupBox("Excel Preview")
        preview_layout = QVBoxLayout()
        
        self.excel_table = QTableWidget()
        self.excel_table.setColumnCount(3)
        self.excel_table.setHorizontalHeaderLabels(["Product ID", "Product Name", "URL"])
        self.excel_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        preview_layout.addWidget(self.excel_table)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        self.load_excel_btn = QPushButton("Load Excel")
        self.load_excel_btn.clicked.connect(self.load_excel_preview)
        btn_layout.addWidget(self.load_excel_btn)
        
        self.process_bulk_btn = QPushButton("Process All Products")
        self.process_bulk_btn.clicked.connect(self.process_bulk_products)
        self.process_bulk_btn.setEnabled(False)
        btn_layout.addWidget(self.process_bulk_btn)
        
        layout.addLayout(btn_layout)
        
        # Console output
        layout.addWidget(QLabel("Status:"))
        self.bulk_console = QTextEdit()
        self.bulk_console.setReadOnly(True)
        layout.addWidget(self.bulk_console)
        
        # Connect log handler
        log_handler = LogHandler(self.bulk_console)
        logger.addHandler(log_handler)
    
    def setup_settings_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # WooCommerce API settings
        wc_group = QGroupBox("WooCommerce API Settings")
        wc_layout = QGridLayout()
        
        wc_layout.addWidget(QLabel("API URL:"), 0, 0)
        self.wc_url_input = QLineEdit()
        self.wc_url_input.setPlaceholderText("https://yourstore.com/wp-json/wc/v3/products/")
        wc_layout.addWidget(self.wc_url_input, 0, 1)
        
        wc_layout.addWidget(QLabel("Consumer Key:"), 1, 0)
        self.wc_key_input = QLineEdit()
        wc_layout.addWidget(self.wc_key_input, 1, 1)
        
        wc_layout.addWidget(QLabel("Consumer Secret:"), 2, 0)
        self.wc_secret_input = QLineEdit()
        wc_layout.addWidget(self.wc_secret_input, 2, 1)
        
        self.update_wc_checkbox = QCheckBox("Update WooCommerce products")
        self.update_wc_checkbox.setChecked(True)
        wc_layout.addWidget(self.update_wc_checkbox, 3, 0, 1, 2)
        
        wc_group.setLayout(wc_layout)
        layout.addWidget(wc_group)
        
        # API key settings
        model_group = QGroupBox("API Settings")
        model_layout = QGridLayout()
        
        model_layout.addWidget(QLabel("OpenAI API Key:"), 0, 0)
        self.openai_key_input = QLineEdit()
        model_layout.addWidget(self.openai_key_input, 0, 1)
        
        model_layout.addWidget(QLabel("Request Delay (seconds):"), 1, 0)
        self.delay_spinbox = QSpinBox()
        self.delay_spinbox.setRange(1, 10)
        self.delay_spinbox.setValue(2)
        model_layout.addWidget(self.delay_spinbox, 1, 1)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Excel settings
        excel_group = QGroupBox("Excel File Settings")
        excel_layout = QGridLayout()
        
        excel_layout.addWidget(QLabel("Product ID Column Name:"), 0, 0)
        self.id_column_input = QLineEdit("product id")
        excel_layout.addWidget(self.id_column_input, 0, 1)
        
        excel_layout.addWidget(QLabel("Product Name Column Name:"), 1, 0)
        self.name_column_input = QLineEdit("Perfume/Product Name")
        excel_layout.addWidget(self.name_column_input, 1, 1)
        
        excel_layout.addWidget(QLabel("URL Column Name:"), 2, 0)
        self.url_column_input = QLineEdit("Product URL")
        excel_layout.addWidget(self.url_column_input, 2, 1)
        
        excel_group.setLayout(excel_layout)
        layout.addWidget(excel_group)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        btn_layout.addWidget(save_btn)
        
        test_connection_btn = QPushButton("Test WooCommerce Connection")
        test_connection_btn.clicked.connect(self.test_woocommerce_connection)
        btn_layout.addWidget(test_connection_btn)
        
        layout.addLayout(btn_layout)
        
        # Status
        self.settings_status = QLabel("")
        layout.addWidget(self.settings_status)
        
        # Add some stretch at the bottom
        layout.addStretch()
    
    def setup_results_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Results controls
        controls_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        controls_layout.addWidget(self.export_btn)
        
        self.clear_results_btn = QPushButton("Clear Results")
        self.clear_results_btn.clicked.connect(self.clear_results)
        controls_layout.addWidget(self.clear_results_btn)
        
        layout.addLayout(controls_layout)
        
        # Results viewer
        self.results_area = QScrollArea()
        self.results_area.setWidgetResizable(True)
        
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.addStretch()
        
        self.results_area.setWidget(self.results_container)
        layout.addWidget(self.results_area)
    
    def browse_excel_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx *.xls)")
        if file_path:
            self.file_path_input.setText(file_path)
            self.load_excel_preview()
            
    def load_excel_preview(self):
        self.process_bulk_btn.setEnabled(False)
        file_path = self.file_path_input.text()
        if not file_path:
            self.update_status("Please select an Excel file first")
            return
            
        try:
            id_col = self.id_column_input.text() or "product id"
            name_col = self.name_column_input.text() or "Perfume/Product Name"
            url_col = self.url_column_input.text() or "Product URL"
            
            df = pd.read_excel(file_path)
            
            if id_col not in df.columns or name_col not in df.columns or url_col not in df.columns:
                available_cols = ", ".join(df.columns)
                self.update_status(f"[ERROR] Required columns not found in Excel file. Available columns: {available_cols}")
                return
                
            # Clear existing data
            self.excel_table.setRowCount(0)
            
            # Populate table with preview data
            for i, row in df.iterrows():
                self.excel_table.insertRow(i)
                self.excel_table.setItem(i, 0, QTableWidgetItem(str(row[id_col])))
                self.excel_table.setItem(i, 1, QTableWidgetItem(str(row[name_col])))
                self.excel_table.setItem(i, 2, QTableWidgetItem(str(row[url_col])))
            
            self.process_bulk_btn.setEnabled(True)
            self.update_status(f"[SUCCESS] Loaded {len(df)} products from Excel file")
            
        except Exception as e:
            self.update_status(f"[ERROR] Error loading Excel file: {str(e)}")
            logger.error(f"Error loading Excel file: {str(e)}")
            
    def process_single_product(self):
        name = self.single_name_input.text().strip()
        url = self.single_url_input.text().strip()
        product_id = self.single_id_input.text().strip()
        
        if not name or not url:
            self.update_status("Please enter both product name and URL")
            return
            
        # Get API settings
        api_config = self.get_api_config()
        model_config = self.get_model_config()
        
        # Prepare data
        product_data = {
            'product_id': product_id,
            'name': name,
            'url': url
        }
        
        # Add task to worker queue
        self.worker.add_task("single", product_data, api_config, model_config)
        
        self.update_status(f"Processing single product: {name}")
        self.progress_bar.setValue(0)
        
    def process_bulk_products(self):
        file_path = self.file_path_input.text()
        if not file_path:
            self.update_status("Please select an Excel file first")
            return
            
        try:
            id_col = self.id_column_input.text() or "product id"
            name_col = self.name_column_input.text() or "Perfume/Product Name"
            url_col = self.url_column_input.text() or "Product URL"
            
            df = pd.read_excel(file_path)
            
            if id_col not in df.columns or name_col not in df.columns or url_col not in df.columns:
                available_cols = ", ".join(df.columns)
                self.update_status(f"[ERROR] Required columns not found in Excel file. Available columns: {available_cols}")
                return
            
            # Get API settings
            api_config = self.get_api_config()
            model_config = self.get_model_config()
            
            # Prepare data
            products = []
            for _, row in df.iterrows():
                products.append({
                    'product_id': str(row[id_col]),
                    'name': str(row[name_col]),
                    'url': str(row[url_col])
                })
            
            # Add task to worker queue
            self.worker.add_task("bulk", products, api_config, model_config)
            
            self.update_status(f"Starting bulk processing of {len(products)} products")
            self.progress_bar.setValue(0)
            
        except Exception as e:
            self.update_status(f"[ERROR] Processing Excel file: {str(e)}")
            logger.error(f"Error processing Excel file: {str(e)}")
            
    def get_api_config(self):
        """Get WooCommerce API configuration with proper URL formatting"""
        wc_url = self.wc_url_input.text().strip()
        
        # Remove any accidental spaces in URL
        wc_url = wc_url.replace(" ", "")
        
        # Ensure URL has proper format
        if wc_url and not wc_url.startswith(('http://', 'https://')):
            # Add https if missing
            wc_url = 'https://' + wc_url
        
        # Ensure URL ends with slash
        if wc_url and not wc_url.endswith('/'):
            wc_url += '/'
            
        return {
            'wc_api_url': wc_url,
            'wc_consumer_key': self.wc_key_input.text().strip(),
            'wc_consumer_secret': self.wc_secret_input.text().strip(),
            'update_woocommerce': self.update_wc_checkbox.isChecked()
        }
        
    def get_model_config(self):
        return {
            'openai_api_key': self.openai_key_input.text(),
            'request_delay': self.delay_spinbox.value()
        }
        
    def update_progress(self, value):
        """Update the progress bar with the given value"""
        self.progress_bar.setValue(value)
        
    def update_status(self, message):
        """Update status label and log the message"""
        self.status_label.setText(message)
        
        # Convert Unicode symbols to ASCII for logging
        safe_message = safe_log_message(message)
        logger.info(safe_message)
        
    def add_result(self, result):
        # Add to results list
        self.results.append(result)
        
        # Create a result widget
        result_widget = QGroupBox(f"{result['name']}")
        result_layout = QVBoxLayout()
        
        # Status label
        status_label = QLabel(f"Status: {result['status']}")
        result_layout.addWidget(status_label)
        
        # Content tabs
        content_tabs = QTabWidget()
        
        # Add tabs for each content type
        for content_type in ['description', 'short_description', 'seo_title', 'meta_description', 'alt_text', 
                           'image_title', 'image_caption', 'image_description', 'seo_tags', 'focus_keywords']:
            if content_type in result['content'] and result['content'][content_type]:
                content_text = QTextEdit()
                content_text.setReadOnly(True)
                
                if content_type == 'description':
                    content_text.setHtml(result['content'][content_type])
                else:
                    content_text.setText(result['content'][content_type])
                
                # Format the tab title
                title = content_type.replace('_', ' ').title()
                content_tabs.addTab(content_text, title)
        
        result_layout.addWidget(content_tabs)
        
        # Copy buttons
        buttons_layout = QHBoxLayout()
        
        # Add copy buttons for each content type
        copy_html_btn = QPushButton("Copy HTML Description")
        copy_html_btn.clicked.connect(lambda: self.copy_to_clipboard(result['content'].get('description', '')))
        buttons_layout.addWidget(copy_html_btn)
        
        copy_short_btn = QPushButton("Copy Short Description")
        copy_short_btn.clicked.connect(lambda: self.copy_to_clipboard(result['content'].get('short_description', '')))
        buttons_layout.addWidget(copy_short_btn)
        
        copy_all_btn = QPushButton("Copy All Content")
        copy_all_btn.clicked.connect(lambda: self.copy_all_content(result['content']))
        buttons_layout.addWidget(copy_all_btn)
        
        result_layout.addLayout(buttons_layout)
        
        result_widget.setLayout(result_layout)
        
        # Insert at the beginning of the layout
        self.results_layout.insertWidget(0, result_widget)
        
        # Switch to results tab
        tab_widget = self.centralWidget().layout().itemAt(0).widget()
        results_tab_index = [tab_widget.tabText(i) for i in range(tab_widget.count())].index("Results")
        tab_widget.setCurrentIndex(results_tab_index)
        
    def processing_complete(self):
        self.update_status("[SUCCESS] Processing completed!")
        
        # Play sound or show notification
        QMessageBox.information(self, "Processing Complete", "All products have been processed successfully!")
        
        # Reset progress
        self.progress_bar.setValue(100)
        
    def copy_to_clipboard(self, text):
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self.update_status("Content copied to clipboard!")
        
    def copy_all_content(self, content):
        all_content = ""
        for key, value in content.items():
            all_content += f"## {key.replace('_', ' ').title()}\n\n{value}\n\n"
            
        self.copy_to_clipboard(all_content)
        
    def export_results(self):
        if not self.results:
            self.update_status("No results to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "HTML Files (*.html);;Text Files (*.txt);;JSON Files (*.json)")
        if not file_path:
            return
            
        file_ext = file_path.split(".")[-1].lower()
        
        try:
            if file_ext == "html":
                self.export_results_html(file_path)
            elif file_ext == "txt":
                self.export_results_text(file_path)
            elif file_ext == "json":
                self.export_results_json(file_path)
            else:
                self.update_status(f"Unsupported file format: {file_ext}")
                return
                
            self.update_status(f"[SUCCESS] Results exported to {file_path}")
            
        except Exception as e:
            self.update_status(f"[ERROR] Error exporting results: {str(e)}")
            logger.error(f"Error exporting results: {str(e)}")
            
    def export_results_html(self, file_path):
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>GPT-4o SEO Content Generator Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .product { border: 1px solid #ccc; margin-bottom: 20px; padding: 15px; border-radius: 5px; }
                .product h2 { margin-top: 0; }
                .content-section { margin-bottom: 15px; }
                .content-section h3 { margin-bottom: 5px; }
                pre { background-color: #f5f5f5; padding: 10px; border-radius: 3px; white-space: pre-wrap; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>GPT-4o SEO Content Generator Results</h1>
            <p>Generated on: %s</p>
        """ % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        for result in self.results:
            html += f"""
            <div class="product">
                <h2>{result['name']} (ID: {result['product_id']})</h2>
                <p><strong>Status:</strong> {result['status']}</p>
            """
            
            for content_type, content_value in result['content'].items():
                if content_value:
                    title = content_type.replace('_', ' ').title()
                    
                    html += f"""
                    <div class="content-section">
                        <h3>{title}</h3>
                    """
                    
                    if content_type == 'description':
                        html += f"{content_value}"
                    else:
                        html += f"<pre>{content_value}</pre>"
                        
                    html += "</div>"
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html)
            
    def export_results_text(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"GPT-4o SEO Content Generator Results\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for result in self.results:
                f.write(f"===== {result['name']} (ID: {result['product_id']}) =====\n")
                f.write(f"Status: {result['status']}\n\n")
                
                for content_type, content_value in result['content'].items():
                    if content_value:
                        title = content_type.replace('_', ' ').title()
                        
                        # Strip HTML tags for the description
                        if content_type == 'description':
                            content_value = re.sub(r'<[^>]+>', '', content_value)
                            
                        f.write(f"----- {title} -----\n")
                        f.write(f"{content_value}\n\n")
                        
    def export_results_json(self, file_path):
        # Prepare data for JSON export
        export_data = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'products': []
        }
        
        for result in self.results:
            product_data = {
                'product_id': result['product_id'],
                'name': result['name'],
                'status': result['status'],
                'content': result['content']
            }
            export_data['products'].append(product_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            
    def clear_results(self):
        # Clear the results list
        self.results = []
        
        # Remove all result widgets from the layout
        while self.results_layout.count() > 1:  # Keep the stretch item
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        self.update_status("Results cleared")
        
    def clear_single_fields(self):
        self.single_name_input.clear()
        self.single_url_input.clear()
        self.single_id_input.clear()
        self.update_status("Single product fields cleared")
        
    def save_settings(self):
        settings = {
            'wc_api_url': self.wc_url_input.text(),
            'wc_consumer_key': self.wc_key_input.text(),
            'wc_consumer_secret': self.wc_secret_input.text(),
            'update_woocommerce': self.update_wc_checkbox.isChecked(),
            'openai_api_key': self.openai_key_input.text(),
            'request_delay': self.delay_spinbox.value(),
            'id_column': self.id_column_input.text(),
            'name_column': self.name_column_input.text(),
            'url_column': self.url_column_input.text()
        }
        
        try:
            # Create settings directory if it doesn't exist
            os.makedirs(os.path.expanduser("~/.seo_app"), exist_ok=True)
            
            # Save settings to JSON file
            with open(os.path.expanduser("~/.seo_app/settings.json"), 'w') as f:
                json.dump(settings, f, indent=2)
                
            self.settings_status.setText("[SUCCESS] Settings saved successfully")
            logger.info("Settings saved successfully")
            
        except Exception as e:
            self.settings_status.setText(f"[ERROR] Error saving settings: {str(e)}")
            logger.error(f"Error saving settings: {str(e)}")
            
    def load_settings(self):
        try:
            settings_path = os.path.expanduser("~/.seo_app/settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
        
        return {}
        
    def apply_saved_settings(self):
        if not self.settings:
            return
            
        # Apply WooCommerce settings
        self.wc_url_input.setText(self.settings.get('wc_api_url', ''))
        self.wc_key_input.setText(self.settings.get('wc_consumer_key', ''))
        self.wc_secret_input.setText(self.settings.get('wc_consumer_secret', ''))
        self.update_wc_checkbox.setChecked(self.settings.get('update_woocommerce', True))
        
        # Apply OpenAI settings
        self.openai_key_input.setText(self.settings.get('openai_api_key', ''))
        self.delay_spinbox.setValue(self.settings.get('request_delay', 2))
        
        # Apply Excel settings
        self.id_column_input.setText(self.settings.get('id_column', 'product id'))
        self.name_column_input.setText(self.settings.get('name_column', 'Perfume/Product Name'))
        self.url_column_input.setText(self.settings.get('url_column', 'Product URL'))
        
    def test_woocommerce_connection(self):
        wc_url = self.wc_url_input.text().strip()
        wc_key = self.wc_key_input.text()
        wc_secret = self.wc_secret_input.text()
        
        # Remove any spaces in URL
        wc_url = wc_url.replace(" ", "")
        
        # Ensure URL has proper format
        if wc_url and not wc_url.startswith(('http://', 'https://')):
            wc_url = 'https://' + wc_url
            
        # Ensure URL ends with slash
        if wc_url and not wc_url.endswith('/'):
            wc_url += '/'
        
        if not all([wc_url, wc_key, wc_secret]):
            self.settings_status.setText("[ERROR] Please fill in all WooCommerce API fields")
            return
            
        try:
            # Make a test request to the API (getting just one product)
            response = requests.get(
                f"{wc_url}?per_page=1",
                auth=(wc_key, wc_secret)
            )
            
            if response.status_code in [200, 201]:
                self.settings_status.setText("[SUCCESS] WooCommerce connection successful")
            else:
                self.settings_status.setText(f"[ERROR] WooCommerce connection failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.settings_status.setText(f"[ERROR] Error connecting to WooCommerce: {str(e)}")
            logger.error(f"Error connecting to WooCommerce: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look and feel
    
    # Set application icon
    app_icon = QIcon("icon.png")  # You can replace with your own icon
    app.setWindowIcon(app_icon)
    
    # Set application font
    app_font = QFont("Segoe UI", 9)  # Modern font
    app.setFont(app_font)
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())