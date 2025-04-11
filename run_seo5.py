import pandas as pd
import openai
import requests
import time
import re
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"seo_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FragranceSEOGenerator:
    """A class to generate SEO content for fragrance products using OpenAI's GPT-4o."""
    
    def __init__(self):
        # Configuration
        self.OPENAI_API_KEY = "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"
        self.WC_API_URL = "https://xsellpoint.com/wp-json/wc/v3/products/"
        self.WC_CONSUMER_KEY = "ck_ad9577c47151c4fa50ca6ee85dd2a58d2d6e6e79"
        self.WC_CONSUMER_SECRET = "cs_dc38dd138ae05842942ed84ff64a207a4b41b109"
        self.INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fragrance_products.xlsx")
        
        # Initialize OpenAI client
        openai.api_key = self.OPENAI_API_KEY
        self.client = openai.OpenAI(api_key=self.OPENAI_API_KEY)
        
        # Required output fields
        self.required_fields = [
            "description", "short_description", "seo_title", 
            "meta_description", "focus_keywords", "seo_tags"
        ]
        
        # Output sections to parse from GPT's response
        self.output_sections = [
            ("Long Description", "description"),
            ("Short Description", "short_description"),
            ("SEO Title", "seo_title"),
            ("Meta Description", "meta_description"),
            ("Focus Keywords", "focus_keywords"),
            ("SEO-Optimized Tags", "seo_tags")
        ]
    
    def build_prompt(self, name, url):
        """Build the prompt for GPT-4o based on the fragrance name and URL."""
        return f"""
You are an expert eCommerce SEO product description writer specializing in fragrance content optimization. Your goal is to create a detailed, SEO-optimized product description for a fragrance based on the provided product link, Product name and Competitor websites.

📌 Product Information:
✅ Fragrance Name: {name}
✅ Product Link: {url}
✅ Competitor websites: ([https://www.fragrantica.com], [https://klinq.com], [https://www.brandatt.com], [http://tatayab.com], [https://fragrancekw.com], [https://perfumeskuwait.com], [https://en.bloomingdales.com.kw])

✅ Requirements:
**1-Keyword Optimization**: Use Google Trends, SEMrush, Uber Suggest, and similar tools to extract the highest-ranking keywords for the fragrance and include them naturally in the content with bold font.

**2-Long Description (300+ words):**
Craft a compelling, informative, and SEO-rich product description.
-Focus Keyword should appear at the beginning of your content.
-Focus Keyword should appear in the content many times.
-Focus Keyword should found in subheading(s) like H2, H3, H4, etc...
The description should Include
-Table Information (Size, Gender, product type, Concentration and Brand name).
- Key Features
- History and about this perfume
- One Most common frequently searched question related to this fragrance or a similar one in its scent category, along with a detailed answer.
-THE DESCRIPTION SHOULD evoke an emotional connection, inspiring the customer to visualize themselves with suitable icons and emoticons.
-choose six words to have clickable links, 3 referable links refer to information about the targeted perfume from this website [https://www.wikiparfum.com/ar/] articles or other perfume information websites about the perfume name, other 3 referable links refer to other perfumes pages from this website to increase flow rate (https://Xsellpoint.com).
give the result of all the long description in html code to copy.
-Fragrance Notes Table: Clearly break down the scent pyramid (Top Notes, Heart Notes, Base Notes) in a structured table format.

**3-Short Description (max 50 words):**
A concise, engaging summary of the fragrance, emphasizing its uniqueness and main notes.

**4-SEO Elements (Rank Math SEO Plugin Optimized):**
**A-SEO Title:** A well-structured, keyword-rich title that aligns with Rank Math best practices.
-The title should CONTAIN Focus Keyword.
- The title should be less than 60 CHARACTERS for context.
-The title should have at least one power word.
-The title should have one positive or negative sentiment word.
- The title should have one number

**B-Meta Description** (max 155 characters): A compelling, keyword-focused snippet to improve search engine rankings and click-through rate (CTR). Also
 - should include call to action.
 -use active voice.
 -should CONTAIN Focus Keyword

**C-Focus Keywords**: Suggest primary and secondary focus keywords based on high search volume and relevance to Kuwait's and Gulf's fragrance market and competitor websites.

**5. Develop an SEO-optimized list of 6 tags: for the product,**
-Focusing on high-search-volume keywords.
-Then put together SEPARATED BY COMMAS.
-The tags in found in the descriptions make it in bold fonts

Please write in English language.

IMPORTANT: Please structure your response with clear headings for each section so I can easily parse them:
- "## Long Description"
- "## Short Description"
- "## SEO Title"
- "## Meta Description"
- "## Focus Keywords"
- "## SEO-Optimized Tags"
"""

    def parse_gpt_output(self, text):
        """Parse GPT-4o's response into structured fields."""
        fields = {}
        
        # Try to find each section using regex for more robust extraction
        for section_name, field_key in self.output_sections:
            pattern = rf"##\s*{section_name}.*?(?=##|$)"
            matches = re.findall(pattern, text, re.DOTALL)
            
            if matches:
                # Extract content after the heading
                content = matches[0].strip()
                # Remove the heading itself
                content = re.sub(rf"##\s*{section_name}.*?\n", "", content, flags=re.DOTALL).strip()
                
                # Special handling for HTML content in description section
                if field_key == "description" and "```html" in content:
                    html_match = re.search(r"```html\s*(.*?)\s*```", content, re.DOTALL)
                    if html_match:
                        content = html_match.group(1).strip()
                
                fields[field_key] = content
            else:
                logger.warning(f"Section '{section_name}' not found in GPT's response")
                fields[field_key] = ""
        
        # Log found and missing fields
        logger.info(f"Found fields: {list(fields.keys())}")
        missing_fields = [field for field in self.required_fields if not fields.get(field)]
        if missing_fields:
            logger.warning(f"Missing fields: {missing_fields}")
        
        return fields

    def update_woocommerce_product(self, product_id, content):
        """Update a WooCommerce product with the generated SEO content."""
        try:
            # Process tags - split by commas and clean up
            tags_raw = content.get("seo_tags", "")
            if isinstance(tags_raw, str):
                # Remove any markdown formatting and split by commas
                tags_clean = re.sub(r'\*\*|\*', '', tags_raw)
                tags_list = [tag.strip() for tag in tags_clean.split(',') if tag.strip()]
            else:
                tags_list = []
            
            data = {
                "description": content.get("description", ""),
                "short_description": content.get("short_description", ""),
                "tags": [{"name": tag} for tag in tags_list],
                "meta_data": [
                    {"key": "rank_math_title", "value": content.get("seo_title", "")},
                    {"key": "rank_math_description", "value": content.get("meta_description", "")},
                    {"key": "rank_math_focus_keyword", "value": content.get("focus_keywords", "")},
                ]
            }

            response = requests.put(
                self.WC_API_URL + str(product_id),
                auth=(self.WC_CONSUMER_KEY, self.WC_CONSUMER_SECRET),
                json=data
            )
            
            if response.status_code not in [200, 201]:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                
            return response.status_code, response.json()
        except Exception as e:
            logger.error(f"Error updating product: {e}")
            return 500, {"error": str(e)}

    def process_products(self):
        """Process all products in the Excel file."""
        try:
            # Load Excel file
            df = pd.read_excel(self.INPUT_FILE)
            logger.info(f"Loaded {len(df)} products from {self.INPUT_FILE}")
            
            # Process each product
            for index, row in df.iterrows():
                try:
                    pid = row['product id']
                    name = row['Perfume/Product Name']
                    url = row['Product URL']
                    
                    logger.info(f"Processing {index+1}/{len(df)}: {name} (ID: {pid})")
                    
                    # Generate content with GPT-4o
                    prompt = self.build_prompt(name, url)
                    
                    # Call OpenAI API
                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an expert eCommerce SEO content writer specializing in fragrance products."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=4000,
                        temperature=0.7
                    )
                    
                    result = response.choices[0].message.content
                    
                    # Save raw response for reference
                    output_dir = "gpt_responses"
                    os.makedirs(output_dir, exist_ok=True)
                    with open(f"{output_dir}/{pid}_{name[:20].replace(' ', '_')}.txt", "w", encoding="utf-8") as f:
                        f.write(result)
                    
                    # Parse GPT's response
                    content = self.parse_gpt_output(result)
                    
                    # Check for required fields
                    missing_fields = [field for field in self.required_fields if not content.get(field)]
                    if missing_fields:
                        logger.warning(f"Missing required fields for {name}: {', '.join(missing_fields)}")
                        continue
                    
                    # Update WooCommerce
                    status, resp = self.update_woocommerce_product(pid, content)
                    
                    if status in [200, 201]:
                        logger.info(f"✅ Updated product {pid} - {name}")
                    else:
                        logger.error(f"❌ Failed to update product {pid}: {resp}")
                    
                    # Wait between requests to avoid rate limits
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error processing product {row.get('Perfume/Product Name', 'Unknown')}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    try:
        # Create and run the SEO generator
        seo_generator = FragranceSEOGenerator()
        seo_generator.process_products()
        logger.info("Processing complete!")
    except Exception as e:
        logger.critical(f"Script execution failed: {e}")
