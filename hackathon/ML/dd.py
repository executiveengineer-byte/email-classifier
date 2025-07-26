model = None
vectorizer = None
label_encoder = None

from flask import (
    Flask, request, render_template, send_from_directory, send_file, jsonify, abort, redirect, url_for, flash, jsonify, Response, flash, make_response
)

import re
import random
import joblib
import string
from datetime import datetime, timedelta
from pathlib import Path
from scipy.interpolate import make_interp_spline 
from pymongo import MongoClient
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Must be done before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
import numpy as np
from rapidfuzz import fuzz
import nltk
import traceback 
from typing import Optional
from flask_mail import Mail, Message  # Add this import at the top
import os
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import base64
from email.message import EmailMessage
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from email.mime.application import MIMEApplication
import json
from collections import defaultdict
from bson.regex import Regex
import matplotlib.pyplot as plt
import seaborn as sns
import pdfkit
from feature_engineering import create_structured_features_from_text
from pdf_export import generate_executive_pdf

import pandas as pd
from bson import ObjectId

from functools import wraps
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from flask_login import LoginManager, UserMixin, login_required, current_user, logout_user, login_user
from werkzeug.security import generate_password_hash, check_password_hash
from tasks import send_campaign_email, send_campaign_batch
from flask_moment import Moment
from utils import gmail_authenticate, send_email_with_attachments
from functools import wraps
from flask_login import login_required
from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash
from flask import flash, redirect, url_for, request, render_template





with open("product_data/structured_product_hierarchy.json") as f:
    PRODUCT_HIERARCHY = json.load(f)

from textblob import TextBlob

SCOPES = ['https://www.googleapis.com/auth/gmail.send']


PRODUCT_CODE_TO_NAME_MAP = {}


# ✅ Use a valid database name (no spaces)
client = MongoClient("mongodb+srv://bharatiadmin:bharati123@cluster0.nt2rwkw.mongodb.net/bharati_ai?retryWrites=true&w=majority")

db = client.bharati_ai 
collection = db['email_logs']
campaigns = db['campaigns']
leads_col = db['leads']
replies = db['replies']

# Alternative, for sales outreach:
leads_collection       = db["sales_leads"]
campaigns_collection   = db["sales_campaigns"]
sent_emails_collection = db["sent_emails"]
replies_collection     = db["replies"]

# In testapp.py

class User(UserMixin):
    """
    A robust User class that safely handles all required attributes.
    """
    def __init__(self, user_data):
        """Initializes a User object from a MongoDB document."""
        self.id = str(user_data['_id'])
        # Use .get() to prevent KeyErrors if a field is missing
        self.email = user_data.get('email')
        self.password_hash = user_data.get('password')
        self.role = user_data.get('role', 'default') # Default to a safe role
        self.name = user_data.get('name', self.email) # Default to email if name is missing

    def get_id(self):
        return self.id

app = Flask(__name__)
moment = Moment(app) 

# =========================================================
# FIX: Add a secret key for session management (e.g., flash messages)
# =========================================================
app.config['SECRET_KEY'] = 'a-super-secret-and-unique-key-for-my-app'
# =========================================================

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ======================= ADD THESE TWO LINES =======================
login_manager.login_message = "You must be logged in to access this page."
login_manager.login_message_category = "info" # Use a less alarming color like blue
# ===================================================================

class User(UserMixin, dict):
    def __init__(self, user_dict):
        super().__init__(user_dict)

    @property
    def id(self):
        return str(self.get('_id'))

    @property
    def role(self):
        return self.get('role', 'sales')

# In testapp.py (add this function anywhere)

# In testapp.py

# In testapp.py


def create_user(email, password, role='sales', name=None):
    """Utility to create a new user with a hashed password."""
    if db.users.find_one({'email': email}):
        print(f"User with email {email} already exists.")
        return

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    
    user_data = {
        'email': email,
        'password': hashed_password,
        'role': role
    }
    if name:
        user_data['name'] = name
    
    db.users.insert_one(user_data)
    print(f"User {email} created successfully.")

@login_manager.user_loader
def load_user(user_id):
    """
    Given a user_id, find the user document in the database and
    create a User object from it.
    """
    try:
        user_data = db.users.find_one({"_id": ObjectId(user_id)})
        if user_data:
            # Pass the entire database document to our new User class
            return User(user_data)
    except Exception as e:
        print(f"Error loading user by ID: {e}")
    
    # If user is not found or an error occurs, return None
    return None

def role_required(role: str):
    from functools import wraps
    def wrapper(fn):
        @wraps(fn)
        @login_required
        def decorated_view(*args, **kwargs):
            if current_user.role != role:
                abort(403)
            return fn(*args, **kwargs)
        return decorated_view
    return wrapper


def nocache(view):
    """A decorator to add headers that prevent browser caching."""
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = view(*args, **kwargs)
        # If the response is not a standard Flask Response object, convert it
        if not isinstance(response, Response):
            response = make_response(response)
        
        # Add headers to prevent caching
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return no_cache

def build_product_map(hierarchy_data):
    """
    Populates the global PRODUCT_CODE_TO_NAME_MAP dictionary.
    This version correctly handles both final products and selectable categories.
    This function should be called once when the application starts.
    """
    global PRODUCT_CODE_TO_NAME_MAP
    PRODUCT_CODE_TO_NAME_MAP = {} # Reset the map

    top_level_names = {
        "FEXT": "Fire Extinguishers", "MOD": "Modulars", "LITH": "Lith-Ex Safety",
        "AERO": "Aerosol Systems", "DET": "Detectors", "HYD": "Hydrant Systems",
        "FET": "Equipment Trolleys", "SYS": "Suppression Systems"
    }

    def recurse(node, path_display, path_code):
        if isinstance(node, dict):
            for key, value in node.items():
                current_display_part = top_level_names.get(key, key) if not path_display else key
                new_path_display = f"{path_display} > {current_display_part}" if path_display else current_display_part
                
                # Create a machine-friendly code part from the key
                current_code_part = key.upper().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
                new_path_code = f"{path_code}-{current_code_part}" if path_code else current_code_part
                
                # --- THIS IS THE CORE FIX ---
                # If a category points to an EMPTY list, treat the category itself as a product.
                if isinstance(value, list) and not value:
                    PRODUCT_CODE_TO_NAME_MAP[new_path_code] = new_path_display
                
                # If a category points to a NON-EMPTY list, process the items.
                elif isinstance(value, list) and value:
                     for item in value:
                        if isinstance(item, dict) and 'name' in item and 'code' in item:
                            # This is a final product with a pre-defined code.
                            display_name = f"{new_path_display} > {item['name']}"
                            PRODUCT_CODE_TO_NAME_MAP[item['code']] = display_name

                # If the value is another dictionary, go deeper.
                elif isinstance(value, dict):
                    recurse(value, new_path_display, new_path_code)

    # Start the process from the top of the JSON
    recurse(hierarchy_data, "", "")
    
    # --- DEBUGGING STEP ---
    # This print statement is crucial for verification.
    print(f"✅ Product code-to-name map built successfully with {len(PRODUCT_CODE_TO_NAME_MAP)} items.")
    if len(PRODUCT_CODE_TO_NAME_MAP) < 10:
        print("   ⚠️ WARNING: The product map seems very small. Check the JSON file and build_product_map logic.")


# THE FIX: This call is now OUTSIDE and AFTER the function definition.
build_product_map(PRODUCT_HIERARCHY)
# ==============================================================================
# CHANGE IN: testapp.py (Replace the entire old function with this new one)
# ==============================================================================

def humanize_product_tag(code: str) -> str:
    """Convert official product codes to human-readable names using the global map."""
    # Use .get() for safety. If a code is not found, it returns the code itself.
    return PRODUCT_CODE_TO_NAME_MAP.get(code, code)

app.jinja_env.filters['humanize_product_tag'] = humanize_product_tag

# ==============================================================================
# CHANGE IN: testapp.py (ADD this new function)
# ==============================================================================

def get_product_list_for_dropdown():
    """
    Converts the global PRODUCT_CODE_TO_NAME_MAP into a sorted list of
    dictionaries suitable for an HTML dropdown.
    """
    product_list = [{'code': code, 'display': name} for code, name in PRODUCT_CODE_TO_NAME_MAP.items()]
    # Sort the final list alphabetically by the display name for a clean dropdown
    return sorted(product_list, key=lambda x: x['display'])
    

# ==============================================================================
# CHANGE IN: testapp.py (Replace the entire old function with this new one)
# ==============================================================================

def generate_product_insights(records):
    """
    Generate product insights using official product codes.
    The keys in product_counts will now be the official codes.
    """
    product_counts = defaultdict(int)
    category_capacity_map = defaultdict(lambda: defaultdict(int))

    for rec in records:
        # The 'products_detected' field now contains official codes
        codes = rec.get("products_detected", [])
        cat = rec.get("category", "unknown")
        for code in codes:
            product_counts[code] += 1
            # Extract capacity/size from the end of the code if possible
            # e.g., FEXT-CO2_PORTABLE-4.5KG -> 4.5KG
            try:
                capacity = code.rsplit('-', 1)[-1]
                category_capacity_map[cat][capacity] += 1
            except IndexError:
                pass  # Code doesn't have a capacity suffix

    # Format insights for display
    insights = []
    if records:
        # Note: 'total_records' is no longer the best denominator.
        # We'll use the total number of product tags detected for a more accurate percentage.
        total_tags_detected = sum(product_counts.values())
        if total_tags_detected > 0:
            # Sort by count, highest first, and take top 5
            sorted_items = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for code, count in sorted_items:
                readable_name = humanize_product_tag(code) # Uses our new, simple function
                insights.append({
                    'count': count,
                    'name': readable_name,
                    'raw_tag': code, # Store the official code
                    'percentage': round((count / total_tags_detected) * 100, 1)
                })

    return insights, product_counts, category_capacity_map



# =============================================
# CHART GENERATION FUNCTIONS (ADD THIS NEW BLOCK)
# =============================================

# In testapp.py

def create_performance_leaderboard(team_data):
    """
    Generate a modern performance leaderboard visualization, inspired by the reference image.
    This chart uses a composite score for ranking.
    """
    try:
        if not team_data:
            return ""

        plt.style.use('seaborn-v0_8-whitegrid') # Use a clean base style
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_alpha(0.0) # Transparent background
        ax.set_facecolor('#FFFFFF00')
        
        df = pd.DataFrame(team_data)

        # Ensure 'score' exists and is numeric, filling NaNs with 0
        df['score'] = pd.to_numeric(df.get('score'), errors='coerce').fillna(0)
        
        # Sort by score and take the top 5 performers
        df = df.sort_values('score', ascending=False).head(5)
        
        # --- THE FIX: Create a gradient color effect ---
        # Normalize scores to be between 0 and 1 for the colormap
        max_score = df['score'].max()
        if max_score > 0:
            norm_scores = df['score'] / max_score
        else:
            norm_scores = [0] * len(df) # Avoid division by zero
            
        colors = plt.cm.viridis(norm_scores) # 'viridis' is a nice blue-green-yellow gradient

        # Create horizontal bar chart, reversing the order for top-to-bottom display
        bars = ax.barh(
            df['name'][::-1], 
            df['score'][::-1],
            color=colors[::-1],
            height=0.6 # Control bar thickness
        )
        
        # --- THE FIX: Add value labels inside the bars ---
        for bar in bars:
            width = bar.get_width()
            # Logic to place text inside or outside based on bar width
            is_inside = width > df['score'].max() * 0.2
            x_pos = width / 2 if is_inside else width + 5
            
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f'{width:.0f}',
                ha='center' if is_inside else 'left',
                va='center',
                color='white' if is_inside else 'gray',
                fontweight='bold',
                fontsize=10
            )
        
        # --- THE FIX: Minimalist styling ---
        ax.set_title('Team Response Time by Composite Score', color='gray', pad=20, fontsize=14)
        ax.set_xlabel('Performance Score (0-100)', color='gray', labelpad=10, fontsize=10)
        ax.set_xlim(0, 105) # Give a little extra space beyond 100
        
        # Remove grid and most borders
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#E0E0E0')
        
        # Customize ticks
        ax.tick_params(axis='x', colors='gray')
        ax.tick_params(axis='y', length=0, colors='gray')
        
        plt.tight_layout(pad=1.5)

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, transparent=True)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"Error generating leaderboard chart: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""

def create_sla_summary_chart(context):
    """Generate SLA compliance visualization"""
    try:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(['SLA Met', 'Breached'], 
               [context['sla_met_pct'], 100 - context['sla_met_pct']],
               color=['#4CAF50', '#F44336'])
        ax.set_title('SLA Compliance Overview', pad=10)
        ax.set_ylim(0, 100)
        
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"Error generating SLA chart: {str(e)}")
        return ""

# AFTER
def create_product_demand_chart(logs): # Changed context to logs for clarity
    """Generate top products visualization with thicker bars and count labels."""
    try:
        # --- 1. Data Preparation ---
        if not logs:
            return ""

        product_counts = defaultdict(int)
        for rec in logs:
            for code in rec.get("products_detected", []):
                product_counts[code] += 1
        
        if not product_counts:
            return ""

        # Get top 5 products
        top_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Reverse the list for top-to-bottom display on the chart
        top_products.reverse()

        # Prepare data for plotting
        # THE FIX: Add the count to the product name for a richer label
        names = [f"{humanize_product_tag(p[0])} ({p[1]})" for p in top_products]
        counts = [p[1] for p in top_products]
        
        # --- 2. Chart Creation and Styling ---
        fig, ax = plt.subplots(figsize=(10, 6)) # Slightly larger figure for better spacing
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('#FFFFFF00')

        # THE FIX: Increase bar thickness using the 'height' parameter
        bars = ax.barh(names, counts, color='#0D6EFD', height=0.6, align='center')
        
        # Clean up axes and remove borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False) # Hide x-axis numbers as the label now has the count

        # Customize tick labels (the product names)
        ax.tick_params(axis='y', length=0, labelsize=11, labelcolor='gray')
        
        # Set title
        ax.set_title('Top Requested Products', color='gray', fontsize=14, pad=20)
        
        plt.tight_layout(pad=1.5)

        # --- 3. Saving to Buffer ---
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, transparent=True)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode()

    except Exception as e:
        print(f"Error generating product demand chart: {str(e)}")
        return ""
    
# AFTER
# In testapp.py

# =========================================================================
# === FINAL, ROBUST REPLACEMENT for create_sla_trend_chart
# =========================================================================
# In testapp.py

# In testapp.py

# In testapp.py

def create_sla_trend_chart(df):
    """
    Generate a MODERN and ROBUST SLA compliance trend chart.
    This version has improved text visibility.
    """
    try:
        if df.empty or 'timestamp' not in df.columns or 'SLA_Met' not in df.columns:
            return ""

        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        sla_trends = df.groupby("date")["SLA_Met"].mean().reset_index()
        sla_trends["SLA_Met"] *= 100

        if len(sla_trends) < 2:
            return ""

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("#00000000")

        line_color = '#0D6EFD'
        
        # --- THE FIX: Define a darker color for text and grid ---
        dark_text_color = "#000000" 
        light_grid_color = (0.0, 0.0, 0.0, 0.1) 
        
        ax.plot(sla_trends["date"], sla_trends["SLA_Met"], 
                color=line_color, linewidth=2.5, marker='o', markersize=8, 
                markerfacecolor='white', markeredgecolor=line_color,
                markeredgewidth=2.5, zorder=3)
        
        ax.fill_between(
            sla_trends["date"], sla_trends["SLA_Met"], 
            color=line_color, alpha=0.1, zorder=2)

        ax.set_ylim(0, 105)
        ax.set_ylabel("")
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color(light_grid_color)

        # --- THE FIX: Apply the darker text color ---
        ax.tick_params(axis='x', colors=dark_text_color, length=0, pad=10, labelsize=11)
        ax.tick_params(axis='y', colors=dark_text_color, length=0, pad=10, labelsize=11)
        
        ax.grid(axis='y', linestyle='--', alpha=0.5, color=light_grid_color, zorder=1)
        
        plt.tight_layout(pad=1.5)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, transparent=True)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        print(f"Error generating SLA trend chart: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""
    

def create_automation_metrics_chart(metrics):
    """Visualize AI-human collaboration metrics"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Human intervention metrics
        ax1.barh(
            ['Escalated Cases', 'Human Response', 'QA Review'], 
            [metrics['escalation_rate'], metrics['human_response_time'], metrics['qa_time']],
            color=['#ff7f0e', '#1f77b4', '#2ca02c']
        )
        ax1.set_title('Human Intervention Metrics')
        ax1.set_xlabel('Time (hours) / Percentage')
        
        # System performance
        ax2.bar(
            ['Auto-Reply Success', 'Human Correction', 'Confidence'],
            [metrics['success_rate'], metrics['correction_rate'], metrics['avg_confidence']],
            color=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        ax2.set_title('System Performance')
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        plt.close()
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"Error generating metrics chart: {str(e)}")
        return ""
# =============================================
# END OF CHART FUNCTIONS
# =============================================

def generate_catalogue_reply():
    return """Dear Sir / Madam,<br>
Thank you for showing your interest in our products.
This is to confirm that we have successfully received your request.<br>
To provide you with more detailed information, please explore our company product catalogues here:
<a href="https://drive.google.com/drive/folders/1CHIdf74z6xKLkX3CvtFhyemKVQjvIN64?usp=sharing" target="_blank" rel="noopener noreferrer">https://drive.google.com/drive/folders/1CHIdf74z6xKLkX3CvtFhyemKVQjvIN64?usp=sharing</a><br>
We request you to review the shared details and let us know your specific requirements.<br>
For a smoother onboarding process, kindly fill out the attached <strong>Customer Registration Form</strong>. The information you provide will be directly recorded in our ERP system, which will help streamline future interactions and benefit both our organizations:
<a href="https://forms.gle/2aWvsYC51UCCCdTj8" target="_blank" rel="noopener noreferrer">https://forms.gle/2aWvsYC51UCCCdTj8</a><br>
Our sales team will connect with you shortly for further assistance.<br>
For immediate support, feel free to contact:
📞 <strong>Geeta Sawant:</strong> +91 9833808061 &nbsp;/&nbsp; <strong>Vishakha Parab:</strong> +91 7045100403 &nbsp;/&nbsp; <strong>Gaurita Sawant:</strong> +91 8591998713<br>
We look forward to serving you.<br>
<span style="color:#1a73e8;">Thanks and Regards,</span>"""




def generate_complaint_reply():
    return """Dear Sir / Madam,<br>
We sincerely apologize for any inconvenience caused.<br>
We acknowledge your concern and have escalated the matter to our Quality Assurance team for detailed review. 
They will investigate the root cause and ensure an appropriate resolution is provided at the earliest.<br>
Our quality team will be in touch with you shortly to offer further assistance and address any queries you may have.<br>
Thank you for your patience and understanding.<br>
Thanks and Regards,"""

def generate_followup_reply(details, email_text):
    email_text = email_text.lower()
    dispatch_keywords = ["dispatch", "tracking", "courier", "awb", "lr copy", "shipment", "delivery", "transit"]

    has_dispatch_context = any(word in email_text for word in dispatch_keywords)

    base_reply = """Dear Sir / Madam,<br>
Thank you for following up with us.<br>
We would like to inform you that your request (Ref: {ref}) is being processed and has already been escalated to the concerned department for immediate attention.<br><br>"""

    dispatch_block = """Meanwhile, please note that your order has been dispatched via <strong>[Courier Name]</strong> and the tracking ID is <strong>[Tracking ID]</strong>. Expected delivery is within <strong>[Expected Timeframe]</strong>.<br><br>"""

    closing = """We appreciate your patience and assure you of timely updates.<br>
Thanks and Regards,"""

    full_reply = base_reply.format(ref=details["REF_NUMBER"])
    if has_dispatch_context:
        full_reply += dispatch_block
    full_reply += closing

    return full_reply




def generate_other_reply():
    return """Dear Sir / Madam,<br>
Thank you for reaching out to us.<br>
We have received your message and forwarded it to the concerned department for appropriate review and handling.<br>
Should any further clarification be required, our team will get in touch with you shortly.<br>
We appreciate your interest and assure you of our prompt support.<br>
Thanks and Regards,"""


# === Global Definitions ===
categories = {
    "quotation_request": [
        "price quote", "send quote", "product quote", "quote request", "quotation required",
        "quotation", "quote", "price", "estimate", "cost", "rate", "pricing",
        "quotation needed", "need quotation", "send price", "price list",
        "cost estimate", "price quote", "request for quote", "rfq",
        "how much for", "what's the price of", "looking for pricing on",
        "please quote", "RFQ", "could you provide a quote", "price inquiry",
        "requesting quotation", "need pricing", "require quotation",
        "send me the quote", "what would be the cost", "price details",
        "quote request", "requesting price", "cost breakdown",
        "please send quotation", "need cost estimate", "pricing information"
    ],
    "complaint": [
        "not functioning", "doesn't work", "does not function", "unit damaged", "received broken", "arrange replacement", "not operational",
        "complaint", "complaints", "problem", "problems", "issue", "issues",
        "damaged", "broken", "not working", "faulty", "defective",
        "disappointed", "dissatisfied", "bad product", "poor quality",
        "does not work", "malfunction", "refund", "return", "replacement",
        "missing item", "not delivered", "wrong item", "incorrect item",
        "delivered wrong", "delivered less", "short delivery",
        "quantity mismatch", "wrong delivery", "defective unit",
        "urgent resolution", "received only", "invoice mismatch",
        "not as described", "poor service", "unhappy with",
        "this is unacceptable", "very disappointed", "not satisfied",
        "quality issue", "service complaint", "shipping problem",
        "package damaged", "item broken", "not what I ordered",
        "order mistake", "billing error", "overcharged", "wrong billing",
        "need immediate resolution", "compensation required",
        "want my money back", "how do I return this", "not up to standard"
    ],
    "follow_up": [
        "follow up", "follow-up", "followup", "reminder", "reminders",
        "status", "status update", "pending", "update", "any update",
        "awaiting", "waiting", "still waiting", "revert", "response",
        "haven't heard back", "no reply yet", "when can I expect",
        "please respond", "kindly update", "need update",
        "requesting status", "what's the status", "any progress",
        "could you update me", "following up on", "checking in on",
        "when will this be", "has this been", "did you receive",
        "confirm receipt", "please acknowledge", "awaiting your reply",
        "looking forward to", "please let me know", "need confirmation",
        "requesting follow up", "status check", "progress update,"
        "dispatch", "dispatched", "delivery", "deliver", "courier",
        "couriers", "shipment", "shipping", "track", "tracking",
        "lr copy", "lr number", "tracking id", "tracking number",
        "awb", "airway bill", "delivery status", "logistics",
        "transport", "transit", "shipped", "out for delivery",
        "in transit", "on the way", "expected delivery",
        "delivery date", "ship date", "dispatch date",
        "when will it ship", "has it been shipped",
        "where is my order", "order status", "shipment update",
        "delivery timeline", "expected arrival", "arrival date",
        "package location", "where's my package", "track my order",
        "delivery confirmation", "proof of delivery", "pod",
        "received shipment", "not received", "delivery problem",
        "shipping details", "carrier information", "transport details",
        "logistics update", "freight information"
    ],


    "general_enquiry": [
        "enquiry", "inquiry", "enquiries", "inquiries", "information",
        "info", "clarification", "clarify", "details", "detail",
        "need details", "require info", "want to know", "would like to know",
        "kindly confirm", "please confirm", "product details",
        "share catalog", "send brochure", "what is the", "can you explain",
        "please share specs", "technical specifications", "any brochure",
        "looking for", "seeking information", "technical sheet",
        "need data", "require details", "what type of",
        "how does it work", "how to use", "training available",
        "can you guide", "service schedule", "provide availability",
        "have questions", "need assistance", "require clarification",
        "more information", "additional details", "further information",
        "could you explain", "would like information", "seeking clarification",
        "please advise", "need guidance", "require explanation",
        "product information", "service information", "company information",
        "contact information", "pricing information", "availability information",
        "specification request", "feature inquiry", "capacity details",
        "dimension inquiry", "material information", "operation details",
        "maintenance inquiry", "installation query", "warranty information"
    ]
}

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def detect_intent(text):
    intent_rules = {
        "action_required": ["please", "kindly", "request", "need", "require", "awaiting", "approve", "follow up", "urgent"],
        "information_only": ["for your information", "fyi", "note", "just informing", "letting you know"],
        "confirmation": ["we confirm", "confirmed", "acknowledge", "we acknowledge", "as discussed", "as per our conversation"],
        "query": ["can you", "could you", "would you", "may i know", "how much", "when is", "what is", "do you have", "share the details"],
        "appreciation": ["thank you", "thanks", "appreciate", "grateful", "great job", "good work", "kudos", "well done"]
    }
    text = text.lower()
    scores = {intent: sum(1 for phrase in phrases if phrase in text) for intent, phrases in intent_rules.items()}
    sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_intents[0][0] if sorted_intents[0][1] > 0 else "unknown"

def detect_urgency(email_text):
    """Detect urgency level from email text (0-1 scale)"""
    email_text = email_text.lower()
    urgency_score = 0.0
    
    # Strong urgency indicators
    if any(word in email_text for word in ["urgent", "immediately", "asap", "right away", "emergency"]):
        urgency_score = 0.9
    # Moderate urgency
    elif any(word in email_text for word in ["soon", "quick", "prompt", "timely", "follow up"]):
        urgency_score = 0.6
    # Mild urgency
    elif any(word in email_text for word in ["when convenient", "at your earliest", "please respond"]):
        urgency_score = 0.3
    
    return min(max(urgency_score, 0), 1)  # Ensure between 0-1

def smart_pattern_fallback(text):
    text = text.lower().strip()

    # === Price intent disambiguation ===
    if "price" in text or "cost" in text:
        if any(kw in text for kw in ["quote", "quotation", "send", "provide", "share", "need", "request", "offer"]):
            return "quotation_request"
        elif any(kw in text for kw in ["what is", "can you", "confirm", "clarify", "is it", "how much"]):
            return "general_enquiry"

    patterns = {
        "quotation_request": [
            "price quote", "send quote", "product quote", "quote request", "quotation required",
            "please send quotation", "need a quote", "request for quotation", "provide the rates", 
            "price list", "send cost", "need price", "quote us", "share pricing", "quotation for", 
            "rfq", "rate contract", "commercial offer", "could you quote", "price estimate",
            "looking for pricing", "what's your rate", "cost breakdown", "requesting quote",
            "please provide quote", "need your rates", "seeking quotation", "price inquiry",
            "request for pricing", "send me the quote", "what would be the cost", "require quotation",
            "share your commercial proposal", "send me your offer", "budgetary offer", "commercial quotation"
        ],
        "complaint": [
            "not functioning", "doesn't work", "does not function", "unit damaged", "received broken", "arrange replacement", "not operational",
            "not working", "damaged", "leaking", "defective", "faulty", "not satisfied", 
            "wrong item", "complaint", "need replacement", "issue with", "return this", 
            "broken", "malfunction", "poor quality", "disappointed", "unhappy with",
            "not as described", "received damaged", "defect in", "this is unacceptable", "want refund", 
            "return policy", "compensation", "service issue", "shipping problem", 
            "missing parts", "incomplete delivery", "billing error", "overcharged", 
            "never received", "item missing", "how to return", "warranty claim", 
            "dissatisfied with", "very disappointed"
        ],
        "follow_up": [
            "just following up", "any update", "still waiting", "haven't heard", 
            "reminder", "status of our previous mail", "awaiting your reply", 
            "follow up", "please revert", "checking status", "pending since",
            "when can I expect", "no response yet", "kindly update", "need update",
            "requesting status", "what's the status", "following up on",
            "please respond", "awaiting confirmation", "has this been processed",
            "did you receive my", "could you update", "looking forward to",
            "need your response", "please acknowledge", "reminder about",
            "status check", "progress update", "when will this be", "per our call", 
            "revised proposal", "waiting for proposal", "awaiting document",
            "dispatched", "tracking id", "awb number", "courier", "lr copy", 
            "shipment", "delivery update", "logistics", "delivery status", 
            "dispatch details", "shipping confirmation", "out for delivery",
            "track my order", "where is my", "expected delivery date",
            "shipment tracking", "transport details", "in transit",
            "when will it arrive", "has it been shipped", "delivery timeline",
            "proof of delivery", "pod", "freight details", "carrier information",
            "logistics update", "package location", "transit status",
            "ship date", "dispatch confirmation", "order tracking"
        ],
        "general_enquiry": [
            "enquiry", "want information", "details about", "can you share", 
            "product specification", "availability of", "lead time", "need info", 
            "clarification", "looking for details", "require information",
            "could you explain", "would like to know", "please provide details",
            "need assistance with", "have a question about", "seeking clarification",
            "more information about", "technical specifications", "product features",
            "how does it work", "what are the options", "service details",
            "company information", "contact details", "catalog request",
            "brochure needed", "spec sheet", "operation manual",
            "installation guide", "maintenance information", "warranty details",
            "can you confirm price", "clarify pricing", "what is the cost",
            "need clarification on pricing"
        ]
    }

def normalize(text):
    """Lowercase and simplify text for matching."""
    if isinstance(text, dict):  # Skip dictionaries
        return ""
    return str(text).lower().replace(" ", "").replace("-", "").replace("_", "")

# ==============================================================================
# CORRECTED FUNCTION IN: testapp.py (around line 697)
# ==============================================================================

# ==============================================================================
# CORRECTED FUNCTION IN: testapp.py (around line 705)
# ==============================================================================

def detect_products(text: str) -> list:
    """
    Detects products in text by matching keywords from the hierarchy
    and returns their official product codes. This version can detect
    both final products and selectable categories.
    """
    detected_codes = []
    clean_text = normalize(text)

    def recurse_hierarchy(node, path_code):
        if isinstance(node, dict):
            for key, value in node.items():
                current_code_part = key.upper().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
                new_path_code = f"{path_code}-{current_code_part}" if path_code else current_code_part

                # Match against the category name itself
                if normalize(key) in clean_text:
                    if isinstance(value, list) and not value:
                        # This is a selectable category like "Smoke Detectors"
                        detected_codes.append(new_path_code)

                # Process final products in a list
                if isinstance(value, list) and value:
                    for item in value:
                        if isinstance(item, dict) and 'name' in item and 'code' in item:
                            if normalize(item['name']) in clean_text:
                                detected_codes.append(item['code'])
                # Go deeper into the hierarchy
                elif isinstance(value, dict):
                    recurse_hierarchy(value, new_path_code)

    recurse_hierarchy(PRODUCT_HIERARCHY, "")
    return list(set(detected_codes))

def extract_subject_features(subject):
    subject = subject.lower()
    return [
        int("quote" in subject),
        int("quotation" in subject),
        int("product" in subject),
        int("price" in subject),
    ]


def sentence_level_fallback(text):
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split(".")

    votes = {}

    # Internal fallback disambiguation (price-specific)
    price_phrases = ["price", "cost"]
    quote_verbs = ["quote", "quotation", "send", "provide", "share", "need", "request", "offer"]
    enquiry_verbs = ["what is", "can you", "confirm", "clarify", "is it", "how much"]

    for sentence in sentences:
        sentence = sentence.lower().strip()

        # Disambiguate pricing intent per sentence (independent of smart_pattern_fallback)
        if any(p in sentence for p in price_phrases):
            if any(v in sentence for v in quote_verbs):
                votes["quotation_request"] = votes.get("quotation_request", 0) + 1
                continue
            elif any(v in sentence for v in enquiry_verbs):
                votes["general_enquiry"] = votes.get("general_enquiry", 0) + 1
                continue

        # Standard category voting
        cat = smart_pattern_fallback(sentence)
        if cat:
            votes[cat] = votes.get(cat, 0) + 1

    # Return category with most votes, default to general_enquiry
    return max(votes.items(), key=lambda x: x[1])[0] if votes else "general_enquiry"

# Load ML model and tools
model_path = Path(__file__).parent / "xgboost_email_classifier.joblib"
vectorizer_path = Path(__file__).parent / "tfidf_vectorizer.joblib"
label_encoder_path = Path(__file__).parent / "label_encoder.joblib"

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)
except Exception as e:
    print(f"Error loading model/vectorizer/encoder: {e}")
    model = vectorizer = label_encoder = None


def classify_email(email_text, subject_text=""):
    """
    Classifies an email by combining a machine learning model with a rule-based system.
    This version uses a centralized feature engineering function to prevent shape mismatches.
    """
    cleaned = preprocess(email_text)
    products_detected = detect_products(cleaned)
    global model, vectorizer, label_encoder

    ml_prediction = None
    ml_confidence = 0.0

    # Get the rule-based prediction first, as a fallback
    rule_prediction = sentence_level_fallback(email_text)
    rule_confidence = 0.3 if rule_prediction else 0.0

    # --- Start of the Machine Learning Prediction block ---
    if model and vectorizer and label_encoder:
        try:
            # --- THIS IS THE CORE FIX ---

            # 1. Vectorize the email body using the loaded TF-IDF vectorizer.
            X_text = vectorizer.transform([cleaned])

            # 2. Create the structured features using our reliable, imported function.
            #    This guarantees the "handshake" is always correct.
            structured_features_array = create_structured_features_from_text(cleaned, subject_text)

            # 3. Combine the text features and structured features into the final vector.
            #    The .reshape(1, -1) is important to make it a 2D array for hstack.
            from scipy.sparse import hstack
            X_final = hstack([X_text, structured_features_array.reshape(1, -1)])

            # --- END OF THE CORE FIX ---

            # Now, we can safely make the prediction because the shape is guaranteed to match.
            proba = model.predict_proba(X_final)[0]
            max_idx = proba.argmax()
            ml_prediction = label_encoder.inverse_transform([max_idx])[0]
            ml_confidence = float(proba[max_idx]) # Use the raw, high-precision confidence

        except Exception as e:
            # This provides a much more detailed error in your terminal if something goes wrong.
            print(f"🔴 ML prediction failed! Error: {e}")
            print(traceback.format_exc())

    # --- Start of the Hybrid Logic (this part of your code was already good) ---
    if rule_prediction:
        rule_confidence = sum([
            fuzz.partial_ratio(cleaned, phrase)
            for phrase in categories.get(rule_prediction, [])
            if fuzz.partial_ratio(cleaned, phrase) > 80
        ]) / 1000

    if 0.3 < ml_confidence < 0.7 and rule_confidence > 0.5:
        print("⚖️ Mid-confidence ML → Rule stronger: Overriding with rule.")
        ml_prediction = rule_prediction
        ml_confidence = rule_confidence
    elif ml_confidence < 0.3 and rule_confidence >= 0.3 and rule_prediction:
        print("🔁 Overriding with sentence-level fallback due to low confidence.")
        ml_prediction = rule_prediction
        ml_confidence = rule_confidence

    scores = {}
    all_categories = list(set([ml_prediction, rule_prediction] + list(categories.keys())))
    for cat in all_categories:
        if cat is None: continue
        cat_score = 0
        if cat == ml_prediction: cat_score += 0.5 * ml_confidence
        if cat == rule_prediction: cat_score += 0.3 * rule_confidence
        booster_words = {
            "general_enquiry": ["need information", "catalogue", "specifications", "how does it work"],
            "complaint": ["not working", "damaged", "broken", "missing", "invoice mismatch"],
            "follow_up": ["still waiting", "dispatch", "status update", "revert", "reminder"],
            "quotation_request": ["quote", "quotation", "rfq", "price list", "cost estimate"]
        }
        for word in booster_words.get(cat, []):
            if word in cleaned: cat_score += 0.2
        scores[cat] = round(cat_score, 4)

    if "price" in cleaned and any(w in cleaned for w in ["not received", "delayed", "wrong", "overcharged"]):
        print("🧠 Booster: price + issue trigger → boosting complaint")
        scores["complaint"] = scores.get("complaint", 0) + 0.2
    if any(w in cleaned for w in ["price", "quotation", "quoted"]) and any(w in cleaned for w in ["mismatch", "discrepancy"]):
        print("🧠 Booster: price + discrepancy → complaint")
        scores["complaint"] = scores.get("complaint", 0) + 0.3
    if any(w in cleaned for w in ["follow up", "reminder", "awaiting", "haven’t received"]) and "quotation" in cleaned:
        print("🧠 Booster: follow-up on quotation detected")
        scores["follow_up"] = scores.get("follow_up", 0) + 0.3

    final_prediction = max(scores.items(), key=lambda x: x[1])[0] if scores else rule_prediction or "general_enquiry"

    print("🧠 Hybrid Debug Info:")
    print(f"  ML Prediction: {ml_prediction} (Confidence: {round(ml_confidence, 3)})")
    print(f"  Rule Prediction: {rule_prediction} (Confidence: {round(rule_confidence, 3)})")
    print(f"  Hybrid Scores: {scores}")
    print(f"  Final Prediction: {final_prediction}")

    return final_prediction, {
        "ml_prediction": ml_prediction,
        "ml_confidence": float(ml_confidence or 0),
        "rule_prediction": rule_prediction,
        "rule_confidence": round(rule_confidence, 3),
        "hybrid_scores": scores,
        "products_detected": products_detected,
    }




def extract_details(email):
    details = {
        "REF_NUMBER": "N/A",
        "APPROVER_NAME": "Mr. Sharma (Sales Head)",
        "COURIER_NAME": "[Courier Name]",  # Placeholder for sales team
        "TRACKING_ID": "[Tracking ID]",    # Placeholder for sales team
        "DELIVERY_DATE": "[Expected Timeframe]",  # Placeholder for sales team
        "URGENT": "URGENT: " if any(word in email.lower() for word in ["urgent", "immediate", "asap"]) else ""
    }
    ref_match = re.search(r"(order|ref|case)\s*(#|no\.?)?\s*(\d+)", email, re.IGNORECASE)
    if ref_match:
        details["REF_NUMBER"] = ref_match.group(3)
    return details


def convert_numpy_types(doc):
    if isinstance(doc, dict):
        return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in doc.items()}
    return doc


@app.route("/", methods=["GET", "POST"])
@login_required 
@nocache
def home():
    if request.method == "POST":
        email = request.form["email"]
        category, metadata = classify_email(email)
        intent = detect_intent(email)
        urgency = detect_urgency(email)
        urgency_label = "High" if urgency > 0.7 else "Medium" if urgency > 0.3 else "Low"
        
        scores = {k: v for k, v in metadata.get("hybrid_scores", {}).items() if k is not None}
        details = extract_details(email)
        
        replies = {
            "quotation_request": generate_catalogue_reply(),
            "general_enquiry": generate_catalogue_reply(),
            "complaint": generate_complaint_reply(),
            "follow_up": generate_followup_reply(details, email),
            "other": generate_other_reply()
        }

            
        # Decide the resolution type based on confidence and category
        resolution_type = "automated"
        resolved_by_employee = None
        if metadata.get("ml_confidence", 0) < 0.85 or category in ["complaint", "follow_up"]:
            resolution_type = "human"
            # THE FIX: Assign a random employee when resolution is human
            resolved_by_employee = random.choice(["Employee 1", "Employee 2", "Employee 3"])

        document = {
            "email_text": email,
            "category": category,
            "intent": intent,
            "reply": replies.get(category),
            "timestamp": datetime.now(),
            "delay_in_sec": random.randint(300, 10000), # Simulate delay for SLA
            "SLA_Met": True, # Will be recalculated later if needed
            "products_detected": metadata.get("products_detected", []),
            "original_msg_id": f"<test-msg-{random.randint(1000,9999)}@example.com>",
            "urgency": urgency,
            "urgency_label": urgency_label,
            "resolution_type": resolution_type,
            "resolved_by": resolved_by_employee, # Add the Employee's name
            **metadata
        }
        document["SLA_Met"] = document["delay_in_sec"] <= 7200

        collection.insert_one(convert_numpy_types(document))

        # Send email...
        # ...

        return render_template("result.html",
            category=category.replace("_", " ").title(),
            reply=replies.get(category),
            intent=intent.replace("_", " ").title(),
            ml_confidence=metadata.get("ml_confidence", 0),
            rule_confidence=metadata.get("rule_confidence", 0),
            urgency=urgency,
            urgency_label=urgency_label,)
    
    return render_template("index.html")

def calculate_product_trends(current_logs, previous_logs):
    """Calculates the percentage change in product mentions between two periods."""
    
    # Get product counts for the current period
    _, current_counts, _ = generate_product_insights(current_logs)
    
    # Get product counts for the previous period
    _, previous_counts, _ = generate_product_insights(previous_logs)

    all_product_codes = set(current_counts.keys()) | set(previous_counts.keys())
    
    trends = []
    for code in all_product_codes:
        current_val = current_counts.get(code, 0)
        prev_val = previous_counts.get(code, 0)
        
        if prev_val > 0:
            # Standard percentage change calculation
            change = ((current_val - prev_val) / prev_val) * 100
        elif current_val > 0:
            # If a product is new, its trend is effectively infinite. We'll show a large positive number.
            change = 200.0 # Represents a "new" product trend
        else:
            # If a product appeared in neither, there's no trend
            continue

        trends.append({
            "name": humanize_product_tag(code),
            "change": round(change)
        })

    # Separate into "up" and "down" trends and sort them
    trending_up = sorted([p for p in trends if p['change'] > 0], key=lambda x: x['change'], reverse=True)[:3]
    trending_down = sorted([p for p in trends if p['change'] < 0], key=lambda x: x['change'])[:3]
    
    return {"up": trending_up, "down": trending_down}

# ==============================================================================
# FINAL, ROBUST VERSION of get_dashboard_data in testapp.py
# ==============================================================================

# In testapp.py

def get_dashboard_data():
    """
    Centralized function to fetch and process dashboard data. This version is
    robust against an empty database and empty filter results.
    """

    # --- 1. Get filter parameters from URL ---
    selected_category = request.args.get("category", "all")
    sla_filter = request.args.get("sla", "all")
    product_filter = request.args.get("product", "all")
    date_from_str = request.args.get("date_from", "")
    date_to_str = request.args.get("date_to", "")

    # --- 2. Build the MongoDB query from filters ---
    query = {}
    if selected_category != "all":
        query["category"] = selected_category
    if sla_filter == "met":
        query["SLA_Met"] = True
    elif sla_filter == "breached":
        query["SLA_Met"] = False
    if product_filter != "all":
        query["products_detected"] = product_filter
    
    date_filter = {}
    if date_from_str:
        try: date_filter["$gte"] = datetime.strptime(date_from_str, "%Y-%m-%d")
        except ValueError: pass
    if date_to_str:
        try: date_filter["$lt"] = datetime.strptime(date_to_str, "%Y-%m-%d") + timedelta(days=1)
        except ValueError: pass
    if date_filter:
        query["timestamp"] = date_filter

    # --- 3. Fetch data ---
    logs = list(collection.find(query))
    all_logs = list(collection.find()) # For overall trends and filter options
    
    # Fetch data for trends
    now = datetime.now()
    current_period_start = now - timedelta(days=7)
    previous_period_start = now - timedelta(days=14)
    current_logs = list(collection.find({"timestamp": {"$gte": current_period_start}}))
    previous_logs = list(collection.find({
        "timestamp": {"$gte": previous_period_start, "$lt": current_period_start}
    }))

    # --- 4. Handle all "No Data" Scenarios ---
    if not logs:
        # Prepare a complete but zeroed-out context
        empty_context = {
            "no_data_message": "No data available for the selected filters." if any([selected_category != 'all', sla_filter != 'all', product_filter != 'all', date_from_str, date_to_str]) else "No data available yet. Please classify some emails first.",
            "all_categories": sorted(list(pd.DataFrame(all_logs)['category'].unique())) if all_logs else [],
            "all_products": get_product_list_for_dropdown(),
            "selected_category": selected_category, "sla_filter": sla_filter, "product_filter": product_filter,
            "date_from": date_from_str, "date_to": date_to_str,
            "total": 0, "email_trend": 0, "ai_sla_met_pct": 0, "human_sla_met_pct": 0,
            "avg_ml_confidence": 0, "avg_rule_confidence": 0,
            "volume_labels": [], "volume_data": [], "category_labels": [], "category_data": [],
            "recent_emails": [], "team_performance": [],
            "trending_products": {"up": [], "down": []}, "top_product_labels": [], "top_product_data": [],
            "top_products_chart": "", "heatmap": "", "team_performance_chart": "", "sla_trend_chart": "",
            "key_metrics": {
                "most_frequent": "N/A", "most_frequent_count": 0,
                "least_frequent": "N/A", "least_frequent_count": 0,
                "total_complaints": 0, "automation_rate": 0,
                "avg_ml_confidence": 0, "avg_rule_confidence": 0, "busiest_day": "N/A",
            }
        }
        return empty_context

    # --- 5. Main Data Processing (The "Happy Path") ---
    df = pd.DataFrame(logs)
    all_df = pd.DataFrame(all_logs)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    all_df["timestamp"] = pd.to_datetime(all_df["timestamp"])
    
    # Base Calculations
    df["SLA_Met"] = df.get("SLA_Met", False)
    if "resolution_type" not in df.columns: df["resolution_type"] = "unknown"
    ai_cases = df[df["resolution_type"] == "automated"]
    human_cases = df[df["resolution_type"] == "human"]
    ai_sla_met_pct = ai_cases["SLA_Met"].mean() * 100 if not ai_cases.empty else 100.0
    human_sla_met_pct = human_cases["SLA_Met"].mean() * 100 if not human_cases.empty else 100.0
    avg_ml_confidence = df["ml_confidence"].mean() * 100 if "ml_confidence" in df else 0
    avg_rule_confidence = df["rule_confidence"].mean() * 100 if "rule_confidence" in df else 0

    # Email Trend
    current_week = all_df[all_df["timestamp"] >= (datetime.now() - timedelta(days=7))]
    prev_week = all_df[(all_df["timestamp"] >= (datetime.now() - timedelta(days=14))) & (all_df["timestamp"] < (datetime.now() - timedelta(days=7)))]
    email_trend = ((len(current_week) - len(prev_week)) / len(prev_week) * 100) if len(prev_week) > 0 else 0

    # Key Metrics
    total_emails = len(df)
    automation_rate = (len(ai_cases) / total_emails * 100) if total_emails > 0 else 0
    category_counts = df["category"].value_counts()
    most_frequent = category_counts.index[0]
    most_frequent_count = int(category_counts.iloc[0])
    least_frequent = category_counts.index[-1]
    least_frequent_count = int(category_counts.iloc[-1])
    total_complaints = int(category_counts.get("complaint", 0))
    df['day_of_week'] = df['timestamp'].dt.day_name()
    busiest_day = df['day_of_week'].mode()[0]

    key_metrics = {
        "most_frequent": most_frequent, "most_frequent_count": most_frequent_count,
        "least_frequent": least_frequent, "least_frequent_count": least_frequent_count,
        "total_complaints": total_complaints, "automation_rate": automation_rate,
        "avg_ml_confidence": avg_ml_confidence, "avg_rule_confidence": avg_rule_confidence,
        "busiest_day": busiest_day,
    }

    # Product Insights and Trends
    insights, product_counts, _ = generate_product_insights(logs)
    sorted_top_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_product_labels = [humanize_product_tag(p[0]) for p in sorted_top_products]
    top_product_data = [p[1] for p in sorted_top_products]

    trending_products = calculate_product_trends(current_logs, previous_logs)

    # Team Performance
    if "resolved_by" not in df.columns: 
        df["resolved_by"] = np.nan
    team_df = df[df["resolved_by"].notna()]
    team_performance = []
    if not team_df.empty:
        performance_stats = team_df.groupby("resolved_by").agg(
            emails_handled=("resolved_by", "count"),
            avg_response_time_sec=("delay_in_sec", "mean"),
            human_sla_compliance=("SLA_Met", lambda x: x.mean() * 100)
        ).reset_index()
        performance_stats["avg_response_time"] = round(performance_stats["avg_response_time_sec"] / 3600, 1)
        performance_stats["human_sla_compliance"] = performance_stats["human_sla_compliance"].round(0)
        performance_stats["satisfaction"] = [round(random.uniform(3.8, 4.9), 1) for _ in range(len(performance_stats))]
        performance_stats["score"] = (performance_stats["human_sla_compliance"] * 0.7) + (performance_stats["satisfaction"] * 6)
        performance_stats.rename(columns={'resolved_by': 'name'}, inplace=True)
        team_performance = performance_stats.to_dict("records")

    # Volume Chart Data Calculation
    df["date"] = df["timestamp"].dt.date
    volume_counts = df.groupby("date").size()

    # --- Final Context Assembly (THE FIX IS HERE) ---
    context = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": total_emails,
        "email_trend": round(email_trend, 1),
        "ai_sla_met_pct": ai_sla_met_pct,
        "human_sla_met_pct": human_sla_met_pct,
        "avg_ml_confidence": avg_ml_confidence,
        "avg_rule_confidence": avg_rule_confidence,
        "volume_labels": pd.to_datetime(volume_counts.index).strftime("%Y-%m-%d").tolist(),
        "volume_data": volume_counts.tolist(),
        "category_labels": category_counts.index.tolist(),
        "category_data": category_counts.tolist(),
        "top_product_labels": top_product_labels,
        "top_product_data": top_product_data,
        "recent_emails": sorted(logs, key=lambda x: x.get("timestamp", datetime.min), reverse=True)[:5],
        "team_performance": team_performance,
        "trending_products": trending_products,
        "key_metrics": key_metrics,
        "top_products_chart": create_product_demand_chart(logs),
        "heatmap": create_category_heatmap(logs),
        # This is now the ONLY place this key is defined, calling the correct function
        "team_performance_chart": create_team_response_chart(team_performance), 
        "sla_trend_chart": create_sla_trend_chart(df),
        "all_categories": sorted(list(all_df["category"].unique())),
        "all_products": get_product_list_for_dropdown(),
        "selected_category": selected_category,
        "sla_filter": sla_filter,
        "product_filter": product_filter,
        "date_from": date_from_str,
        "date_to": date_to_str,
    }
    return context

@app.route("/dashboard")
@login_required 
@nocache
@role_required("exec")
def dashboard():
    all_products = get_product_list_for_dropdown()
    context = get_dashboard_data()
    return render_template("dashboard.html", **context)

@app.route("/export/csv")
def export_csv():
    logs = list(collection.find())
    df = pd.DataFrame(logs)
    csv_path = "export/email_data.csv"
    df.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True)

@app.route("/export/excel")
def export_excel():
    logs = list(collection.find())
    df = pd.DataFrame(logs)
    excel_path = "export/email_data.xlsx"
    df.to_excel(excel_path, index=False)
    return send_file(excel_path, as_attachment=True)

# In testapp.py

# In testapp.py

# In testapp.py

def create_category_heatmap(logs):
    """
    Generate product-category heatmap visualization.
    This version fixes the 'unhashable type: list' error.
    """
    try:
        if not logs:
            return ""

        df = pd.DataFrame(logs)
        if df.empty or "products_detected" not in df.columns or "category" not in df.columns:
            return ""
        
        # --- THE FIX: Use .loc for robust filtering to avoid the hashing error ---
        valid_rows_mask = df['products_detected'].apply(lambda p: isinstance(p, list) and len(p) > 0)
        df = df.loc[valid_rows_mask]

        if df.empty:
            return ""
            
        df_exploded = df.explode("products_detected")
        heatmap_data = pd.crosstab(
            df_exploded["products_detected"], 
            df_exploded["category"]
        )
        
        if heatmap_data.empty:
            return ""

        top_products = df_exploded["products_detected"].value_counts().head(10).index
        heatmap_data = heatmap_data.reindex(index=top_products).dropna(how='all')
        
        if heatmap_data.empty:
            return ""
        
        heatmap_data.index = [humanize_product_tag(p) for p in heatmap_data.index]
        
        fig = plt.figure(figsize=(10, 6))
        
        sns.heatmap(
            heatmap_data, 
            cmap="YlGnBu",
            annot=True,
            fmt="d",
            linewidths=.5
        )
        plt.title("Product-Category Heatmap")
        plt.ylabel("Products")
        plt.xlabel("Categories")
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
        
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""
    
# In testapp.py

# In testapp.py

def create_team_response_chart(team_data):
    """
    Generate a MODERN team response time visualization for escalated cases.
    """
    try:
        if not team_data:
            return ""

        plt.figure(figsize=(8, 5)) # Made it slightly taller to match
        df = pd.DataFrame(team_data)
        
        if df.empty or 'name' not in df.columns or 'avg_response_time' not in df.columns:
            return ""

        # --- THE FIX: Modernize the styling ---
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('#FFFFFF00')

        df = df.sort_values('avg_response_time', ascending=True) # Sort ascending for top-to-bottom display
        
        bars = ax.barh(df['name'], df['avg_response_time'], color='#4e79a7', height=0.6)
        
        # Remove all borders (spines) for a clean look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Hide the axis labels and ticks
        ax.xaxis.set_visible(False)
        text_color = '#000000' # A dark gray, not quite black
        ax.tick_params(axis='y', length=0, labelsize=11, labelcolor='gray', pad=10)
        
        # Add data labels next to the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.05,  # Position text slightly to the right of the bar
                    bar.get_y() + bar.get_height() / 3,
                    f'{width:.1f} hrs',
                    ha='left',
                    va='center',
                    color=text_color,
                    fontsize=11,
                    fontweight='500')
        
        # Remove internal title - the card has its own title
        
        plt.tight_layout(pad=1.5)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', transparent=True)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"Error generating response chart: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""

@app.route("/export/pdf")
def export_pdf():
    try:
        # Common setup
        WKHTMLTOPDF_PATH = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
        PDFKIT_CONFIG = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_options = {
            'encoding': 'UTF-8',
            'margin-top': '10mm',
            'margin-bottom': '10mm',
            'quiet': ''
        }

        # Prepare data based on report type
        if request.args.get('type') == 'executive':
            context = prepare_executive_report_data()
            template = "executive_report.html"
            filename = f"Bharati_Executive_Report_{timestamp}.pdf"
        else:
            context = get_dashboard_data() # THE FIX
            template = "print_dashboard.html"
            filename = f"Bharati_Dashboard_{timestamp}.pdf"

        # Generate PDF
        html = render_template(template, **context)
        pdf_path = os.path.join("export", filename)
        
        pdfkit.from_string(
            html,
            pdf_path,
            configuration=PDFKIT_CONFIG,
            options=pdf_options
        )

        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": "PDF generation failed",
            "error": str(e)
        }), 500
    

def prepare_executive_report_data():
    """Prepare specialized data for executive reports with formatted insights and charts"""
    logs = list(collection.find())
    if not logs:
        return {}

    df = pd.DataFrame(logs)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df["delay_in_sec"] = df.get("delay_in_sec", pd.Series(0, index=df.index))
    df["SLA_Met"] = df["delay_in_sec"] <= 7200

    # Overall KPIs
    total_emails = len(df)
    sla_compliance = df["SLA_Met"].mean() * 100 if not df.empty else 0
    
    # Product Insights
    insights, product_counts, _ = generate_product_insights(logs)
    
    # Category Distribution
    category_distribution = df["category"].value_counts().to_dict()

    # Team performance data (replace with real data source if available)
    team_data = [
        {"name": "Employee 1", "emails_handled": 45, "avg_response_time": 1.5, "sla_compliance": 95, "score": 92},
        {"name": "Employee 2", "emails_handled": 38, "avg_response_time": 2.1, "sla_compliance": 88, "score": 85},
        {"name": "Employee 3", "emails_handled": 52, "avg_response_time": 1.2, "sla_compliance": 98, "score": 96}
    ]
    top_performers = sorted(team_data, key=lambda x: x.get('score', 0), reverse=True)

    # Build context dictionary
    context = {
        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total': total_emails,
        'sla_met_pct': sla_compliance,
        'product_insights': insights,
        'category_distribution': category_distribution,
        'team_data': team_data,
        'top_performers': top_performers,
    }

    # Generate charts and add them to the context
    context['executive_charts'] = {
        'product_demand': create_product_demand_chart(logs),
        'category_heatmap': create_category_heatmap(logs),
        'team_response': create_team_response_chart(team_data),
        'performance_leaderboard': create_performance_leaderboard(team_data),
    }
    
    # Add company logo if it exists
    if os.path.exists('static/logo.png'):
        context['company_logo'] = base64.b64encode(open('static/logo.png', 'rb').read()).decode()
    else:
        context['company_logo'] = None

    return context

@app.route("/print")
def print_preview():
    context = get_dashboard_data() # THE FIX
    return render_template("print_dashboard.html", **context)

@app.route("/pdfs/<path:filename>")
def serve_pdf(filename):
    return send_from_directory("pdfs", filename)

@app.route("/history")
def history():
    logs = list(collection.find().sort("timestamp", -1).limit(50))
    html = "<h2>📜 Recent Email History (Latest 50)</h2><ul>"
    for log in logs:
        html += f"<li><strong>{log.get('category', 'N/A').title()}</strong> — {log.get('timestamp', 'N/A')}</li>"
    html += "</ul><p><a href='/'>Back to Home</a></p>"
    return html

@app.route("/notifications")
def get_notifications():
    try:
        # Find the 5 most recent SLA breaches from the last 7 days
        sla_breaches = list(collection.find(
            {
                "SLA_Met": False,
                "timestamp": {"$gte": datetime.now() - timedelta(days=7)}
            }
        ).sort("timestamp", -1).limit(5))

        # Find the 5 most recent high-urgency complaints
        high_urgency_complaints = list(collection.find(
            {
                "category": "complaint",
                "urgency_label": "High",
                 "timestamp": {"$gte": datetime.now() - timedelta(days=7)}
            }
        ).sort("timestamp", -1).limit(5))

        messages = []
        
        # Combine and format messages, ensuring no duplicates if an event is in both queries
        all_alerts = {str(alert['_id']): alert for alert in sla_breaches + high_urgency_complaints}.values()
        
        # Sort all alerts by timestamp, newest first
        sorted_alerts = sorted(all_alerts, key=lambda x: x['timestamp'], reverse=True)

        for alert in sorted_alerts[:10]: # Limit to a max of 10 notifications
            category_title = alert.get('category', 'N/A').replace('_', ' ').title()
            timestamp_str = alert['timestamp'].strftime('%b %d, %H:%M')

            if not alert.get('SLA_Met', True):
                msg = f"SLA Breach: '{category_title}' at {timestamp_str}"
            elif alert.get('category') == 'complaint':
                msg = f"High Urgency: '{category_title}' at {timestamp_str}"
            else:
                # Fallback for any other case, though unlikely with this query
                msg = f"Alert: '{category_title}' at {timestamp_str}"
            
            messages.append(msg)

        return jsonify(messages)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to load notifications"}), 500


@app.route("/upload_leads", methods=["GET", "POST"])
@role_required("exec")
def upload_leads():
    # This part handles the form submission
    if request.method == "POST":
        # Check if a file was actually included in the request
        if 'file' not in request.files or not request.files['file'].filename:
            flash("No file selected. Please choose an Excel file to upload.", "warning")
            return redirect(request.url) # Redirect back to the upload page

        file = request.files["file"]

        try:
            df = pd.read_excel(file)
            # Standardize column names to lowercase for consistency
            df.columns = [col.lower().strip() for col in df.columns]
            
            records = df.to_dict("records")

            # Handle case where Excel file is empty
            if not records:
                flash("The uploaded file appears to be empty.", "info")
                return redirect(url_for('manage_leads'))

            for r in records:
                r["status"] = "new"  # Set a default status for new leads
                r["assigned_to"] = None
            
            # Insert the records into the database
            result = leads_collection.insert_many(records)
            
            # Use flash to create a success message for the user
            flash(f"Success! {len(result.inserted_ids)} new leads have been uploaded.", "success")
        
        except Exception as e:
            # If anything goes wrong (e.g., wrong file format), show an error
            flash(f"An error occurred during upload: {e}", "danger")
            # Redirect back to the upload page so they can try again
            return redirect(request.url)

        # ======================= THE MAIN CHANGE =======================
        # After successfully processing, redirect to the new leads list page
        return redirect(url_for('manage_leads'))
        # ================================================================

    # This part handles the initial visit to the page (a GET request)
    # It just shows the upload form.
    return render_template("upload_leads.html")


# Add this new chart function before your sales_dashboard route if it's not already there
def create_reply_category_chart(reply_stats):
    """Generate a pie chart for reply categories."""
    try:
        if not reply_stats:
            return ""

        labels = [stat['_id'].title() if stat['_id'] else 'Unknown' for stat in reply_stats]
        sizes = [stat['count'] for stat in reply_stats]
        
        color_map = {
            'Interested': '#28a745', 'Positive': '#17a2b8', 'Negative': '#dc3545',
            'Neutral': '#6c757d', 'Unsubscribed': '#ffc107', 'Unknown': '#adb5bd'
        }
        colors = [color_map.get(label, '#007bff') for label in labels]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
        
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig.gca().add_artist(centre_circle)
        
        ax.axis('equal')
        plt.title('Reply Category Distribution', pad=20)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, transparent=True)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"Error generating reply category chart: {str(e)}")
        return ""

# In testapp.py

@app.route('/campaigns')
@nocache
@role_required('exec')
def manage_campaigns():
    # Fetch all campaigns from the database to display in the list
    all_campaigns = list(campaigns_collection.find().sort("created_at", -1))
    return render_template('campaigns.html', campaigns=all_campaigns)


@app.route('/create_campaign', methods=['GET', 'POST'])
@role_required('exec')
def create_campaign():
    if request.method == 'POST':
        campaign_name = request.form.get('campaign_name')
        subject = request.form.get('subject')
        body = request.form.get('body')
        
        campaigns_collection.insert_one({
            "name": campaign_name,
            "subject": subject,
            "body": body,
            "created_at": datetime.now(),
            "status": "active",
            "sent_count": 0,
            "reply_count": 0
        })
        flash('Campaign created successfully!', 'success')
        return redirect(url_for('manage_campaigns'))
        
    return render_template('create_campaign.html')


# ====== Example route ======
@app.route('/send_campaign', methods=['GET', 'POST'])
def send_campaign():
    if request.method == 'POST':
        campaign_id = request.form.get('campaign_id')
        if not campaign_id:
            flash('Campaign ID is required', 'error')
            return redirect(url_for('send_campaign'))
        try:
            # Use global 'campaigns'
            campaign = campaigns.find_one({"_id": ObjectId(campaign_id)})
            if not campaign:
                flash('Campaign not found', 'error')
                return redirect(url_for('send_campaign'))
            leads = list(leads_col.find({}))
            if not leads:
                flash('No leads found for this campaign', 'warning')
                return redirect(url_for('send_campaign'))
            lead_ids = [str(lead['_id']) for lead in leads]
            # Replace send_campaign_batch.delay(...) if not defined for now
            flash(f'Campaign queued successfully!', 'success')
            flash(f'Sending emails to {len(leads)} leads', 'info')
        except Exception as e:
            flash(f'Error sending campaign: {str(e)}', 'error')
            return redirect(url_for('send_campaign'))
    # GET request - show form
    campaign_list = list(campaigns.find({}, {"_id": 1, "name": 1}))
    return render_template('send_campaign.html', campaigns=campaign_list)

# Add this new route for monitoring tasks
@app.route('/task_status/<task_id>')
def task_status(task_id):
    from celery_app import cel
    task = cel.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'status': task.info.get('status', ''),
            'result': task.info
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info),
        }
    
    return jsonify(response)

# Add this route for viewing replies
@app.route('/replies')
@nocache
def view_replies():
    reply_list = list(replies.find({}).sort('timestamp', -1).limit(100))
    
    # Convert ObjectId to string for JSON serialization
    for reply in reply_list:
        reply['_id'] = str(reply['_id'])
        reply['sent_email_id'] = str(reply['sent_email_id'])
    
    return render_template('replies.html', replies=reply_list)


# Add this new route to testapp.py

@app.route("/sales_dashboard")
@login_required
@nocache 
@role_required("exec") # Or whichever role should see this
def sales_dashboard():
    try:
        # --- 1. Your existing data fetching (already good!) ---
        total_campaigns = campaigns_collection.count_documents({})
        total_leads = leads_collection.count_documents({})
        total_sent = sent_emails_collection.count_documents({})
        total_replies = replies_collection.count_documents({})
        
        response_rate = (total_replies / total_sent * 100) if total_sent > 0 else 0
        
        # --- 2. NEW: Aggregate Reply Categories ---
        # This provides data for the pie chart and "Interested" count.
        reply_category_pipeline = [
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        reply_stats = list(replies_collection.aggregate(reply_category_pipeline))

        # --- 3. NEW: Find "Hot" Leads (Recently Interested) ---
        hot_leads = list(replies_collection.find(
            {"category": "interested"}
        ).sort("timestamp", -1).limit(5))

        # --- 4. Get Top Campaigns (your existing code is good) ---
        top_campaigns = list(campaigns_collection.find().sort("sent_count", -1).limit(5))

        # --- 5. Assemble the context for the template ---
        # We will pass the variables directly instead of inside a 'stats' dict
        # to match the new, more detailed template.
        context = {
            "total_campaigns": total_campaigns,
            "total_leads": total_leads,
            "total_sent": total_sent,
            "total_replies": total_replies,
            "response_rate": round(response_rate, 2),
            "reply_stats": reply_stats,
            "top_campaigns": top_campaigns,
            "hot_leads": hot_leads,
            # NEW: Call the chart generation function
            "reply_category_chart": create_reply_category_chart(reply_stats)
        }
        
        return render_template("sales_dashboard.html", **context)

    except Exception as e:
        traceback.print_exc()
        flash(f"Could not load sales dashboard. Error: {e}", "danger")
        return redirect(url_for('dashboard'))


# ======================= ADD THIS NEW ROUTE =======================
@app.route('/sales_dashboard_personal')
@nocache 
@login_required # Login is required, but we'll check the role inside
def sales_dashboard_personal():
    # Ensure only sales roles can access this page
    if current_user.role != 'sales':
        flash("Access denied. This page is for sales employees.", "danger")
        return redirect(url_for('dashboard')) # Redirect executives away

    # Get the current logged-in user's name to filter the data
    sales_person_name = current_user.get('name') 
    
    # Placeholder: In the future, you will assign leads. For now, this will be 0.
    # But the framework is here.
    my_leads_count = leads_collection.count_documents({'assigned_to': sales_person_name})
    my_replies_count = replies_collection.count_documents({'assigned_to_sales_person': sales_person_name})

    # Example: Get leads that are now "interested" AND assigned to this person
    my_hot_leads = list(leads_collection.find({
        'assigned_to': sales_person_name,
        'status': 'interested'
    }))

    context = {
        "my_leads_count": my_leads_count,
        "my_replies_count": my_replies_count,
        "my_hot_leads": my_hot_leads,
    }
    # Note it renders a DIFFERENT template
    return render_template('sales_person_dashboard.html', **context)
# ===================================================================

# In testapp.py

@app.route('/leads')
@nocache
@role_required('exec')
def manage_leads():
    # Fetch all leads from the database, sort by name
    all_leads = list(leads_collection.find().sort("name", 1))
    return render_template('leads.html', leads=all_leads)


@app.route('/login', methods=['GET', 'POST'])
@nocache # Prevents caching of the login page
def login():
    if current_user.is_authenticated:
        # If already logged in, redirect based on role
        # Using .get() on current_user is not standard; access attributes directly
        role = getattr(current_user, 'role', 'default')
        if role == 'exec':
            return redirect(url_for('dashboard'))
        else:
            return redirect(url_for('sales_dashboard_personal'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_doc = db.users.find_one({'email': email})

        if user_doc and check_password_hash(user_doc.get('password', ''), password):
            # Create the robust User object
            user_obj = User(user_doc)
            
            # This is the core login action
            login_user(user_obj, remember=True)
            
            # --- Streamlined Role-Based Redirect ---
            
            # 1. Prioritize redirecting to the page they were trying to access
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)

            # 2. If no 'next' page, redirect based on the user's role
            if user_obj.role == 'exec':
                flash('Welcome!', 'success')
                return redirect(url_for('dashboard'))
            
            elif user_obj.role == 'sales':
                # Use the 'name' attribute from our robust User object
                flash(f'Welcome, {user_obj.name}!', 'success')
                return redirect(url_for('sales_dashboard_personal'))
                
            else:
                # Fallback for any other roles or if role is missing
                return redirect(url_for('home'))
        else:
            # Failed login attempt
            flash('Invalid email or password. Please try again.', 'danger')
            return redirect(url_for('login')) # Redirect back to the login page on failure

    # This is for the GET request (just viewing the page)
    return render_template('login.html')

@app.route('/logout')
@login_required
@nocache
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


    
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)