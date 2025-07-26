import os
import pdfkit
from datetime import datetime
from flask import render_template
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def generate_executive_pdf(context):
    """Generate CEO-friendly PDF with custom formatting"""
    
    # --- 1. Format Product Names for Readability ---
    formatted_products = []
    for product in context.get('top_product_mentions', []):
        formatted_products.append(humanize_product_tag(product))
    context['formatted_products'] = formatted_products

    # --- 2. Create Custom Charts for PDF ---
    context['executive_charts'] = {
        'sla_summary': create_sla_summary_chart(context),
        'product_demand': create_product_demand_chart(context)
    }

    # --- 3. Render HTML Template ---
    html = render_template("executive_report.html", **context)

    # --- 4. Generate PDF ---
    pdf_path = f"export/executive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    pdfkit.from_string(
        html,
        pdf_path,
        configuration=pdfkit.configuration(wkhtmltopdf='/usr/local/bin/wkhtmltopdf'),
        options={
            'encoding': 'UTF-8',
            'margin-top': '10mm',
            'margin-bottom': '10mm',
            'quiet': ''
        }
    )
    
    return pdf_path

def humanize_product_tag(tag: str) -> str:
    """Enhanced product name formatting"""
    replacements = {
        "fireextinguishers.": "",
        "drychemicalpowder": "Dry Chemical",
        "trolleymounted": "Trolley",
        "portable(squeezegripcartridgetype)": "Portable",
        "map(storedpressure)/abc": "ABC Type",
        "kgs": "kg",
        "litres": "L",
        "_": " "
    }
    for k, v in replacements.items():
        tag = tag.replace(k, v)
    return tag.title()

def create_sla_summary_chart(context):
    """Generate SLA summary visualization"""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(['SLA Met', 'Breached'], 
           [context['sla_met_pct'], 100-context['sla_met_pct']],
           color=['#4CAF50', '#F44336'])
    ax.set_title('SLA Compliance Overview')
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode()

def create_product_demand_chart(context):
    """Generate top products visualization"""
    products = context['formatted_products'][:5]
    counts = [p['count'] for p in context['product_insights']][:5]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(products, counts, color='#2196F3')
    ax.set_title('Top Product Demand')
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode()