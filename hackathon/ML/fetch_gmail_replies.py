# START OF FILE: fetch_gmail_replies.py (IMPROVED)

from gmail_utils import gmail_authenticate, get_clean_email_body
from pymongo import MongoClient
import re
import os
from datetime import datetime

# IMPROVEMENT: Centralize configuration
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://bharatiadmin:Secure123@cluster0.nt2rwkw.mongodb.net/")
client = MongoClient(MONGO_URI)
db = client["bharati_ai"]
replies_collection = db["replies"]
leads_collection = db["sales_leads"]
sent_emails_collection = db["sent_emails"]

def clean_reply_body(body):
    """ A more robust way to remove quoted text and signatures. """
    # Remove standard reply headers
    body = re.sub(r'On.*wrote:.*', '', body, flags=re.DOTALL)
    # Remove ">" quote lines
    lines = [line for line in body.splitlines() if not line.strip().startswith('>')]
    # Attempt to remove common signature lines
    cleaned_lines = []
    for line in lines:
        if line.strip() in ['--', '---', '–––']: # Stop at signature markers
            break
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

def analyze_reply_intent(text):
    """ Simple keyword-based analysis to categorize a reply. """
    text = text.lower()
    positive_keywords = ['interested', 'schedule a call', 'sounds good', 'let\'s talk', 'send more details', 'proposal']
    negative_keywords = ['not interested', 'unsubscribe', 'remove me', 'not a good fit', 'stop sending']
    
    if any(kw in text for kw in positive_keywords):
        return 'interested'
    if any(kw in text for kw in negative_keywords):
        return 'negative'
    return 'neutral' # Default if no strong signal is found

def fetch_and_process_replies():
    """ Fetches replies to sent campaign emails, analyzes them, and updates lead status. """
    service = gmail_authenticate()
    # Query for unread messages that are replies (have a thread)
    query = "is:unread" 
    results = service.users().messages().list(userId="me", q=query).execute()
    messages = results.get("messages", [])

    print(f"Found {len(messages)} unread messages to check for replies.")

    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"], format='full').execute()
        headers = msg_data['payload']['headers']
        
        # Check if this email is a reply to something we sent
        in_reply_to = next((h['value'] for h in headers if h['name'] == 'In-Reply-To'), None)
        if not in_reply_to:
            continue # This is not a reply, skip it

        # Find the original email we sent using the 'In-Reply-To' header
        original_sent_email = sent_emails_collection.find_one({'message_id': in_reply_to})
        if not original_sent_email:
            continue # It's a reply, but not to one of our tracked campaigns

        lead_id = original_sent_email['lead_id']
        subject, body = get_clean_email_body(msg_data['payload'])
        cleaned_body = clean_reply_body(body)
        
        # IMPROVEMENT: Analyze the intent and update the lead
        intent = analyze_reply_intent(cleaned_body)
        print(f"\n--- Found Reply for Lead ID: {lead_id} ---")
        print(f"Intent Detected: {intent.upper()}")

        # Update the lead's status in the leads collection
        leads_collection.update_one(
            {'_id': lead_id},
            {'$set': {'status': intent, 'last_reply_at': datetime.now()}}
        )
        print(f"✅ Updated lead status to '{intent}'.")
        
        # Mark as read
        service.users().messages().modify(userId='me', id=msg['id'], body={'removeLabelIds': ['UNREAD']}).execute()

if __name__ == "__main__":
    fetch_and_process_replies()