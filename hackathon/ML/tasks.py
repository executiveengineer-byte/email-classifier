from celery_app import cel
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient
import logging
# ====================== CORRECT IMPORT ======================
from utils import gmail_authenticate, send_email_with_attachments
# ==========================================================

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection
client = MongoClient("mongodb+srv://bharatiadmin:bharati123@cluster0.nt2rwkw.mongodb.net/bharati_ai?retryWrites=true&w=majority")
db = client["bharati_ai"]
leads_collection = db["sales_leads"]
sent_emails_collection = db["sent_emails"]
campaigns_collection = db["sales_campaigns"]
replies_collection = db["replies"]

from textblob import TextBlob

def classify_reply_category(text):
    """Simple classifier for sales replies."""
    text = text.lower()
    if any(keyword in text for keyword in ["unsubscribe", "remove me", "stop sending"]):
        return "unsubscribed"
    if any(keyword in text for keyword in ["interested", "quote", "pricing", "like to know more", "send details"]):
        return "interested"
    if any(keyword in text for keyword in ["not interested", "not for us", "no thanks"]):
        return "negative"
    
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0.3:
        return "positive"
    if sentiment < -0.2:
        return "negative"
    return "neutral"

@cel.task
def check_for_replies():
    """
    Periodically checks for new replies, classifies them, and links them to leads/campaigns.
    """
    service = gmail_authenticate()
    results = service.users().messages().list(userId='me', q='is:unread label:inbox').execute()
    messages = results.get('messages', [])

    if not messages:
        print("No new replies found.")
        return "No new replies."

    for msg_summary in messages:
        msg_id = msg_summary['id']
        try:
            msg = service.users().messages().get(userId='me', id=msg_id, format='metadata').execute()
            
            headers = {h['name'].lower(): h['value'] for h in msg['payload']['headers']}
            
            in_reply_to_header = headers.get('in-reply-to') or headers.get('references')
            if not in_reply_to_header:
                continue

            in_reply_to_id = in_reply_to_header.split()[-1].strip('<>')

            original_sent_email = sent_emails_collection.find_one({"gmail_message_id": in_reply_to_id})

            if original_sent_email:
                lead_id = original_sent_email['lead_id']
                campaign_id = original_sent_email['campaign_id']
                
                full_msg = service.users().messages().get(userId='me', id=msg_id).execute()
                snippet = full_msg['snippet']
                category = classify_reply_category(snippet)

                replies_collection.insert_one({
                    "lead_id": lead_id,
                    "campaign_id": campaign_id,
                    "gmail_message_id": msg_id,
                    "from": headers.get('from'),
                    "subject": headers.get('subject'),
                    "snippet": snippet,
                    "category": category,
                    "timestamp": datetime.now()
                })

                leads_collection.update_one(
                    {"_id": ObjectId(lead_id)},
                    {"$set": {"status": category}}
                )

                service.users().messages().modify(userId='me', id=msg_id, body={'removeLabelIds': ['UNREAD']}).execute()
                print(f"Processed reply from lead {lead_id} for campaign {campaign_id}. Category: {category}")
        except Exception as e:
            logger.error(f"Failed to process message {msg_id}: {e}")
            
    return f"Processed {len(messages)} messages."

@cel.task(bind=True)
def send_campaign_batch(self, campaign_id, lead_ids):
    """
    Master task to queue individual email sending tasks for a whole campaign.
    This version includes progress updates.
    """
    try:
        total_leads = len(lead_ids)
        logger.info(f"Starting batch send for campaign {campaign_id} with {total_leads} leads")

        for index, lead_id in enumerate(lead_ids):
            send_campaign_email.delay(campaign_id, str(lead_id))
            self.update_state(state='PROGRESS', meta={'current': index + 1, 'total': total_leads, 'status': f'Queuing email {index + 1} of {total_leads}'})
        
        logger.info(f"Successfully queued {total_leads} email tasks for campaign {campaign_id}")
        return {'status': 'Batch processing complete', 'total_queued': total_leads}
    except Exception as exc:
        logger.error(f"Error in batch send for campaign {campaign_id}: {exc}")
        self.update_state(state='FAILURE', meta={'exc': str(exc)})
        raise exc

@cel.task(bind=True, max_retries=3, default_retry_delay=60)
def send_campaign_email(self, campaign_id, lead_id):
    """
    Sends a single campaign email to a specific lead and logs it with the Gmail Message-ID.
    """
    try:
        logger.info(f"Processing email for campaign {campaign_id} to lead {lead_id}")
        
        # ======================= FIX START =======================
        # REMOVED: from testapp import send_email_with_attachments
        # The function is already available from the global import at the top of the file.
        # ======================== FIX END ========================
        
        campaign = campaigns_collection.find_one({"_id": ObjectId(campaign_id)})
        if not campaign:
            logger.error(f"Campaign {campaign_id} not found in database.")
            return {"status": "failed", "error": "Campaign not found"}
            
        lead = leads_collection.find_one({"_id": ObjectId(lead_id)})
        if not lead:
            logger.error(f"Lead {lead_id} not found in database.")
            return {"status": "failed", "error": "Lead not found"}

        subject = campaign.get("subject", "").format(company=lead.get("company", ""), name=lead.get("name", ""))
        body = campaign.get("body", "").format(name=lead.get("name", ""), company=lead.get("company", ""))

        gmail_message_id = send_email_with_attachments(
            to_email=lead["email"],
            subject=subject,
            body=body,
            attachments=[]
        )

        sent_emails_collection.insert_one({
            "campaign_id": ObjectId(campaign_id),
            "lead_id": ObjectId(lead_id),
            "gmail_message_id": gmail_message_id,
            "sent_at": datetime.utcnow(),
            "status": "sent"
        })
        
        logger.info(f"Email sent successfully to lead {lead_id}. Gmail Message ID: {gmail_message_id}")
        return {"status": "sent", "gmail_message_id": gmail_message_id}

    except Exception as exc:
        logger.error(f"Error sending email to lead {lead_id}: {exc}. Retrying...")
        raise self.retry(exc=exc)