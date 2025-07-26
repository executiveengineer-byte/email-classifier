from celery_app import cel
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient
from testapp import gmail_authenticate, send_email_with_attachments

client = MongoClient("mongodb+srv://bharatiadmin:bharati123@cluster0.nt2rwkw.mongodb.net/bharati_ai?retryWrites=true&w=majority")
db        = client["bharati_ai"]
leads_col = db["sales_leads"]
sent_col  = db["sent_emails"]
campaigns = db["sales_campaigns"]

@cel.task(bind=True, max_retries=3)
def send_campaign_email(self, campaign_id, lead_id):
    try:
        campaign = campaigns.find_one({"_id": ObjectId(campaign_id)})
        lead     = leads_col.find_one({"_id": ObjectId(lead_id)})

        subject = campaign["template"]["subject"].format(company=lead["company"])
        body    = campaign["template"]["body"].format(contact=lead["contact"])

        msg_id = send_email_with_attachments(
            to_email=lead["email"],
            body=body,
            attachments=[]
        )

        sent_col.insert_one({
            "campaign_id": campaign_id,
            "lead_id": lead_id,
            "msg_id": msg_id,
            "sent_at": datetime.utcnow(),
            "status": "sent"
        })

    except Exception as exc:
        self.retry(countdown=60, exc=exc)