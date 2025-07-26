import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from email.message import EmailMessage
from email.mime.application import MIMEApplication
import base64

# Define scopes at the top level
# In utils.py

# BEFORE:
# SCOPES = ['https://www.googleapis.com/auth/gmail.send']

# AFTER: Add the permission to read and modify emails
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def gmail_authenticate():
    """Authenticate with Gmail API with proper error handling and token refresh"""
    creds = None
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
        except Exception as e:
            print(f"Error loading credentials: {e}")
            creds = None
    
    if not creds or not creds.valid:
        try:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            # Make sure the port is not in use or choose another
            creds = flow.run_local_server(port=0) 
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        except Exception as e:
            print(f"Authentication failed: {e}")
            raise Exception("Gmail authentication failed. Please check credentials.json")
    
    return build('gmail', 'v1', credentials=creds)

def send_email_with_attachments(to_email, subject, body, attachments=[], in_reply_to=None):
    """Send email with proper error handling and return the message ID."""
    try:
        service = gmail_authenticate()
        msg = EmailMessage()
        msg['To'] = to_email
        msg['From'] = "me"
        msg['Subject'] = subject
        msg.set_content(body, subtype='html')
        
        if in_reply_to:
            msg['In-Reply-To'] = in_reply_to
            msg['References'] = in_reply_to

        for path in attachments:
            if not os.path.exists(path):
                print(f"Attachment not found: {path}")
                continue
            with open(path, 'rb') as f:
                part = MIMEApplication(f.read(), _subtype="pdf")
                part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(path))
                msg.add_attachment(part.get_payload(decode=True), maintype='application', subtype='pdf', filename=os.path.basename(path))

        encoded_msg = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        send_message = {'raw': encoded_msg}
        
        sent_message = service.users().messages().send(userId="me", body=send_message).execute()
        print(f"Email sent successfully. Message ID: {sent_message['id']}")
        return sent_message['id']

    except Exception as e:
        print(f"Failed to send email: {e}")
        raise