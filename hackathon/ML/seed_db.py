# File: seed_db.py
from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from datetime import datetime
from bson import ObjectId





client = MongoClient("mongodb+srv://bharatiadmin:bharati123@cluster0.nt2rwkw.mongodb.net/bharati_ai?retryWrites=true&w=majority")
db = client["bharati_ai"]

# 1. Create a user to log in with
users_collection = db.users


# --- ADD THIS NEW BLOCK FOR YOUR NEW USER ---
if users_collection.count_documents({'email': 'amit.s@example.com'}) == 0:
    users_collection.insert_one({
        "name": "Amit Sharma",
        "email": "amit.s@example.com",
        "password": generate_password_hash("amit123", method="pbkdf2:sha256"),
        "role": "sales" 
    })
    print("User 'amit.s@example.com' created successfully.")
else:
    print("User 'amit.s@example.com' already exists.")
    
if users_collection.count_documents({'email': 'exec@example.com'}) == 0:
    users_collection.insert_one({
        "email": "exec@example.com",
        "password": generate_password_hash("exec123", method="pbkdf2:sha256"),
        "role": "exec"  # Make sure this role matches your @role_required decorator
    })
    print("User 'exec@example.com' created.")
else:
    print("User 'exec@example.com' already exists.")


# 2. Create a sales user
if users_collection.count_documents({'email': 'sales1@example.com'}) == 0:
    users_collection.insert_one({
        "name": "Sales Person One",
        "email": "sales1@example.com",
        "password": generate_password_hash("sales123", method="pbkdf2:sha256"),
        "role": "sales" # Note the different role
    })
    print("User 'sales1@example.com' created.")
else:
    print("User 'sales1@example.com' already exists.")

# 3. Create a sample campaign
campaigns_collection = db.sales_campaigns
if campaigns_collection.count_documents({'name': 'Welcome Campaign'}) == 0:
    campaigns_collection.insert_one({
        "name": "Welcome Campaign",
        "subject": "A special welcome from {{company}}!",
        "body": "Hi {{name}},\n\nWelcome! We're glad to have you here.\n\nThanks,\nThe Team",
        "created_at": datetime.now(),
        "status": "active"
    })
    print("Sample 'Welcome Campaign' created.")
else:
    print("Sample 'Welcome Campaign' already exists.")

print("Database seeding complete.")

