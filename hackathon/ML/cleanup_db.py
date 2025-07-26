from pymongo import MongoClient

# --- IMPORTANT: MAKE SURE YOUR testapp.py is NOT RUNNING when you run this script ---

# Connect to your MongoDB
client = MongoClient("mongodb+srv://bharatiadmin:Secure123@cluster0.nt2rwkw.mongodb.net/")
db = client["bharati_ai"]
collection = db["email_logs"]

# Define the categories you want to remove
unwanted_categories = ["dispatch_update", "sales_approval", "feedback"]

# Create a query to find documents with these categories
query = { "category": { "$in": unwanted_categories } }

# Delete the matching documents
result = collection.delete_many(query)

print(f"Cleanup complete.")
print(f"Total documents removed: {result.deleted_count}")