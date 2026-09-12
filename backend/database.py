"""
One MongoClient, one DB, one `users` collection for both auth profile
data AND the per-user stock watchlist - merged per your request so the
watchlist sits "side by side" with the rest of the user's info instead of
living in a separate `user_stocks` collection.

Each user document now looks like:

    {
      "_id": ObjectId(...),
      "google_id": "...",
      "email": "...",
      "username": "...",
      "photo": "...",
      "google_access_token": "...",
      "google_refresh_token": "...",
      "stocks": {
        "AAPL": {
          "live_price": 227.5,
          "predicted_price": 231.2,   # config.NO_PREDICTION_SENTINEL (-1) until predicted
          "updated_at": "2026-09-11T12:00:00+00:00"
        },
        ...
      }
    }

agents.py's get_user_stocks / store_user_stocks / store_predicted_price
read and write the "stocks" sub-document on the SAME users_collection
document instead of a second collection - see agents.py Section 3.
"""

from pymongo import MongoClient
import config

client = MongoClient(config.MONGODB_URI)
db = client[config.MONGODB_DB_NAME]

users_collection = db["users"]
users_collection.create_index("email", unique=True)
#users_collection.create_index("google_id", unique=True)

# Note: rag.py opens its OWN MongoClient against config.MONGODB_URI /
# MONGODB_DB_NAME / MONGODB_COLLECTION for the vector index - left as-is
# per instructions not to touch rag/. It's the same cluster, just a
# separate connection, which is fine (pymongo pools per-process anyway).
