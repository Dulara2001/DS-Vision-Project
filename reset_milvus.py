# reset_milvus.py

from pymilvus import connections, utility

connections.connect(host="localhost", port="19530")

if utility.has_collection("visitor_faces"):
    utility.drop_collection("visitor_faces")
    print("✅ Dropped old collection: visitor_faces")
else:
    print("ℹ️ Collection didn't exist.")