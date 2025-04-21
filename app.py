from fastapi import FastAPI
from pydantic import BaseModel
from title_verification import calculate_verification_score, rule_based_checks
import supabase

app = FastAPI()

SUPABASE_URL = "https://axzjwmwuwwmprsicdqxy.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF4emp3bXd1d3dtcHJzaWNkcXh5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzg5MjI5MzEsImV4cCI6MjA1NDQ5ODkzMX0.NiT7cMO3xnOpPP2U3BSV619U3_rgOkhFdaLoJY0tUPg"
supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

class TitleRequest(BaseModel):
    title: str

def check_title_existence(title: str) -> bool:
    """Check if a similar title exists in Supabase using FAISS"""
    response = supabase_client.table("titles").select("title_name").execute()

    if response.data:
        existing_titles = [row["title_name"].strip().lower() for row in response.data]
        return title.lower() in existing_titles
    return False

@app.post("/verify-title/")
async def verify_title(request: TitleRequest):
    title = request.title.strip()

    if check_title_existence(title):
        return {"status": "rejected", "message": "Title already exists in the database", "verification_score": 0}

    is_valid, message = rule_based_checks(title)
    if not is_valid:
        return {"status": "rejected", "message": message, "verification_score": 0}

    verification_score, explanation = calculate_verification_score(title)

    return {
        "status": "accepted" if verification_score > 50 else "rejected",
        "verification_score": verification_score,
        "message": explanation
    }
#uvicorn app:app --reload --host 127.0.0.1 --port 8080
