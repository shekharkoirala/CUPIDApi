from typing import Optional
from fastapi import HTTPException, Header
from os import getenv
from dotenv import load_dotenv

load_dotenv()

API_KEY: str | None = getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable is not set")


async def verify_api_key(X_API_Key: Optional[str] = Header(None)):
    if X_API_Key is None:
        raise HTTPException(status_code=401, detail="API Key is missing")
    if X_API_Key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return X_API_Key
