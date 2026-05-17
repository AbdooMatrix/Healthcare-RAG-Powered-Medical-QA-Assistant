from dotenv import load_dotenv
import os
load_dotenv()
key = os.getenv("GROQ_API_KEY", "")
print("GROQ key loaded:", bool(key), "|", key[:8] + "..." if key else "MISSING")