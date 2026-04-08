import uvicorn
from fastapi import FastAPI
from server.main import app as main_app

app = FastAPI()
app.mount("/", main_app)

def main():
    """Entry point for console script."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
