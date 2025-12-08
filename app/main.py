# app/main.py
# 코드 설명: FastAPI 애플리케이션의 진입점입니다.
from fastapi import FastAPI
from app.core.config import settings
from app.api.api_v1.router import api_router

app = FastAPI(title=settings.PROJECT_NAME)

# Register API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {"message": "Hello from Heydi_AI! Visit /docs for API documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)