from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.core.database import get_db

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Spring Boot 서버와 간단한 통신 테스트용
    """
    return {"status": "ok", "message": "AI server is running"}

@router.get("/db-check")
async def db_connection_check(db: AsyncSession = Depends(get_db)):
    """
    PostgreSQL 연결 테스트용 (SELECT 1 실행)
    """
    try:
        result = await db.execute(text("SELECT 1"))
        return {"status": "ok", "db_response": result.scalar()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")