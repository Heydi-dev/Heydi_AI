# app/api/api_v1/router.py
# 코드 설명: API 버전 1의 라우터를 정의합니다.
# 앞으로 추가될 엔드포인트들을 이 파일에 포함시킵니다.
from fastapi import APIRouter
from app.api.api_v1.endpoints import system, test_model

api_router = APIRouter()

# System endpoints
api_router.include_router(system.router, prefix="/system", tags=["system"])
api_router.include_router(test_model.router, prefix="/test-model", tags=["test-model"])

# You can include other endpoint routers here as needed
# api_router.include_router(other.router, prefix="/other", tags=["other"])