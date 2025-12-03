from fastapi import APIRouter
from app.api.api_v1.endpoints import system

api_router = APIRouter()

# System endpoints
api_router.include_router(system.router, prefix="/system", tags=["system"])

# You can include other endpoint routers here as needed
# api_router.include_router(other.router, prefix="/other", tags=["other"])