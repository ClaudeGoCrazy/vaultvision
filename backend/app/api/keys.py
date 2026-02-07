from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.auth import generate_api_key, verify_api_key
from app.models.models import APIKey

router = APIRouter(prefix="/api/v1", tags=["api-keys"])


@router.post("/keys")
async def create_api_key(body: dict, db: AsyncSession = Depends(get_db)):
    name = body.get("name", "default")
    key = generate_api_key()
    api_key = APIKey(key=key, name=name)
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)
    return {
        "id": api_key.id,
        "key": api_key.key,
        "name": api_key.name,
        "created_at": api_key.created_at.isoformat(),
    }


@router.get("/keys")
async def list_api_keys(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(APIKey).order_by(APIKey.created_at.desc()))
    keys = result.scalars().all()
    return [
        {
            "id": k.id,
            "key_preview": k.key[:8] + "...",
            "name": k.name,
            "created_at": k.created_at.isoformat(),
            "is_active": bool(k.is_active),
        }
        for k in keys
    ]


@router.delete("/keys/{key_id}")
async def deactivate_api_key(
    key_id: str,
    db: AsyncSession = Depends(get_db),
    _key: str = Depends(verify_api_key),
):
    key = await db.get(APIKey, key_id)
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    key.is_active = 0
    await db.commit()
    return {"message": "API key deactivated"}
