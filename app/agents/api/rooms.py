from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..repositories import create_room, delete_room, get_room, list_rooms, room_to_dict, update_room
from ..schemas import RoomCreate, RoomRead, RoomUpdate


router = APIRouter(prefix="/api/rooms", tags=["rooms"])


@router.get("", response_model=list[RoomRead])
async def get_all(db: AsyncSession = Depends(get_db)):
    return [room_to_dict(room) for room in await list_rooms(db)]


@router.post("", response_model=RoomRead, status_code=status.HTTP_201_CREATED)
async def create(payload: RoomCreate, db: AsyncSession = Depends(get_db)):
    try:
        room = await create_room(db, payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return room_to_dict(room)


@router.get("/{room_id}", response_model=RoomRead)
async def get_one(room_id: int, db: AsyncSession = Depends(get_db)):
    room = await get_room(db, room_id)
    if room is None:
        raise HTTPException(404, "Room not found")
    return room_to_dict(room)


@router.put("/{room_id}", response_model=RoomRead)
async def update(room_id: int, payload: RoomUpdate, db: AsyncSession = Depends(get_db)):
    room = await get_room(db, room_id)
    if room is None:
        raise HTTPException(404, "Room not found")
    try:
        room = await update_room(db, room, payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return room_to_dict(room)


@router.delete("/{room_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove(room_id: int, db: AsyncSession = Depends(get_db)):
    room = await get_room(db, room_id)
    if room is None:
        raise HTTPException(404, "Room not found")
    await delete_room(db, room)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
