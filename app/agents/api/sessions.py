from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..repositories import create_or_resume_session, get_room, get_session, list_lines
from ..schemas import LineRead, SessionCreate, SessionRead


router = APIRouter(tags=["sessions"])


@router.post("/api/rooms/{room_id}/session", response_model=SessionRead)
async def create_session(
    room_id: int, payload: SessionCreate, db: AsyncSession = Depends(get_db)
):
    room = await get_room(db, room_id)
    if room is None:
        raise HTTPException(404, "Room not found")
    if len(room.members) < 2:
        raise HTTPException(422, "A live room needs at least two characters")
    try:
        return await create_or_resume_session(db, room_id, payload.resume_session_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/api/sessions/{session_id}/transcript", response_model=list[LineRead])
async def transcript(session_id: int, db: AsyncSession = Depends(get_db)):
    if await get_session(db, session_id) is None:
        raise HTTPException(404, "Session not found")
    return await list_lines(db, session_id)


@router.get("/api/sessions/{session_id}/transcript.txt", response_class=PlainTextResponse)
async def transcript_text(session_id: int, db: AsyncSession = Depends(get_db)):
    if await get_session(db, session_id) is None:
        raise HTTPException(404, "Session not found")
    lines = await list_lines(db, session_id)
    return "\n".join(f"{line.speaker}: {line.text}" for line in lines)
