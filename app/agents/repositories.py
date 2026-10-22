from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import Character, ConversationSession, Line, Room, RoomMember
from .schemas import CharacterCreate, CharacterUpdate, RoomCreate, RoomUpdate


async def list_characters(db: AsyncSession) -> list[Character]:
    return list((await db.scalars(select(Character).order_by(Character.name))).all())


async def get_character(db: AsyncSession, character_id: int) -> Character | None:
    return await db.get(Character, character_id)


async def create_character(db: AsyncSession, payload: CharacterCreate) -> Character:
    item = Character(**payload.model_dump())
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return item


async def update_character(
    db: AsyncSession, item: Character, payload: CharacterUpdate
) -> Character:
    for key, value in payload.model_dump().items():
        setattr(item, key, value)
    await db.commit()
    await db.refresh(item)
    return item


async def delete_character(db: AsyncSession, item: Character) -> None:
    await db.delete(item)
    await db.commit()


def room_to_dict(room: Room) -> dict[str, Any]:
    ordered = sorted(room.members, key=lambda member: member.position)
    return {
        "id": room.id,
        "name": room.name,
        "safe_word": room.safe_word,
        "settings": room.settings_json or {},
        "character_ids": [member.character_id for member in ordered],
        "characters": [member.character for member in ordered],
    }


async def list_rooms(db: AsyncSession) -> list[Room]:
    statement = (
        select(Room)
        .options(selectinload(Room.members).selectinload(RoomMember.character))
        .order_by(Room.name)
    )
    return list((await db.scalars(statement)).unique().all())


async def get_room(db: AsyncSession, room_id: int) -> Room | None:
    statement = (
        select(Room)
        .where(Room.id == room_id)
        .options(selectinload(Room.members).selectinload(RoomMember.character))
    )
    return (await db.scalars(statement)).unique().one_or_none()


async def _validate_character_ids(db: AsyncSession, ids: list[int]) -> None:
    if not ids:
        return
    existing = set((await db.scalars(select(Character.id).where(Character.id.in_(ids)))).all())
    missing = set(ids) - existing
    if missing:
        raise ValueError(f"Unknown character ids: {sorted(missing)}")


async def create_room(db: AsyncSession, payload: RoomCreate) -> Room:
    await _validate_character_ids(db, payload.character_ids)
    room = Room(name=payload.name, safe_word=payload.safe_word, settings_json=payload.settings)
    db.add(room)
    await db.flush()
    for position, character_id in enumerate(payload.character_ids):
        db.add(RoomMember(room_id=room.id, character_id=character_id, position=position))
    await db.commit()
    return await get_room(db, room.id)  # type: ignore[return-value]


async def update_room(db: AsyncSession, room: Room, payload: RoomUpdate) -> Room:
    await _validate_character_ids(db, payload.character_ids)
    room.name = payload.name
    room.safe_word = payload.safe_word
    room.settings_json = payload.settings
    await db.execute(delete(RoomMember).where(RoomMember.room_id == room.id))
    await db.flush()
    for position, character_id in enumerate(payload.character_ids):
        db.add(RoomMember(room_id=room.id, character_id=character_id, position=position))
    await db.commit()
    return await get_room(db, room.id)  # type: ignore[return-value]


async def delete_room(db: AsyncSession, room: Room) -> None:
    await db.delete(room)
    await db.commit()


async def create_or_resume_session(
    db: AsyncSession, room_id: int, resume_session_id: int | None = None
) -> ConversationSession:
    if resume_session_id is not None:
        item = await db.get(ConversationSession, resume_session_id)
        if item is None or item.room_id != room_id:
            raise ValueError("Session does not belong to this room")
        if item.status == "stopped":
            raise ValueError("Stopped sessions cannot be resumed")
        item.status = "paused"
    else:
        item = ConversationSession(room_id=room_id, status="paused")
        db.add(item)
    await db.commit()
    await db.refresh(item)
    return item


async def get_session(db: AsyncSession, session_id: int) -> ConversationSession | None:
    return await db.get(ConversationSession, session_id)


async def set_session_status(db: AsyncSession, session_id: int, status: str) -> None:
    item = await db.get(ConversationSession, session_id)
    if item is None:
        return
    item.status = status
    if status == "stopped":
        item.ended_at = datetime.now(timezone.utc)
    await db.commit()


async def list_lines(db: AsyncSession, session_id: int) -> list[Line]:
    statement = select(Line).where(Line.session_id == session_id).order_by(Line.sequence)
    return list((await db.scalars(statement)).all())


async def append_line(
    db: AsyncSession,
    *,
    session_id: int,
    speaker: str,
    text: str,
    character_id: int | None = None,
    bid_json: dict[str, Any] | None = None,
    audio_path: str | None = None,
) -> Line:
    sequence = int(
        (await db.scalar(select(func.coalesce(func.max(Line.sequence), 0)).where(Line.session_id == session_id)))
        or 0
    ) + 1
    item = Line(
        session_id=session_id,
        sequence=sequence,
        speaker=speaker,
        character_id=character_id,
        text=text,
        bid_json=bid_json,
        audio_path=audio_path,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return item


async def update_summary(db: AsyncSession, session_id: int, summary: str) -> None:
    item = await db.get(ConversationSession, session_id)
    if item is not None:
        item.summary = summary
        await db.commit()
