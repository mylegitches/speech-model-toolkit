from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Character(Base):
    __tablename__ = "characters"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    voice_path: Mapped[str] = mapped_column(Text, nullable=False)
    voice_config_path: Mapped[str] = mapped_column(Text, nullable=False)
    speaker_id: Mapped[int] = mapped_column(Integer, default=0)
    length_scale: Mapped[float] = mapped_column(Float, default=1.0)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    eagerness: Mapped[float] = mapped_column(Float, default=0.5)
    max_words: Mapped[int] = mapped_column(Integer, default=80)
    color: Mapped[str] = mapped_column(String(16), default="#7c5cff")
    avatar: Mapped[str | None] = mapped_column(Text, nullable=True)

    memberships: Mapped[list[RoomMember]] = relationship(
        back_populates="character", cascade="all, delete-orphan"
    )


class Room(Base):
    __tablename__ = "rooms"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    safe_word: Mapped[str | None] = mapped_column(String(120), nullable=True)
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    members: Mapped[list[RoomMember]] = relationship(
        back_populates="room",
        cascade="all, delete-orphan",
        order_by="RoomMember.position",
    )
    sessions: Mapped[list[ConversationSession]] = relationship(
        back_populates="room", cascade="all, delete-orphan"
    )


class RoomMember(Base):
    __tablename__ = "room_members"
    __table_args__ = (UniqueConstraint("room_id", "position"),)

    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id", ondelete="CASCADE"), primary_key=True)
    character_id: Mapped[int] = mapped_column(
        ForeignKey("characters.id", ondelete="CASCADE"), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    room: Mapped[Room] = relationship(back_populates="members")
    character: Mapped[Character] = relationship(back_populates="memberships")


class ConversationSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="paused")
    summary: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    room: Mapped[Room] = relationship(back_populates="sessions")
    lines: Mapped[list[Line]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Line.sequence"
    )


class Line(Base):
    __tablename__ = "lines"
    __table_args__ = (UniqueConstraint("session_id", "sequence"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    speaker: Mapped[str] = mapped_column(String(120), nullable=False)
    character_id: Mapped[int | None] = mapped_column(
        ForeignKey("characters.id", ondelete="SET NULL"), nullable=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    audio_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    bid_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped[ConversationSession] = relationship(back_populates="lines")
