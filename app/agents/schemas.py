from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CharacterBase(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    voice_path: str = Field(min_length=1)
    voice_config_path: str = Field(min_length=1)
    speaker_id: int = Field(default=0, ge=0)
    length_scale: float = Field(default=1.0, ge=0.25, le=4.0)
    system_prompt: str = Field(min_length=1, max_length=10_000)
    eagerness: float = Field(default=0.5, ge=0, le=1)
    max_words: int = Field(default=80, ge=5, le=500)
    color: str = Field(default="#7c5cff", pattern=r"^#[0-9a-fA-F]{6}$")
    avatar: str | None = None


class CharacterCreate(CharacterBase):
    pass


class CharacterUpdate(CharacterBase):
    pass


class CharacterRead(CharacterBase):
    model_config = ConfigDict(from_attributes=True)
    id: int


class PreviewRequest(BaseModel):
    text: str = Field(default="Hello. This is what my voice sounds like.", min_length=1, max_length=500)


class RoomBase(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    safe_word: str | None = Field(default=None, max_length=120)
    settings: dict[str, Any] = Field(default_factory=dict)
    character_ids: list[int] = Field(default_factory=list)

    @field_validator("character_ids")
    @classmethod
    def unique_members(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("A character can appear only once in a room")
        return value


class RoomCreate(RoomBase):
    pass


class RoomUpdate(RoomBase):
    pass


class RoomRead(RoomBase):
    id: int
    characters: list[CharacterRead] = Field(default_factory=list)


class SessionCreate(BaseModel):
    resume_session_id: int | None = None


class SessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    room_id: int
    status: str
    summary: str
    started_at: datetime
    ended_at: datetime | None


class LineRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    session_id: int
    sequence: int
    speaker: str
    character_id: int | None
    text: str
    audio_path: str | None
    bid_json: dict[str, Any] | None
    ts: datetime


class VoiceInfo(BaseModel):
    id: str
    name: str
    model_path: str
    config_path: str
    sample_rate: int | None
    num_speakers: int
    language: str
    length_scale: float
    status: str = "untested"
    error: str | None = None
