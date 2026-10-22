from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..db import get_db
from ..repositories import (
    create_character,
    delete_character,
    get_character,
    list_characters,
    update_character,
)
from ..schemas import CharacterCreate, CharacterRead, CharacterUpdate, PreviewRequest, VoiceInfo
from ..tts import tts_service
from ..voice_catalog import voice_catalog


router = APIRouter(prefix="/api/characters", tags=["characters"])


def _validate_voice(model_path: str, config_path: str, speaker_id: int) -> None:
    if voice_catalog.is_remote(model_path):
        try:
            voice = voice_catalog.resolve_remote(model_path, config_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if speaker_id >= max(1, len(voice.speakers)):
            raise HTTPException(422, f"speaker_id must be below {max(1, len(voice.speakers))}")
        return
    try:
        model, config = voice_catalog.resolve_pair(model_path, config_path)
        metadata = json.loads(config.read_text(encoding="utf-8"))
        num_speakers = max(1, int(metadata.get("num_speakers", 1) or 1))
        if speaker_id >= num_speakers:
            raise ValueError(f"speaker_id must be below {num_speakers}")
        if model.stat().st_size == 0:
            raise ValueError("Voice model is empty")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("", response_model=list[CharacterRead])
async def get_all(db: AsyncSession = Depends(get_db)):
    return await list_characters(db)


@router.post("", response_model=CharacterRead, status_code=status.HTTP_201_CREATED)
async def create(payload: CharacterCreate, db: AsyncSession = Depends(get_db)):
    await asyncio.to_thread(_validate_voice, payload.voice_path, payload.voice_config_path, payload.speaker_id)
    return await create_character(db, payload)


@router.get("/{character_id}", response_model=CharacterRead)
async def get_one(character_id: int, db: AsyncSession = Depends(get_db)):
    item = await get_character(db, character_id)
    if item is None:
        raise HTTPException(404, "Character not found")
    return item


@router.put("/{character_id}", response_model=CharacterRead)
async def update(character_id: int, payload: CharacterUpdate, db: AsyncSession = Depends(get_db)):
    item = await get_character(db, character_id)
    if item is None:
        raise HTTPException(404, "Character not found")
    await asyncio.to_thread(_validate_voice, payload.voice_path, payload.voice_config_path, payload.speaker_id)
    return await update_character(db, item, payload)


@router.delete("/{character_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove(character_id: int, db: AsyncSession = Depends(get_db)):
    item = await get_character(db, character_id)
    if item is None:
        raise HTTPException(404, "Character not found")
    await delete_character(db, item)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{character_id}/preview")
async def preview(
    character_id: int, payload: PreviewRequest, db: AsyncSession = Depends(get_db)
):
    item = await get_character(db, character_id)
    if item is None:
        raise HTTPException(404, "Character not found")
    try:
        audio = await tts_service.synthesize(
            payload.text,
            item.voice_path,
            item.voice_config_path,
            item.speaker_id,
            item.length_scale,
        )
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc
    return Response(audio, media_type="audio/wav")


@router.post("/{character_id}/voice", response_model=CharacterRead)
async def upload_voice(
    character_id: int,
    model: UploadFile = File(...),
    config: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    item = await get_character(db, character_id)
    if item is None:
        raise HTTPException(404, "Character not found")
    model_name = Path(model.filename or "").name
    config_name = Path(config.filename or "").name
    if not model_name.endswith(".onnx") or config_name != f"{model_name}.json":
        raise HTTPException(422, "Upload a matching .onnx and .onnx.json pair")
    safe_stem = re.sub(r"[^a-zA-Z0-9_.-]+", "-", Path(model_name).stem).strip(".-")
    if not safe_stem:
        raise HTTPException(422, "Invalid model filename")
    model_data = await model.read(300 * 1024 * 1024 + 1)
    config_data = await config.read(2 * 1024 * 1024 + 1)
    if len(model_data) > 300 * 1024 * 1024 or len(config_data) > 2 * 1024 * 1024:
        raise HTTPException(413, "Voice upload is too large")
    try:
        json.loads(config_data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(422, "Voice config is not valid JSON") from exc

    target_dir = get_settings().uploads_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_model = target_dir / f"{safe_stem}.onnx"
    target_config = target_dir / f"{safe_stem}.onnx.json"
    with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as temp_model:
        temp_model.write(model_data)
        temp_model_path = Path(temp_model.name)
    with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as temp_config:
        temp_config.write(config_data)
        temp_config_path = Path(temp_config.name)
    os.replace(temp_model_path, target_model)
    os.replace(temp_config_path, target_config)
    item.voice_path = target_model.as_posix()
    item.voice_config_path = target_config.as_posix()
    await db.commit()
    await db.refresh(item)
    return item


voices_router = APIRouter(prefix="/api/voices", tags=["voices"])


@voices_router.get("", response_model=list[VoiceInfo])
async def voices():
    return await asyncio.to_thread(voice_catalog.scan)
