from __future__ import annotations

import json

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from .session_manager import session_manager


router = APIRouter()


@router.websocket("/ws/rooms/{room_id}")
async def room_socket(websocket: WebSocket, room_id: int, session_id: int = Query(...)):
    await websocket.accept()
    try:
        active = await session_manager.connect(room_id, session_id, websocket)
    except ValueError as exc:
        await websocket.send_json({"type": "error", "code": "not_found", "message": str(exc)})
        await websocket.close(code=4404)
        return
    except RuntimeError as exc:
        await websocket.send_json({"type": "error", "code": "room_busy", "message": str(exc)})
        await websocket.close(code=4409)
        return

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await session_manager.handle_binary(active, message["bytes"])
            elif message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "code": "invalid_json", "message": "Invalid JSON"}
                    )
                    continue
                await session_manager.handle_json(active, payload)
    except WebSocketDisconnect:
        pass
    finally:
        await session_manager.disconnect(active)
