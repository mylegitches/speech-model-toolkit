import { FormEvent, useEffect, useState } from "react";
import { ArrowDown, ArrowUp, DoorOpen, Pencil, Plus, Save, Shield, Trash2, UsersRound, X } from "lucide-react";
import { api } from "../api";
import type { Character, Room, RoomDraft, Session } from "../types";

const blankRoom = (): RoomDraft => ({
  name: "", safe_word: "", character_ids: [],
  settings: { max_context_lines: 30, summary_interval: 20, bid_threshold: 3, interjection_mode: "cut" },
});

export default function RoomsView({ onEnter }: { onEnter: (room: Room, session: Session) => void }) {
  const [rooms, setRooms] = useState<Room[]>([]);
  const [characters, setCharacters] = useState<Character[]>([]);
  const [editing, setEditing] = useState<Room | null | "new">(null);
  const [draft, setDraft] = useState<RoomDraft>(blankRoom());
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const load = async () => {
    const [roomRows, characterRows] = await Promise.all([api<Room[]>("api/rooms"), api<Character[]>("api/characters")]);
    setRooms(roomRows); setCharacters(characterRows);
  };
  useEffect(() => { load().catch((e) => setError(e.message)); }, []);

  const beginNew = () => { setDraft(blankRoom()); setEditing("new"); setError(""); };
  const beginEdit = (room: Room) => {
    setDraft({ name: room.name, safe_word: room.safe_word, settings: room.settings, character_ids: room.character_ids });
    setEditing(room); setError("");
  };
  const toggleCharacter = (id: number) => setDraft((current) => ({ ...current, character_ids: current.character_ids.includes(id) ? current.character_ids.filter((value) => value !== id) : [...current.character_ids, id] }));
  const move = (id: number, direction: -1 | 1) => setDraft((current) => {
    const ids = [...current.character_ids]; const from = ids.indexOf(id); const to = from + direction;
    if (from < 0 || to < 0 || to >= ids.length) return current;
    [ids[from], ids[to]] = [ids[to], ids[from]]; return { ...current, character_ids: ids };
  });

  const save = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError("");
    try {
      const path = editing === "new" ? "api/rooms" : `api/rooms/${editing!.id}`;
      const method = editing === "new" ? "POST" : "PUT";
      await api(path, { method, body: JSON.stringify({ ...draft, safe_word: draft.safe_word || null }) });
      await load(); setEditing(null);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  const remove = async (room: Room) => {
    if (!confirm(`Delete ${room.name} and its saved sessions?`)) return;
    try { await api(`api/rooms/${room.id}`, { method: "DELETE" }); await load(); }
    catch (e) { setError((e as Error).message); }
  };

  const enter = async (room: Room) => {
    setBusy(true); setError("");
    try {
      const session = await api<Session>(`api/rooms/${room.id}/session`, { method: "POST", body: JSON.stringify({}) });
      onEnter(room, session);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  return <div>
    <div className="section-intro"><div><h2>Set the table</h2><p>Choose a cast, tune the room, and let the conversation find its own shape.</p></div><button className="primary" onClick={beginNew}><Plus size={17} /> New room</button></div>
    {error && <div className="alert">{error}<button onClick={() => setError("")}><X size={16} /></button></div>}
    <div className="room-grid">
      {rooms.map((room) => <article className="room-card" key={room.id}>
        <div className="room-card-head"><div className="room-icon"><UsersRound /></div><div><h3>{room.name}</h3><span>{room.characters.length} voices</span></div></div>
        <div className="cast-stack">{room.characters.map((character) => <div key={character.id} className="cast-chip"><i style={{ background: character.color }}>{character.name.slice(0, 1)}</i>{character.name}</div>)}</div>
        <div className="room-meta"><span><Shield size={14} /> {room.safe_word || "No safe word"}</span><span>{String(room.settings.max_context_lines ?? 30)} lines in context</span></div>
        <div className="room-actions"><button className="primary" disabled={busy || room.characters.length < 2} onClick={() => enter(room)}><DoorOpen size={17} /> Enter</button><button className="icon-button" onClick={() => beginEdit(room)}><Pencil size={17} /></button><button className="icon-button danger" onClick={() => remove(room)}><Trash2 size={17} /></button></div>
      </article>)}
      {!rooms.length && <div className="empty-state"><div className="empty-orb"><UsersRound /></div><h3>No rooms yet</h3><p>Create two characters, then bring them into the same room.</p><button className="primary" onClick={beginNew}>Create a room</button></div>}
    </div>

    {editing && <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && setEditing(null)}><form className="modal" onSubmit={save}>
      <div className="modal-head"><div><span className="eyebrow">ROOM BUILDER</span><h2>{editing === "new" ? "New room" : `Edit ${editing.name}`}</h2></div><button type="button" className="icon-button" onClick={() => setEditing(null)}><X /></button></div>
      <div className="form-grid"><label><span>Room name</span><input required value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} placeholder="Late night table" /></label><label><span>Safe word <em>optional</em></span><input value={draft.safe_word ?? ""} onChange={(e) => setDraft({ ...draft, safe_word: e.target.value })} placeholder="That’s a wrap" /></label></div>
      <div className="field-title"><span>Cast order</span><small>At least two characters are required to enter.</small></div>
      <div className="member-picker">{characters.map((character) => {
        const selected = draft.character_ids.includes(character.id); const position = draft.character_ids.indexOf(character.id);
        return <div key={character.id} className={`member-row ${selected ? "selected" : ""}`}><button type="button" className="member-toggle" onClick={() => toggleCharacter(character.id)}><i style={{ background: character.color }}>{character.name.slice(0, 1)}</i><span><strong>{character.name}</strong><small>{character.voice_path.split("/").pop()?.replace(".onnx", "")}</small></span><b>{selected ? position + 1 : "+"}</b></button>{selected && <div className="order-buttons"><button type="button" disabled={position === 0} onClick={() => move(character.id, -1)}><ArrowUp size={15} /></button><button type="button" disabled={position === draft.character_ids.length - 1} onClick={() => move(character.id, 1)}><ArrowDown size={15} /></button></div>}</div>;
      })}{!characters.length && <p className="subtle">Create characters before building a room.</p>}</div>
      <details className="advanced"><summary>Conversation settings</summary><div className="form-grid compact"><label><span>Context lines</span><input type="number" min="5" max="200" value={Number(draft.settings.max_context_lines ?? 30)} onChange={(e) => setDraft({ ...draft, settings: { ...draft.settings, max_context_lines: Number(e.target.value) } })} /></label><label><span>Summary interval</span><input type="number" min="5" max="100" value={Number(draft.settings.summary_interval ?? 20)} onChange={(e) => setDraft({ ...draft, settings: { ...draft.settings, summary_interval: Number(e.target.value) } })} /></label></div></details>
      <div className="modal-actions"><button type="button" className="ghost" onClick={() => setEditing(null)}>Cancel</button><button className="primary" disabled={busy}><Save size={17} /> {busy ? "Saving…" : "Save room"}</button></div>
    </form></div>}
  </div>;
}
