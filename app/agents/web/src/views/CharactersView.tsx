import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { Headphones, Pencil, Plus, Save, Trash2, Upload, X } from "lucide-react";
import { api } from "../api";
import type { Character, CharacterDraft, VoiceInfo } from "../types";

const blankCharacter = (voice?: VoiceInfo): CharacterDraft => ({
  name: "",
  voice_path: voice?.model_path ?? "",
  voice_config_path: voice?.config_path ?? "",
  speaker_id: 0,
  length_scale: voice?.length_scale ?? 1,
  system_prompt: "",
  eagerness: 0.5,
  max_words: 80,
  color: "#7c5cff",
  avatar: null,
});

export default function CharactersView() {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [voices, setVoices] = useState<VoiceInfo[]>([]);
  const [editing, setEditing] = useState<Character | null | "new">(null);
  const [draft, setDraft] = useState<CharacterDraft>(blankCharacter());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const modelInput = useRef<HTMLInputElement>(null);
  const configInput = useRef<HTMLInputElement>(null);

  const load = async () => {
    const [characterRows, voiceRows] = await Promise.all([
      api<Character[]>("api/characters"),
      api<VoiceInfo[]>("api/voices"),
    ]);
    setCharacters(characterRows);
    setVoices(voiceRows);
    return voiceRows;
  };

  useEffect(() => { load().catch((e) => setError(e.message)); }, []);

  const selectedVoice = useMemo(
    () => voices.find((voice) => voice.model_path === draft.voice_path),
    [voices, draft.voice_path],
  );

  const beginNew = () => {
    setEditing("new");
    setDraft(blankCharacter(voices[0]));
    setError("");
  };

  const beginEdit = (character: Character) => {
    setEditing(character);
    const { id: _, ...values } = character;
    setDraft(values);
    setError("");
  };

  const chooseVoice = (path: string) => {
    const voice = voices.find((item) => item.model_path === path);
    if (!voice) return;
    setDraft((current) => ({
      ...current,
      voice_path: voice.model_path,
      voice_config_path: voice.config_path,
      length_scale: voice.length_scale,
      speaker_id: 0,
    }));
  };

  const save = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true); setError("");
    try {
      if (editing === "new") await api<Character>("api/characters", { method: "POST", body: JSON.stringify(draft) });
      else if (editing) await api<Character>(`api/characters/${editing.id}`, { method: "PUT", body: JSON.stringify(draft) });
      await load();
      setEditing(null);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  const remove = async (character: Character) => {
    if (!confirm(`Delete ${character.name}? Rooms using this character will lose that member.`)) return;
    try { await api(`api/characters/${character.id}`, { method: "DELETE" }); await load(); }
    catch (e) { setError((e as Error).message); }
  };

  const preview = async (character: Character) => {
    setBusy(true); setError("");
    try {
      const response = await fetch(`api/characters/${character.id}/preview`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: `Hello. I'm ${character.name}, and this is what my voice sounds like.` }),
      });
      if (!response.ok) throw new Error((await response.json()).detail ?? "Preview failed");
      const url = URL.createObjectURL(await response.blob());
      const audio = new Audio(url);
      audio.addEventListener("ended", () => URL.revokeObjectURL(url), { once: true });
      await audio.play();
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  const uploadPair = async () => {
    if (!editing || editing === "new") return;
    const model = modelInput.current?.files?.[0];
    const config = configInput.current?.files?.[0];
    if (!model || !config) { setError("Choose both the .onnx model and matching .onnx.json config."); return; }
    const data = new FormData(); data.append("model", model); data.append("config", config);
    setBusy(true); setError("");
    try {
      const updated = await api<Character>(`api/characters/${editing.id}/voice`, { method: "POST", body: data });
      await load(); beginEdit(updated);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  return (
    <div>
      <div className="section-intro">
        <div><h2>Build your cast</h2><p>Pair a local voice with a point of view. Characters can appear in any number of rooms.</p></div>
        <button className="primary" onClick={beginNew}><Plus size={17} /> New character</button>
      </div>
      {error && <div className="alert">{error}<button onClick={() => setError("")}><X size={16} /></button></div>}
      <div className="card-grid">
        {characters.map((character) => (
          <article className="character-card" key={character.id} style={{ "--accent": character.color } as React.CSSProperties}>
            <div className="avatar">{character.avatar ? <img src={character.avatar} alt="" /> : character.name.slice(0, 2).toUpperCase()}</div>
            <div className="card-copy"><h3>{character.name}</h3><span>{character.voice_path.split("/").pop()?.replace(".onnx", "")}</span><p>{character.system_prompt}</p></div>
            <div className="card-actions">
              <button className="icon-button" title="Preview voice" disabled={busy} onClick={() => preview(character)}><Headphones size={17} /></button>
              <button className="icon-button" title="Edit" onClick={() => beginEdit(character)}><Pencil size={17} /></button>
              <button className="icon-button danger" title="Delete" onClick={() => remove(character)}><Trash2 size={17} /></button>
            </div>
          </article>
        ))}
        {!characters.length && <div className="empty-state"><div className="empty-orb"><Headphones /></div><h3>No characters yet</h3><p>Your {voices.length} local voices are ready to become a cast.</p><button className="primary" onClick={beginNew}>Create the first one</button></div>}
      </div>

      {editing && <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && setEditing(null)}>
        <form className="modal" onSubmit={save}>
          <div className="modal-head"><div><span className="eyebrow">CHARACTER BUILDER</span><h2>{editing === "new" ? "New character" : `Edit ${editing.name}`}</h2></div><button type="button" className="icon-button" onClick={() => setEditing(null)}><X /></button></div>
          <div className="form-grid">
            <label><span>Name</span><input required value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} placeholder="The Instigator" /></label>
            <label><span>Voice</span><select required value={draft.voice_path} onChange={(e) => chooseVoice(e.target.value)}>{voices.map((voice) => <option key={voice.model_path} value={voice.model_path}>{voice.name} · {voice.language}</option>)}</select></label>
            <label className="wide"><span>Personality / system prompt</span><textarea required rows={6} value={draft.system_prompt} onChange={(e) => setDraft({ ...draft, system_prompt: e.target.value })} placeholder="You are curious, fast, direct, and willing to challenge assumptions..." /></label>
            <label><span>Eagerness <b>{draft.eagerness.toFixed(2)}</b></span><input type="range" min="0" max="1" step="0.05" value={draft.eagerness} onChange={(e) => setDraft({ ...draft, eagerness: Number(e.target.value) })} /></label>
            <label><span>Maximum words</span><input type="number" min="5" max="500" value={draft.max_words} onChange={(e) => setDraft({ ...draft, max_words: Number(e.target.value) })} /></label>
            <label><span>Speech speed <b>{draft.length_scale.toFixed(2)}</b></span><input type="range" min="0.5" max="2" step="0.05" value={draft.length_scale} onChange={(e) => setDraft({ ...draft, length_scale: Number(e.target.value) })} /></label>
            <label><span>Speaker ID</span><input type="number" min="0" max={(selectedVoice?.num_speakers ?? 1) - 1} value={draft.speaker_id} onChange={(e) => setDraft({ ...draft, speaker_id: Number(e.target.value) })} /></label>
            <label><span>Accent color</span><div className="color-field"><input type="color" value={draft.color} onChange={(e) => setDraft({ ...draft, color: e.target.value })} /><code>{draft.color}</code></div></label>
            <label><span>Avatar URL <em>optional</em></span><input value={draft.avatar ?? ""} onChange={(e) => setDraft({ ...draft, avatar: e.target.value || null })} placeholder="https://…" /></label>
          </div>
          {editing !== "new" && <div className="upload-strip"><Upload size={18} /><div><strong>Replace this character’s voice</strong><span>Choose an exact .onnx / .onnx.json pair.</span></div><input ref={modelInput} type="file" accept=".onnx" /><input ref={configInput} type="file" accept=".json" /><button type="button" className="secondary" onClick={uploadPair}>Upload</button></div>}
          <div className="modal-actions"><button type="button" className="ghost" onClick={() => setEditing(null)}>Cancel</button><button className="primary" disabled={busy || !voices.length}><Save size={17} /> {busy ? "Saving…" : "Save character"}</button></div>
        </form>
      </div>}
    </div>
  );
}
