import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, Bug, Download, Mic, Pause, Play, Send, Square, Volume2, X } from "lucide-react";
import { uid } from "../api";
import type { BidRow, Room, Session, TranscriptLine } from "../types";

type RoomState = "connecting" | "paused" | "pause_requested" | "playing" | "stopped";
interface DraftLine { turnId: string; speaker: string; characterId: number; text: string; color: string }
interface AudioItem { url: string; lineId: number }

export default function LiveRoomView({ room, session, onExit }: { room: Room; session: Session; onExit: () => void }) {
  const [state, setState] = useState<RoomState>("connecting");
  const [lines, setLines] = useState<TranscriptLine[]>([]);
  const [draftLine, setDraftLine] = useState<DraftLine | null>(null);
  const [topic, setTopic] = useState("");
  const [interjection, setInterjection] = useState("");
  const [error, setError] = useState("");
  const [connected, setConnected] = useState(false);
  const [recording, setRecording] = useState(false);
  const [micState, setMicState] = useState("idle");
  const [bids, setBids] = useState<BidRow[]>([]);
  const [showDebug, setShowDebug] = useState(false);
  const transcriptEnd = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<number | null>(null);
  const shouldReconnect = useRef(true);
  const stateRef = useRef<RoomState>("connecting");
  const pendingAudio = useRef<{ lineId: number } | null>(null);
  const audioQueue = useRef<AudioItem[]>([]);
  const activeAudio = useRef<HTMLAudioElement | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const micHeld = useRef(false);

  const send = useCallback((type: string, values: Record<string, unknown> = {}) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(JSON.stringify({ type, command_id: uid(), ...values }));
  }, []);

  const playNext = useCallback(() => {
    if (activeAudio.current || !audioQueue.current.length) return;
    const item = audioQueue.current.shift()!;
    const audio = new Audio(item.url);
    activeAudio.current = audio;
    const complete = (kind: "playback_finished" | "playback_cancelled") => {
      URL.revokeObjectURL(item.url);
      if (activeAudio.current === audio) activeAudio.current = null;
      send(kind, { line_id: item.lineId });
      playNext();
    };
    audio.onended = () => complete("playback_finished");
    audio.onerror = () => complete("playback_cancelled");
    audio.play().catch(() => complete("playback_cancelled"));
  }, [send]);

  const cancelAudio = useCallback(() => {
    if (activeAudio.current) {
      activeAudio.current.pause();
      activeAudio.current.src = "";
      activeAudio.current = null;
    }
    for (const item of audioQueue.current) URL.revokeObjectURL(item.url);
    audioQueue.current = [];
    pendingAudio.current = null;
  }, []);

  useEffect(() => { stateRef.current = state; }, [state]);

  useEffect(() => {
    shouldReconnect.current = true;
    const connect = () => {
      const protocol = location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocket(`${protocol}://${location.host}${location.pathname.replace(/[^/]*$/, "")}ws/rooms/${room.id}?session_id=${session.id}`);
      socket.binaryType = "blob";
      wsRef.current = socket;
      socket.onopen = () => { setConnected(true); setError(""); };
      socket.onmessage = (event) => {
        if (event.data instanceof Blob) {
          const meta = pendingAudio.current;
          if (meta) {
            audioQueue.current.push({ url: URL.createObjectURL(event.data), lineId: meta.lineId });
            pendingAudio.current = null;
            playNext();
          }
          return;
        }
        const message = JSON.parse(event.data);
        switch (message.type) {
          case "snapshot": setLines(message.lines); setState(message.state); break;
          case "state": setState(message.state); break;
          case "line_start": setDraftLine({ turnId: message.turn_id, speaker: message.speaker, characterId: message.character_id, color: message.color, text: "" }); break;
          case "line_delta": setDraftLine((current) => {
            if (!current || current.turnId !== message.turn_id) return current;
            return { ...current, text: current.text + message.text };
          }); break;
          case "line_final":
            setLines((current) => current.some((line) => line.id === message.line.id) ? current : [...current, message.line]);
            setDraftLine((current) => current?.turnId === message.turn_id ? null : current);
            break;
          case "audio_start": pendingAudio.current = { lineId: message.line_id }; break;
          case "cancel_audio": cancelAudio(); setDraftLine(null); break;
          case "bid": setBids(message.candidates); break;
          case "mic_state": setMicState(message.state); break;
          case "transcription": setInterjection(message.text); break;
          case "error": setError(message.message); break;
        }
      };
      socket.onclose = (event) => {
        setConnected(false);
        // 4404: room/session gone or already stopped; retrying cannot succeed.
        if (event.code === 4404) shouldReconnect.current = false;
        if (shouldReconnect.current && stateRef.current !== "stopped") reconnectTimer.current = window.setTimeout(connect, 1500);
      };
      socket.onerror = () => setError("The live-room connection failed.");
    };
    connect();
    return () => {
      shouldReconnect.current = false;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close(); cancelAudio();
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    };
    // A live room is intentionally bound to its initial room/session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [room.id, session.id]);

  useEffect(() => { transcriptEnd.current?.scrollIntoView({ behavior: "smooth" }); }, [lines, draftLine]);

  const submitInterjection = (event: FormEvent) => {
    event.preventDefault(); const text = interjection.trim(); if (!text) return;
    send("interject_text", { text }); setInterjection("");
  };

  const startMic = async () => {
    if (recording || state === "stopped") return;
    micHeld.current = true;
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      mediaStreamRef.current = stream;
      const preferred = "audio/webm;codecs=opus";
      const mimeType = MediaRecorder.isTypeSupported(preferred) ? preferred : "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType }); recorderRef.current = recorder;
      send("mic_start", { mime_type: mimeType });
      recorder.ondataavailable = (event) => { if (event.data.size && wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(event.data); };
      recorder.onstop = () => { send("mic_end"); stream.getTracks().forEach((track) => track.stop()); setRecording(false); };
      recorder.start(200); setRecording(true);
      if (!micHeld.current) recorder.stop();
    } catch (e) { setError(`Microphone unavailable: ${(e as Error).message}`); }
  };

  const stopMic = () => {
    micHeld.current = false;
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
  };
  const colorFor = (characterId: number | null) => room.characters.find((character) => character.id === characterId)?.color ?? "#b8b4a8";
  const isStarted = lines.length > 0;

  return <div className="live-layout">
    <section className="live-stage">
      <div className="live-toolbar">
        <button className="ghost" onClick={onExit}><ArrowLeft size={17} /> Rooms</button>
        <div className={`live-status ${state}`}><span />{state.replace("_", " ")}{!connected && " · reconnecting"}</div>
        <div className="toolbar-actions"><button className="icon-button" title="Bid debugger" onClick={() => setShowDebug(!showDebug)}><Bug size={17} /></button><a className="icon-button" title="Download transcript" href={`api/sessions/${session.id}/transcript.txt`} download={`${room.name}-transcript.txt`}><Download size={17} /></a></div>
      </div>
      <div className="room-presence">{room.characters.map((character) => <div className="presence" key={character.id}><div className="presence-avatar" style={{ background: character.color }}>{character.name.slice(0, 1)}</div><span>{character.name}</span></div>)}</div>
      <div className="transcript">
        {!lines.length && !draftLine && <div className="kickoff"><span className="eyebrow">THE ROOM IS LISTENING</span><h2>Give them something to talk about.</h2><p>Type a starting point below, or hold the microphone and say it out loud.</p></div>}
        {lines.map((line) => <article className={`line ${line.character_id === null ? "you" : ""}`} key={line.id} style={{ "--speaker": colorFor(line.character_id) } as React.CSSProperties}><div className="line-speaker"><i />{line.speaker}</div><p>{line.text}</p></article>)}
        {draftLine && <article className="line speaking" style={{ "--speaker": draftLine.color } as React.CSSProperties}><div className="line-speaker"><i />{draftLine.speaker}<Volume2 size={14} /></div><p>{draftLine.text}<span className="cursor" /></p></article>}
        <div ref={transcriptEnd} />
      </div>
      {error && <div className="alert live-alert">{error}<button onClick={() => setError("")}><X size={16} /></button></div>}
      <div className="control-dock">
        {!isStarted && <input className="topic-input" value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="Kick off with a topic…" onKeyDown={(e) => { if (e.key === "Enter") { send("start", { topic }); setTopic(""); } }} />}
        {isStarted && <form className="interject-form" onSubmit={submitInterjection}><input value={interjection} onChange={(e) => setInterjection(e.target.value)} placeholder="Jump into the conversation…" /><button className="icon-button" aria-label="Send interjection"><Send size={18} /></button></form>}
        <div className="transport">
          {state !== "playing" && state !== "pause_requested" && state !== "stopped" && <button className="transport-button play" disabled={!connected} onClick={() => { send("start", { topic }); setTopic(""); }}><Play fill="currentColor" /></button>}
          {(state === "playing" || state === "pause_requested") && <button className="transport-button" onClick={() => send("pause")} disabled={state === "pause_requested"}><Pause fill="currentColor" /></button>}
          <button className={`transport-button mic ${recording ? "recording" : ""}`} disabled={!connected || state === "stopped" || micState === "transcribing"} onPointerDown={startMic} onPointerUp={stopMic} onPointerCancel={stopMic}><Mic /> <span>{recording ? "Release" : micState === "transcribing" ? "Thinking" : "Hold"}</span></button>
          <button className="transport-button stop" disabled={state === "stopped"} onClick={() => send("stop")}><Square fill="currentColor" /></button>
        </div>
      </div>
    </section>
    {showDebug && <aside className="debug-panel"><div className="debug-head"><div><span className="eyebrow">LAST AUCTION</span><h3>Who speaks next?</h3></div><button className="icon-button" onClick={() => setShowDebug(false)}><X size={17} /></button></div>{bids.length ? [...bids].sort((a, b) => b.score - a.score).map((bid, index) => <div className="bid-row" key={bid.character_id}><div><i>{index + 1}</i><span>{bid.name}<small>raw {bid.raw_bid.toFixed(1)}</small></span></div><strong>{bid.score.toFixed(1)}</strong><div className="bid-bar"><span style={{ width: `${Math.min(100, bid.score * 8)}%` }} /></div></div>) : <p className="subtle">Bid scores appear after the first turn.</p>}</aside>}
  </div>;
}
