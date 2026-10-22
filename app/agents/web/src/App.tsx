import { useEffect, useState } from "react";
import { AudioLines, Bot, DoorOpen, Radio, UsersRound } from "lucide-react";
import { api } from "./api";
import CharactersView from "./views/CharactersView";
import RoomsView from "./views/RoomsView";
import LiveRoomView from "./views/LiveRoomView";
import type { Room, Session } from "./types";

type View = "characters" | "rooms" | "live";

interface Health {
  status: string;
  voices: { count: number; errors: string[]; tts_dependency: boolean };
  llm: { configured: boolean; model: string };
}

export default function App() {
  const [view, setView] = useState<View>("characters");
  const [live, setLive] = useState<{ room: Room; session: Session } | null>(null);
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api<Health>("api/health").then(setHealth).catch(() => setHealth(null));
  }, []);

  const enterRoom = (room: Room, session: Session) => {
    setLive({ room, session });
    setView("live");
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><AudioLines size={22} /></div>
          <div><strong>Voice Room</strong><span>agent studio</span></div>
        </div>
        <nav>
          <button className={view === "characters" ? "active" : ""} onClick={() => setView("characters")}>
            <Bot size={18} /> Characters
          </button>
          <button className={view === "rooms" ? "active" : ""} onClick={() => setView("rooms")}>
            <UsersRound size={18} /> Rooms
          </button>
          <button className={view === "live" ? "active" : ""} disabled={!live} onClick={() => live && setView("live")}>
            <Radio size={18} /> Live room
          </button>
        </nav>
        <div className="system-card">
          <div className="system-row"><span className={`status-dot ${health?.llm.configured ? "ok" : "warn"}`} /> AI</div>
          <small>{health?.llm.configured ? health.llm.model : "Add one in Settings → AI connections"}</small>
          <div className="system-row"><span className={`status-dot ${health?.voices.count ? "ok" : "warn"}`} /> Voices</div>
          <small>{health ? `${health.voices.count} discovered` : "Backend unavailable"}</small>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div>
            <span className="eyebrow">VOICE ROOM</span>
            <h1>{view === "characters" ? "Cast" : view === "rooms" ? "Rooms" : live?.room.name ?? "Live room"}</h1>
          </div>
          {view !== "rooms" && <button className="ghost" onClick={() => setView("rooms")}><DoorOpen size={17} /> Browse rooms</button>}
        </header>
        <section className="view-frame">
          {view === "characters" && <CharactersView />}
          {view === "rooms" && <RoomsView onEnter={enterRoom} />}
          {view === "live" && live && <LiveRoomView room={live.room} session={live.session} onExit={() => setView("rooms")} />}
        </section>
      </main>
    </div>
  );
}
