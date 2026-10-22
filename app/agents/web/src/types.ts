export interface VoiceInfo {
  id: string;
  name: string;
  model_path: string;
  config_path: string;
  sample_rate: number | null;
  num_speakers: number;
  language: string;
  length_scale: number;
  status: string;
  error: string | null;
}

export interface Character {
  id: number;
  name: string;
  voice_path: string;
  voice_config_path: string;
  speaker_id: number;
  length_scale: number;
  system_prompt: string;
  eagerness: number;
  max_words: number;
  color: string;
  avatar: string | null;
}

export type CharacterDraft = Omit<Character, "id">;

export interface Room {
  id: number;
  name: string;
  safe_word: string | null;
  settings: Record<string, unknown>;
  character_ids: number[];
  characters: Character[];
}

export interface RoomDraft {
  name: string;
  safe_word: string | null;
  settings: Record<string, unknown>;
  character_ids: number[];
}

export interface Session {
  id: number;
  room_id: number;
  status: string;
  summary: string;
  started_at: string;
  ended_at: string | null;
}

export interface TranscriptLine {
  id: number;
  sequence: number;
  speaker: string;
  character_id: number | null;
  text: string;
  ts: string;
}

export interface BidRow {
  character_id: number;
  name: string;
  raw_bid: number;
  score: number;
}
