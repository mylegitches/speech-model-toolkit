"""Initial Voice Agent Room schema."""

from typing import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "characters",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("voice_path", sa.Text(), nullable=False),
        sa.Column("voice_config_path", sa.Text(), nullable=False),
        sa.Column("speaker_id", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("length_scale", sa.Float(), nullable=False, server_default="1"),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("eagerness", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("max_words", sa.Integer(), nullable=False, server_default="80"),
        sa.Column("color", sa.String(16), nullable=False, server_default="#7c5cff"),
        sa.Column("avatar", sa.Text(), nullable=True),
    )
    op.create_table(
        "rooms",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("safe_word", sa.String(120), nullable=True),
        sa.Column("settings_json", sa.JSON(), nullable=False),
    )
    op.create_table(
        "room_members",
        sa.Column("room_id", sa.Integer(), sa.ForeignKey("rooms.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("character_id", sa.Integer(), sa.ForeignKey("characters.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.UniqueConstraint("room_id", "position"),
    )
    op.create_table(
        "sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("room_id", sa.Integer(), sa.ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_sessions_room_id", "sessions", ["room_id"])
    op.create_table(
        "lines",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("speaker", sa.String(120), nullable=False),
        sa.Column("character_id", sa.Integer(), sa.ForeignKey("characters.id", ondelete="SET NULL"), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("audio_path", sa.Text(), nullable=True),
        sa.Column("bid_json", sa.JSON(), nullable=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("session_id", "sequence"),
    )
    op.create_index("ix_lines_session_id", "lines", ["session_id"])


def downgrade() -> None:
    op.drop_table("lines")
    op.drop_table("sessions")
    op.drop_table("room_members")
    op.drop_table("rooms")
    op.drop_table("characters")
