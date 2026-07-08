#!/usr/bin/env python3
"""
CSV AI Agent — FastAPI service
───────────────────────────────
Run with:
    uvicorn main:app --reload

Endpoints:
    POST   /sessions                     -> create a new session
    POST   /chat                         -> {session_id, question} -> {answer}
    GET    /sessions/{session_id}/history-> full stored history for a session
    DELETE /sessions/{session_id}        -> delete a session and its turns
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import CSV_PATH, TOP_K_TURNS
import db
from agent import build_crew

app = FastAPI(title="CSV AI Agent")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Serves everything under /static (not strictly needed for a single-file
# frontend, but keeps room to add CSS/JS/assets later without code changes).
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.on_event("startup")
def on_startup() -> None:
    if not os.path.exists(CSV_PATH):
        raise RuntimeError(
            f"CSV file not found: {CSV_PATH}. "
            "Set the CSV_PATH environment variable or update config.py"
        )
    db.init_db()


# --- Request/response models ---

class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str


class TurnOut(BaseModel):
    role: str
    content: str
    created_at: str


# --- Endpoints ---

@app.post("/sessions", response_model=SessionResponse)
def create_session():
    session_id = db.create_session()
    return SessionResponse(session_id=session_id)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    if not db.session_exists(req.session_id):
        raise HTTPException(
            status_code=404,
            detail=f"session_id '{req.session_id}' does not exist. "
            "Create one via POST /sessions first.",
        )

    recent_turns = db.get_recent_turns(req.session_id, k=TOP_K_TURNS)

    try:
        crew = build_crew(req.question, recent_turns)
        result = crew.kickoff()
        answer = str(result).strip()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    db.add_turn(req.session_id, "user", req.question)
    db.add_turn(req.session_id, "assistant", answer)

    return ChatResponse(answer=answer)


@app.get("/sessions/{session_id}/history", response_model=list[TurnOut])
def get_history(session_id: str):
    if not db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return db.get_all_history(session_id)


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    if not db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    db.delete_session(session_id)
    return {"detail": "session deleted"}
