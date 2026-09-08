import asyncio
import fcntl
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import json
import re
import secrets
import time
from typing import Literal
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import dateparser
from fastapi import FastAPI, Depends, HTTPException, Request, Response, Header
from fastapi.responses import JSONResponse
import httpx
import jwt
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from sqlalchemy import select, update, delete, func

from .config import settings
from .database import (
    engine,
    transaction,
    check_migrations,
    users,
    sessions,
    attempts,
    entries,
    audits,
    examples,
    rounds,
    record,
    push_subscriptions,
)
from .security import (
    encrypt,
    decrypt,
    password_hash,
    verify_password,
    keyed,
    DUMMY_HASH,
)
from .preferences import DEFAULTS, prefs, expenditure, affordable
from .local_model import classify
from . import summarization
from . import notifications, assistant_actions, snips
from fl import pipeline
from fl.privacy import EPSILON_PER_ROUND, DELTA_PER_ROUND, DELTA_CAP, CLIP_NORM

COOKIE = "ppda_session"
PREFIX = "/api/v1"


async def reminder_loop():
    # One-second due checks; Web Push remains best effort, not an exact OS alarm.
    while True:
        try:
            await asyncio.to_thread(notifications.enqueue_due)
        except Exception:
            import logging

            logging.getLogger("ppda").error(
                "Notification worker failed; retry deferred"
            )
        await asyncio.sleep(1)


async def push_delivery_loop():
    while True:
        try:
            await asyncio.to_thread(notifications.deliver_batch)
        except Exception:
            import logging

            logging.getLogger("ppda").error(
                "Push delivery worker failed; retry deferred"
            )
        await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app):
    # One supervisor per database; reject concurrent startup instead of corrupting recovery.
    lock = open(str(engine.url.database) + ".lock", "a")
    task = None
    push_task = None
    started = False
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(
            "PPDA supports one backend process per database; do not use multiple Uvicorn workers."
        )
    try:
        check_migrations()
        pipeline.initialize_model()
        from .onnx_model import probabilities

        with engine.connect() as conn:
            _, weights = pipeline.active_model(conn)
        probabilities(weights, "hello")
        if settings.pipeline_enabled:
            pipeline.start()
            started = True
        task = asyncio.create_task(reminder_loop())
        push_task = asyncio.create_task(push_delivery_loop())
        yield
    finally:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if push_task:
            push_task.cancel()
            try:
                await push_task
            except asyncio.CancelledError:
                pass
        if started:
            pipeline.shutdown()
        lock.close()


app = FastAPI(title="PPDA local-first assistant", lifespan=lifespan)
from .body_limit import BodyLimit

app.add_middleware(BodyLimit)


@app.middleware("http")
async def boundary(request: Request, call_next):
    length = request.headers.get("content-length", "0")
    if not length.isdigit() or int(length) > 100_000:
        return JSONResponse({"detail": "Request too large"}, status_code=413)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        # Same-origin UI; forwarded Host is preserved by the Angular proxy.
        if origin and (
            urlparse(origin).netloc != request.headers.get("host")
            or urlparse(origin).scheme not in ("http", "https")
        ):
            return JSONResponse(
                {"detail": "Cross-origin request rejected"}, status_code=403
            )
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.exception_handler(Exception)
async def failure(request, exc):
    # Do not echo secrets, ciphertexts, prompts or provider response bodies.
    return JSONResponse(
        {"detail": "Operation failed safely. Please retry or contact the operator."},
        status_code=500,
    )


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Login(Strict):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class Register(Login):
    name: str = Field(min_length=1, max_length=40)
    local_consent: bool = False


class Preferences(Strict):
    assistant: bool
    calendar: bool
    notes: bool
    summary: bool
    training: bool
    cloud: bool
    epsilon: float = Field(ge=1, le=10, allow_inf_nan=False)
    reduced: bool
    timezone: str = "Asia/Kolkata"
    mode: Literal["Default", "Privacy", "Global"]

    @field_validator("timezone")
    @classmethod
    def timezone_exists(cls, value):
        try:
            ZoneInfo(value)
        except Exception:
            raise ValueError("Use a valid IANA timezone")
        return value


class Profile(Strict):
    name: str = Field(min_length=1, max_length=40)


class PasswordChange(Strict):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=72)


class Entry(Strict):
    title: str = Field(min_length=1, max_length=160)
    detail: str = Field(default="", max_length=20000)
    done: bool = False
    learning_text: str | None = Field(default=None, max_length=10000)
    expected_version: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @field_validator("title")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Title is required")
        return value.strip()


class Command(Strict):
    text: str = Field(min_length=1, max_length=10000)
    mode: Literal["Default", "Privacy", "Global"] = "Default"


class Example(Strict):
    text: str = Field(min_length=1, max_length=10000)
    label: Literal["calendar", "reminder", "note", "summary", "chat"]


def current_user(request: Request):
    token = request.cookies.get(COOKIE)
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            issuer="ppda",
            audience="ppda-web",
            options={"require": ["exp", "sub", "sid", "iat"]},
        )
        uid = int(payload["sub"])
        sid = payload["sid"]
        with engine.connect() as conn:
            session = (
                conn.execute(
                    select(sessions).where(
                        sessions.c.sid == sid, sessions.c.user_id == uid
                    )
                )
                .mappings()
                .first()
            )
            if not session or session["revoked"] or session["expires"] <= time.time():
                raise ValueError("Expired session")
            user = conn.execute(select(users).where(users.c.id == uid)).mappings().one()
        if request.method not in ("GET", "HEAD", "OPTIONS") and not hmac.compare_digest(
            request.headers.get("x-csrf-token", ""), keyed("csrf:" + sid)
        ):
            raise HTTPException(403, "Invalid CSRF token; refresh the page")
        request.state.sid = sid
        return dict(user)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(401, "Please sign in")


def login_response(conn, user, response):
    sid = secrets.token_hex(32)
    now = time.time()
    expires = now + settings.session_hours * 3600
    conn.execute(
        sessions.insert().values(
            sid=sid, user_id=user["id"], expires=expires, revoked=False
        )
    )
    token = jwt.encode(
        {
            "sub": str(user["id"]),
            "sid": sid,
            "iat": int(now),
            "exp": int(expires),
            "iss": "ppda",
            "aud": "ppda-web",
        },
        settings.jwt_secret,
        algorithm="HS256",
    )
    response.set_cookie(
        COOKIE,
        token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path=PREFIX,
        max_age=settings.session_hours * 3600,
    )
    if settings.cookie_partitioned:
        # CHIPS for embedded HTTPS previews; Python 3.11 SimpleCookie lacks this attribute.
        response.headers["set-cookie"] += "; Partitioned"
    record(conn, user["id"], "SESSION_CREATED")
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "preferences": prefs(user),
        "csrf": keyed("csrf:" + sid),
    }


def rate_limit(request, email):
    now = time.time()
    # Direct peer only; do not trust attacker-supplied X-Forwarded-For.
    keys = [
        (keyed("login:" + email), 5),
        (keyed("peer:" + (request.client.host if request.client else "unknown")), 40),
    ]
    blocked = False
    with transaction() as conn:
        for key, cap in keys:
            row = (
                conn.execute(select(attempts).where(attempts.c.key == key))
                .mappings()
                .first()
            )
            times = (
                [t for t in json.loads(row["times"]) if t > now - 300] if row else []
            )
            if len(times) >= cap:
                blocked = True
            else:
                times.append(now)
                if row:
                    conn.execute(
                        update(attempts)
                        .where(attempts.c.key == key)
                        .values(times=json.dumps(times))
                    )
                else:
                    conn.execute(
                        attempts.insert().values(key=key, times=json.dumps(times))
                    )
    if blocked:
        raise HTTPException(
            429,
            "Too many attempts. Try again in five minutes.",
            headers={"Retry-After": "300"},
        )


def require(user, category):
    if not prefs(user)[category]:
        with transaction() as conn:
            record(conn, user["id"], "CONSENT_BLOCKED_" + category.upper())
        raise HTTPException(403, f"Enable {category} consent in Settings first")


def training_example(conn, user, text, label):
    user = conn.execute(select(users).where(users.c.id == user["id"])).mappings().one()
    category = {
        "calendar": "calendar",
        "reminder": "calendar",
        "note": "notes",
        "summary": "summary",
        "chat": "assistant",
    }[label]
    if not prefs(user)["training"] or not prefs(user)[category]:
        return False
    count = conn.execute(
        select(func.count())
        .select_from(examples)
        .where(examples.c.user_id == user["id"], examples.c.used == False)
    ).scalar()
    if count >= 100:
        return False
    conn.execute(
        examples.insert().values(
            user_id=user["id"],
            ciphertext=encrypt(user["id"], "training", {"text": text, "label": label}),
            used=False,
        )
    )
    record(conn, user["id"], "LOCAL_EXAMPLE_QUEUED")
    pipeline.trigger.set()
    return True


@app.get("/health")
def health():
    return {"status": "ok", "app": "PPDA"}


@app.post(PREFIX + "/auth/register", status_code=201)
def register(body: Register, request: Request, response: Response):
    email = str(body.email).lower()
    rate_limit(request, email)
    try:
        hashed = password_hash(body.password)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    with transaction() as conn:
        if conn.execute(select(users.c.id).where(users.c.email == email)).first():
            raise HTTPException(
                409, "Registration unavailable for this email. Try signing in."
            )
        preferences = DEFAULTS.copy()
        if body.local_consent:
            preferences.update(assistant=True, calendar=True, notes=True, summary=True)
        uid = conn.execute(
            users.insert().values(
                email=email,
                name=body.name.strip() or "User",
                password_hash=hashed,
                preferences=json.dumps(preferences),
                created_at=time.time(),
            )
        ).inserted_primary_key[0]
        user = conn.execute(select(users).where(users.c.id == uid)).mappings().one()
        record(conn, uid, "REGISTERED")
        if body.local_consent:
            record(conn, uid, "LOCAL_PROCESSING_CONSENT_GRANTED")
        return login_response(conn, user, response)


@app.post(PREFIX + "/auth/login")
def login(body: Login, request: Request, response: Response):
    email = str(body.email).lower()
    rate_limit(request, email)
    with engine.connect() as conn:
        user = (
            conn.execute(select(users).where(users.c.email == email)).mappings().first()
        )
    valid = verify_password(
        body.password, user["password_hash"] if user else DUMMY_HASH
    )
    if not user or not valid:
        raise HTTPException(401, "Incorrect email or password")
    with transaction() as conn:
        # Re-read to prevent a password change racing verification from creating a new session.
        fresh = (
            conn.execute(select(users).where(users.c.id == user["id"])).mappings().one()
        )
        if fresh["password_hash"] != user["password_hash"]:
            raise HTTPException(401, "Please sign in again")
        conn.execute(delete(attempts).where(attempts.c.key == keyed("login:" + email)))
        return login_response(conn, fresh, response)


@app.get(PREFIX + "/auth/session")
def session(request: Request, user=Depends(current_user)):
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "preferences": prefs(user),
        "csrf": keyed("csrf:" + request.state.sid),
    }


@app.post(PREFIX + "/auth/logout")
def logout(request: Request, response: Response, user=Depends(current_user)):
    with transaction() as conn:
        conn.execute(
            update(sessions)
            .where(sessions.c.sid == request.state.sid)
            .values(revoked=True)
        )
        record(conn, user["id"], "LOGGED_OUT")
    response.delete_cookie(
        COOKIE,
        path=PREFIX,
        secure=settings.cookie_secure,
        httponly=True,
        samesite=settings.cookie_samesite,
    )
    if settings.cookie_partitioned:
        response.headers["set-cookie"] += "; Partitioned"
    return {"ok": True}


@app.post(PREFIX + "/auth/password")
def change_password(
    body: PasswordChange,
    response: Response,
    request: Request,
    user=Depends(current_user),
):
    rate_limit(request, user["email"])
    if not verify_password(body.current_password, user["password_hash"]):
        raise HTTPException(401, "Incorrect current password")
    try:
        hashed = password_hash(body.new_password)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    with transaction() as conn:
        conn.execute(
            update(users).where(users.c.id == user["id"]).values(password_hash=hashed)
        )
        conn.execute(
            update(sessions)
            .where(sessions.c.user_id == user["id"])
            .values(revoked=True)
        )
        record(conn, user["id"], "PASSWORD_CHANGED_SESSIONS_REVOKED")
    response.delete_cookie(
        COOKIE,
        path=PREFIX,
        secure=settings.cookie_secure,
        httponly=True,
        samesite=settings.cookie_samesite,
    )
    if settings.cookie_partitioned:
        response.headers["set-cookie"] += "; Partitioned"
    return {"ok": True}


@app.patch(PREFIX + "/profile")
def profile(body: Profile, user=Depends(current_user)):
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Name is required")
    with transaction() as conn:
        conn.execute(update(users).where(users.c.id == user["id"]).values(name=name))
        record(conn, user["id"], "PROFILE_UPDATED")
    return {"name": name}


@app.put(PREFIX + "/settings")
def save_preferences(body: Preferences, user=Depends(current_user)):
    with transaction() as conn:
        fresh = (
            conn.execute(select(users).where(users.c.id == user["id"])).mappings().one()
        )
        previous = json.loads(fresh["preferences"])
        values = body.model_dump()
        values["_consent_version"] = previous.get("_consent_version", 0) + 1
        conn.execute(
            update(users)
            .where(users.c.id == user["id"])
            .values(preferences=json.dumps(values))
        )
        # Revoke any local category: drop all unused examples conservatively.
        revoked = any(
            previous.get(key, False) and not values[key]
            for key in ("assistant", "calendar", "notes", "summary")
        )
        if not body.training or revoked:
            conn.execute(
                delete(examples).where(
                    examples.c.user_id == user["id"], examples.c.used == False
                )
            )
        if not body.calendar:
            conn.execute(
                delete(push_subscriptions).where(
                    push_subscriptions.c.user_id == user["id"]
                )
            )
        record(conn, user["id"], "PREFERENCES_AND_CONSENT_UPDATED")
    pipeline.trigger.set()
    return body.model_dump()


KINDS = {"Calendar": "calendar", "Reminders": "calendar", "Notes": "notes"}


def kind_consent(kind, user):
    if kind not in KINDS:
        raise HTTPException(404, "Not found")
    require(user, KINDS[kind])


def entry_view(row, user):
    content = decrypt(row["user_id"], "entry:" + row["kind"], row["ciphertext"])
    if row["due_at"]:
        local = (
            datetime.fromisoformat(row["due_at"])
            .astimezone(ZoneInfo(prefs(user)["timezone"]))
            .replace(tzinfo=None)
        )
        content["detail"] = local.isoformat(
            timespec="microseconds"
            if local.microsecond
            else "seconds"
            if local.second
            else "minutes"
        )
    version = hashlib.sha256(
        (
            str(row["id"])
            + ":"
            + row["ciphertext"]
            + ":"
            + str(row["done"])
            + ":"
            + prefs(user)["timezone"]
        ).encode()
    ).hexdigest()
    return {
        "id": row["id"],
        **content,
        "done": row["done"],
        "fired": row["fired"],
        "version": version,
    }


@app.get(PREFIX + "/entries/{kind}")
def list_entries(kind: str, user=Depends(current_user)):
    kind_consent(kind, user)
    with engine.connect() as conn:
        rows = (
            conn.execute(
                select(entries)
                .where(entries.c.user_id == user["id"], entries.c.kind == kind)
                .order_by(entries.c.id.desc())
            )
            .mappings()
            .all()
        )
    return [entry_view(row, user) for row in rows]


def entry_values(kind, body, user):
    due = None
    if kind != "Notes":
        try:
            dt = datetime.fromisoformat(body.detail)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo(prefs(user)["timezone"]))
            due = dt.astimezone(timezone.utc).isoformat()
        except Exception:
            raise HTTPException(422, "Choose a valid date and time")
    return {
        "ciphertext": encrypt(
            user["id"], "entry:" + kind, {"title": body.title, "detail": body.detail}
        ),
        "due_at": due,
        "done": body.done,
    }


@app.post(PREFIX + "/entries/{kind}", status_code=201)
def create_entry(kind: str, body: Entry, user=Depends(current_user)):
    kind_consent(kind, user)
    values = entry_values(kind, body, user)
    with transaction() as conn:
        uid = conn.execute(
            entries.insert().values(
                user_id=user["id"], kind=kind, fired=False, **values
            )
        ).inserted_primary_key[0]
        record(conn, user["id"], kind.upper() + "_CREATED")
        if body.learning_text:
            training_example(
                conn,
                user,
                body.learning_text,
                {"Calendar": "calendar", "Reminders": "reminder", "Notes": "note"}[
                    kind
                ],
            )
        row = conn.execute(select(entries).where(entries.c.id == uid)).mappings().one()
        return entry_view(row, user)


@app.put(PREFIX + "/entries/{kind}/{item_id}")
def edit_entry(kind: str, item_id: int, body: Entry, user=Depends(current_user)):
    kind_consent(kind, user)
    values = entry_values(kind, body, user)
    with transaction() as conn:
        row = (
            conn.execute(
                select(entries).where(
                    entries.c.id == item_id,
                    entries.c.user_id == user["id"],
                    entries.c.kind == kind,
                )
            )
            .mappings()
            .first()
        )
        if not row:
            raise HTTPException(404, "Not found")
        if (
            body.expected_version
            and body.expected_version != entry_view(row, user)["version"]
        ):
            raise HTTPException(
                409,
                "This item changed since the draft was opened. Refresh and review it again.",
            )
        if values["due_at"] != row["due_at"]:
            values["fired"] = False
            notifications.cancel_entry(conn, item_id)
        if body.done:
            notifications.cancel_entry(conn, item_id)
        conn.execute(update(entries).where(entries.c.id == item_id).values(**values))
        record(conn, user["id"], kind.upper() + "_UPDATED")
        return entry_view(
            conn.execute(select(entries).where(entries.c.id == item_id))
            .mappings()
            .one(),
            user,
        )


@app.delete(PREFIX + "/entries/{kind}/{item_id}")
def delete_entry(
    kind: str,
    item_id: int,
    version: str | None = Header(default=None, alias="If-Match"),
    user=Depends(current_user),
):
    kind_consent(kind, user)
    with transaction() as conn:
        row = (
            conn.execute(
                select(entries).where(
                    entries.c.id == item_id,
                    entries.c.user_id == user["id"],
                    entries.c.kind == kind,
                )
            )
            .mappings()
            .first()
        )
        if not row:
            raise HTTPException(404, "Not found")
        if version and version != entry_view(row, user)["version"]:
            raise HTTPException(
                409, "This item changed. Refresh and review before deleting."
            )
        notifications.cancel_entry(conn, item_id)
        conn.execute(delete(entries).where(entries.c.id == item_id))
        record(conn, user["id"], kind.upper() + "_DELETED")
    return {"ok": True}


def proposal(text, label, user):
    kind = {"calendar": "Calendar", "reminder": "Reminders", "note": "Notes"}[label]
    detail = ""
    title = text
    if label != "note":
        match = re.search(
            r"\b(tomorrow|today|next\s+\w+|in\s+\d+\s+(?:hours?|minutes?|days?)|at\s+\d{1,2}(?::\d{2})?(?:\s*(?:am|pm))?|\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?)\b.*",
            text,
            re.I,
        )
        if match:
            date = dateparser.parse(
                match.group(),
                settings={
                    "PREFER_DATES_FROM": "future",
                    "TIMEZONE": prefs(user)["timezone"],
                    "RETURN_AS_TIMEZONE_AWARE": True,
                },
            )
            if date:
                detail = date.strftime("%Y-%m-%dT%H:%M")
                title = text[: match.start()].strip()
    title = re.sub(
        r"^(remind me to|remind me|set a reminder to|set a reminder|add a reminder to|add a reminder|schedule|create (?:a )?(?:note|task|event)|write a note|add (?:a )?(?:note|reminder|event))\s*[:\-]?\s*",
        "",
        title,
        flags=re.I,
    )
    quoted = re.search(r'"([^"\n]+)"', title)
    if quoted:
        title = quoted.group(1)
    return {"kind": kind, "title": (title or text)[:160], "detail": detail}


async def cloud_answer(text):
    if not settings.openai_api_key:
        raise HTTPException(
            503,
            "Global provider is not configured. Use Privacy mode or configure OPENAI_API_KEY on the backend.",
        )
    try:
        async with httpx.AsyncClient(timeout=35) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": "Bearer " + settings.openai_api_key},
                json={
                    "model": settings.openai_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. You cannot access or modify the user calendar, notes, or reminders. Do not claim to have performed actions.",
                        },
                        {"role": "user", "content": text},
                    ],
                    "max_tokens": 700,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(
            503, "Cloud provider unavailable. No local task was executed."
        )


async def local_chat(text):
    if settings.ollama_url and settings.ollama_model:
        parsed = urlparse(settings.ollama_url)
        if parsed.hostname not in (
            "localhost",
            "127.0.0.1",
            "::1",
        ) or parsed.scheme not in ("http", "https"):
            raise HTTPException(503, "Privacy-mode LLM must use a loopback runtime")
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
                r = await client.post(
                    settings.ollama_url.rstrip("/") + "/api/generate",
                    json={
                        "model": settings.ollama_model,
                        "prompt": text,
                        "stream": False,
                    },
                )
                r.raise_for_status()
                return r.json()["response"]
        except Exception:
            raise HTTPException(
                503, "Local LLM unavailable. Nothing was sent to the cloud."
            )
    if re.search(r"\b(hi|hello|thanks|help|morning|hey)\b", text, re.I):
        return "Hello! I can help with calendar events, reminders, notes, and local text summaries. Try “remind me to call Rahul tomorrow at 5 pm”. I’ll ask you to review it before saving."
    return "The local intent model handles scheduling, reminders, notes, and configured local summaries—not general conversation. Configure a local Ollama runtime for broader offline chat, or explicitly enable Global mode."


@app.post(PREFIX + "/assistant/command")
async def command(body: Command, user=Depends(current_user)):
    require(user, "assistant")
    text = body.text.strip()
    if not text:
        raise HTTPException(422, "Enter a message")
    with engine.connect() as conn:
        version, weights = pipeline.active_model(conn)
    label, confidence, explanation = classify(weights, text)
    # ML never executes tools. Explicit workspace mutations take precedence over words in titles.
    kind = assistant_actions.infer_kind(text)
    op = assistant_actions.operation(text)
    if kind is None and op in ("update", "delete"):
        selector, value = assistant_actions.target_text(text)
        if selector == "id":
            with engine.connect() as conn:
                kind = conn.execute(
                    select(entries.c.kind).where(
                        entries.c.user_id == user["id"], entries.c.id == value
                    )
                ).scalar()
    if kind and op in ("update", "delete", "list"):
        intent = {"Calendar": "calendar", "Reminders": "reminder", "Notes": "note"}[
            kind
        ]
    elif re.search(
        r"\b(summari[sz](?:e|er)|summary|tldr|shorten)\b|key points", text, re.I
    ):
        intent = "summary"
    else:
        intent = {"Calendar": "calendar", "Reminders": "reminder", "Notes": "note"}.get(
            kind, "chat"
        )
    # Default never sends workspace commands to cloud. Only explicit Global, or opted-in
    # unsupported general chat in Default, can use a provider. No history attached.
    use_cloud = body.mode == "Global" or (
        body.mode == "Default"
        and intent == "chat"
        and op not in ("update", "delete")
        and prefs(user)["cloud"]
        and bool(settings.openai_api_key)
        and not re.search(r"\b(hi|hello|thanks|help|morning|hey)\b", text, re.I)
    )
    result = {
        "intent": intent,
        "model_intent": label,
        "confidence": confidence,
        "explanation": explanation,
        "explanation_method": "token occlusion (local intent classifier)",
        "model_version": version,
        "location": "local backend",
        "proposal": None,
        "text": "",
        "summarization_engine": None,
    }
    if body.mode == "Default" and re.search(
        r"\b(password|secret|private|bank|medical|diagnosis|credit|ssn|token|address|personal)\b|[\w.+-]+@[\w.-]+",
        text,
        re.I,
    ):
        use_cloud = False
    if use_cloud:
        require(user, "cloud")
        with transaction() as conn:
            record(conn, user["id"], "CLOUD_PROMPT_AUTHORIZED")
        try:
            result["text"] = await cloud_answer(text)
        except HTTPException:
            with transaction() as conn:
                record(conn, user["id"], "CLOUD_PROVIDER_FAILED")
            raise
        result["location"] = "OpenAI · " + settings.openai_model
        if intent == "summary":
            result["summarization_engine"] = "OpenAI · " + settings.openai_model
    elif intent == "summary":
        require(user, "summary")
        try:
            result["text"] = await asyncio.to_thread(summarization.summarize, text)
        except summarization.SummaryInputError as exc:
            raise HTTPException(422, str(exc))
        except summarization.SummaryUnavailable as exc:
            raise HTTPException(503, str(exc))
        result["summarization_engine"] = summarization.engine_label()
    elif intent in ("calendar", "reminder", "note"):
        kind = {"calendar": "Calendar", "reminder": "Reminders", "note": "Notes"}[
            intent
        ]
        kind_consent(kind, user)
        result.update(
            assistant_actions.plan(
                text,
                kind,
                list_entries(kind, user),
                prefs(user)["timezone"],
                lambda: proposal(text, intent, user),
            )
        )
    elif op in ("update", "delete"):
        result["text"] = (
            "Specify an existing calendar event or reminder using its ID or a quoted title. Nothing was changed."
        )
    else:
        result["text"] = await local_chat(text)
    with transaction() as conn:
        record(
            conn,
            user["id"],
            "ASSISTANT_CLOUD_COMPLETED" if use_cloud else "ASSISTANT_LOCAL_COMPLETED",
        )
    return result


@app.post(PREFIX + "/learning/examples", status_code=201)
def add_example(body: Example, user=Depends(current_user)):
    require(user, "training")
    require(
        user,
        {
            "calendar": "calendar",
            "reminder": "calendar",
            "note": "notes",
            "summary": "summary",
            "chat": "assistant",
        }[body.label],
    )
    with transaction() as conn:
        queued = training_example(conn, user, body.text, body.label)
    if not queued:
        raise HTTPException(
            409, "Queue is full or consent changed. No example was added."
        )
    return {
        "ok": True,
        "detail": "Validated example queued locally; training requires a real cohort.",
    }


@app.get(PREFIX + "/learning/status")
def learning_status(user=Depends(current_user)):
    with engine.connect() as conn:
        epsilon, delta = expenditure(conn, user["id"])
        queued = conn.execute(
            select(func.count())
            .select_from(examples)
            .where(examples.c.user_id == user["id"], examples.c.used == False)
        ).scalar()
        version, _ = pipeline.active_model(conn)
        history = []
        for row in conn.execute(select(rounds).order_by(rounds.c.id.desc())).mappings():
            if user["id"] in json.loads(row["participants"]):
                history.append(
                    {"id": row["id"], "status": row["status"], "detail": row["detail"]}
                )
                if len(history) >= 10:
                    break
        state = "Not participating"
        if prefs(user)["training"]:
            state = (
                "Budget exhausted"
                if not affordable(conn, user)
                else (
                    "Awaiting 3 validated examples"
                    if queued < 3
                    else "Waiting for 3 eligible clients"
                )
            )
        if history and history[0]["status"] == "training":
            state = "Training in independent client processes"
    return {
        "state": state,
        "queued": queued,
        "epsilon_spent": epsilon,
        "delta_spent": delta,
        "remaining": max(0, prefs(user)["epsilon"] - epsilon),
        "delta_cap": DELTA_CAP,
        "epsilon_per_round": EPSILON_PER_ROUND,
        "delta_per_round": DELTA_PER_ROUND,
        "clip_norm": CLIP_NORM,
        "model_version": version,
        "history": history,
        "mechanism": "Client-local Gaussian DP; conservative sequential composition; lifetime accounting",
        "scope": "Single trusted host, real OS workers; not independent physical devices.",
        "pipeline_enabled": settings.pipeline_enabled,
    }


@app.post(PREFIX + "/learning/run")
def request_learning(user=Depends(current_user)):
    require(user, "training")
    if not settings.pipeline_enabled:
        raise HTTPException(503, "Learning pipeline disabled by the operator")
    pipeline.trigger.set()
    return {
        "detail": "Eligibility check requested. No round is fabricated if fewer than three clients qualify."
    }


@app.get(PREFIX + "/audit")
def audit_list(user=Depends(current_user)):
    with engine.connect() as conn:
        rows = (
            conn.execute(
                select(audits)
                .where(audits.c.user_id == user["id"])
                .order_by(audits.c.id.desc())
                .limit(100)
            )
            .mappings()
            .all()
        )
    return [
        {
            "id": r["id"],
            "action": r["action"],
            "created_at": r["created_at"],
            "integrity_hash": r["integrity_hash"],
        }
        for r in rows
    ]


@app.get(PREFIX + "/audit/verify")
def audit_verify(user=Depends(current_user)):
    with engine.connect() as conn:
        rows = (
            conn.execute(
                select(audits)
                .where(audits.c.user_id == user["id"])
                .order_by(audits.c.id)
            )
            .mappings()
            .all()
        )
    previous = "GENESIS"
    for row in rows:
        payload = json.dumps(
            [previous, user["id"], row["action"], row["created_at"]],
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        if (
            row["prev_hash"] != previous
            or digest != row["integrity_hash"]
            or not hmac.compare_digest(row["signature"], keyed("audit:" + digest))
        ):
            return {"valid": False, "broken_at_id": row["id"], "checked": len(rows)}
        previous = digest
    return {
        "valid": True,
        "checked": len(rows),
        "limitation": "No external checkpoint: whole-history deletion/truncation is not independently detectable.",
    }


@app.get(PREFIX + "/runtime")
def runtime(user=Depends(current_user)):
    return {
        "local_model": "ONNX Runtime · 128-feature softmax intent model",
        "local_llm": bool(settings.ollama_url and settings.ollama_model),
        "summarization_engine": summarization.engine_label(),
        "local_summary": summarization.status(),
        "cloud_model": settings.openai_model,
        "cloud_configured": bool(settings.openai_api_key),
        "cloud_provider": "OpenAI",
        "encrypted_storage": "AES-256-GCM",
        "secure_cookie": settings.cookie_secure,
        "boundary": "Processing runs on this backend host, not in the browser.",
        "crypto_review": "Research implementation; not independently audited.",
    }


class SubscriptionKeys(Strict):
    p256dh: str = Field(min_length=1, max_length=128)
    auth: str = Field(min_length=1, max_length=64)


class PushSubscription(Strict):
    endpoint: str = Field(min_length=1, max_length=2048)
    keys: SubscriptionKeys
    expirationTime: float | None = None


class PushUnsubscribe(Strict):
    endpoint: str = Field(min_length=1, max_length=2048)


@app.get(PREFIX + "/notifications/status")
def notification_status(user=Depends(current_user)):
    with engine.connect() as conn:
        count = conn.execute(
            select(func.count())
            .select_from(push_subscriptions)
            .where(push_subscriptions.c.user_id == user["id"])
        ).scalar()
    return {
        "configured": bool(notifications.public_key()),
        "subscriptions": count,
        "detail": "Best-effort browser push at due time. Backend must stay online; OS/network/browser policies can delay delivery. Lock-screen text is generic.",
    }


@app.get(PREFIX + "/notifications/public-key")
def notification_key(user=Depends(current_user)):
    require(user, "calendar")
    key = notifications.public_key()
    if not key:
        raise HTTPException(
            503, "Push is not configured. The operator must run scripts/init_vapid.py."
        )
    return {"public_key": key}


@app.post(PREFIX + "/notifications/subscribe", status_code=201)
def subscribe(body: PushSubscription, user=Depends(current_user)):
    require(user, "calendar")
    if not notifications.public_key():
        raise HTTPException(503, "Push is not configured")
    data = body.model_dump(exclude={"expirationTime"})
    try:
        notifications.validate_subscription(data)
    except Exception:
        raise HTTPException(422, "Invalid or unsupported browser push subscription")
    identifier = keyed("push:" + body.endpoint)
    with transaction() as conn:
        row = (
            conn.execute(
                select(push_subscriptions).where(
                    push_subscriptions.c.endpoint_hash == identifier
                )
            )
            .mappings()
            .first()
        )
        if row and row["user_id"] != user["id"]:
            raise HTTPException(
                409,
                "This browser subscription belongs to another account. Unsubscribe in the browser and enable again.",
            )
        values = {
            "user_id": user["id"],
            "endpoint_hash": identifier,
            "ciphertext": encrypt(user["id"], "push-subscription", data),
            "created_at": time.time(),
        }
        if row:
            conn.execute(
                update(push_subscriptions)
                .where(push_subscriptions.c.id == row["id"])
                .values(**values)
            )
        else:
            count = conn.execute(
                select(func.count())
                .select_from(push_subscriptions)
                .where(push_subscriptions.c.user_id == user["id"])
            ).scalar()
            if count >= 10:
                raise HTTPException(
                    409, "At most 10 browser subscriptions are allowed per account"
                )
            conn.execute(push_subscriptions.insert().values(**values))
        record(conn, user["id"], "PUSH_NOTIFICATIONS_ENABLED")
    return {"ok": True}


@app.post(PREFIX + "/notifications/unsubscribe")
def unsubscribe(body: PushUnsubscribe, user=Depends(current_user)):
    with transaction() as conn:
        conn.execute(
            delete(push_subscriptions).where(
                push_subscriptions.c.user_id == user["id"],
                push_subscriptions.c.endpoint_hash == keyed("push:" + body.endpoint),
            )
        )
        record(conn, user["id"], "PUSH_NOTIFICATIONS_DISABLED")
    return {"ok": True}


@app.get(PREFIX + "/models/snips/status")
def snips_status(user=Depends(current_user)):
    return snips.status(settings.snips_dir)


class SnipsText(Strict):
    text: str = Field(min_length=1, max_length=10000)


@app.post(PREFIX + "/models/snips/predict")
def snips_predict(body: SnipsText, user=Depends(current_user)):
    require(user, "assistant")
    if not snips.status(settings.snips_dir)["available"]:
        raise HTTPException(503, "SNIPS benchmark model not trained on this deployment")
    result = snips.predict(settings.snips_dir, body.text)
    with transaction() as conn:
        record(conn, user["id"], "SNIPS_LOCAL_INFERENCE")
    return result


@app.post(PREFIX + "/notifications/subscription-status")
def subscription_status(body: PushUnsubscribe, user=Depends(current_user)):
    with engine.connect() as conn:
        found = conn.execute(
            select(push_subscriptions.c.id).where(
                push_subscriptions.c.user_id == user["id"],
                push_subscriptions.c.endpoint_hash == keyed("push:" + body.endpoint),
            )
        ).first()
    return {"enabled": bool(found)}
