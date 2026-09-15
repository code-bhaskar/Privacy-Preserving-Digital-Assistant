"""Deterministic CRUD planning. Returns drafts only; never mutates stored records.

Two responsibilities:

1. Turn a natural-language request into a *draft* (create/update/delete) plus the
   list of items it would touch. Nothing is written here; the API requires an
   explicit confirmation against an item version hash.
2. Describe the draft as a human-readable change list with a correctness check,
   so the review dialog can answer "are these the changes you meant?" instead of
   only showing raw fields.

Time parsing runs on text that `app.normalizer` has already rewritten, so spoken
numbers ("seven thirty pm") arrive here as "7:30 pm". dateparser 1.2.1 cannot
parse "next monday" at all, so a narrow, disclosed fallback covers it.
"""

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import dateparser

TIME_PATTERN = re.compile(
    r"\b(?:tomorrow|today|tonight|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month)|in\s+\d+\s+(?:hours?|hrs?|minutes?|mins?|days?|weeks?)|on\s+\w+|at\s+\d{1,2}(?::\d{2})?(?:\s*(?:am|pm))?|\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?)\b.*",
    re.I,
)
NEXT_WEEKDAY = re.compile(
    r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I
)
CLOCK_ONLY = re.compile(
    r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b(?=\s*$|\s+(?:to|for|and|with|about)\b)",
    re.I,
)


def time_span(text):
    """The part of the sentence that carries the time, or None."""
    return TIME_PATTERN.search(text)


def _settings(timezone, existing="", phrase=""):
    settings = {
        "PREFER_DATES_FROM": "future",
        "TIMEZONE": timezone,
        "RETURN_AS_TIMEZONE_AWARE": True,
    }
    if existing and phrase.lower().startswith("at "):
        settings["RELATIVE_BASE"] = datetime.fromisoformat(existing).replace(
            tzinfo=ZoneInfo(timezone)
        )
    return settings


def _combine(phrase, timezone, existing=""):
    """Last-resort parse: read the clock separately from the date phrase."""
    clock = CLOCK_ONLY.search(phrase)
    if not clock:
        return None
    hour = int(clock.group(1))
    minute = int(clock.group(2) or 0)
    meridiem = (clock.group(3) or "").lower()
    if hour > 23 or minute > 59:
        return None
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    zone = ZoneInfo(timezone)
    date_part = phrase[: clock.start()].strip()
    base = None
    if date_part:
        base = dateparser.parse(
            NEXT_WEEKDAY.sub(r"on \1", date_part),
            settings=_settings(timezone, existing, date_part),
        )
    if base is None:
        base = (
            datetime.fromisoformat(existing).replace(tzinfo=zone)
            if existing
            else datetime.now(zone)
        )
    candidate = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if not existing and candidate <= datetime.now(zone):
        # No date was given, so a bare clock time means the next occurrence.
        candidate += timedelta(days=1)
    return candidate


def _candidates(phrase):
    """The matched phrase, then progressively shorter prefixes of it.

    dateparser rejects a phrase that still carries the rest of the sentence
    ("in 20 minutes to stretch"), so the trailing prose is dropped word by word.
    """
    words = phrase.split()
    seen = [phrase]
    for count in range(len(words) - 1, 0, -1):
        candidate = " ".join(words[:count])
        if candidate not in seen:
            seen.append(candidate)
        if len(seen) >= 6:
            break
    return seen


def parse_time(text, timezone, existing=""):
    """Return (ISO local string, note). Empty string when no time was found."""
    match = time_span(text)
    if not match:
        return "", ""
    phrase = match.group()
    result = dateparser.parse(phrase, settings=_settings(timezone, existing, phrase))
    note = ""
    if result is None and len(phrase.split()) > 2:
        for candidate in _candidates(phrase)[1:]:
            result = dateparser.parse(
                candidate, settings=_settings(timezone, existing, candidate)
            )
            if result is not None:
                note = f"Read the time as “{candidate}”"
                break
    if result is None and NEXT_WEEKDAY.search(phrase):
        # dateparser cannot parse "next monday"; read it as the upcoming weekday.
        rewritten = NEXT_WEEKDAY.sub(r"on \1", phrase)
        result = dateparser.parse(rewritten, settings=_settings(timezone, existing, phrase))
        if result is not None:
            note = f"“{match.group().split(' at ')[0]}” read as {result:%A %d %b}"
    if result is None:
        result = _combine(phrase, timezone, existing)
        if result is not None:
            note = f"Date and clock time parsed separately for “{phrase.strip()}”"
    if result is None:
        return "", f"“{phrase.strip()}” could not be parsed as a time"
    if not note and CLOCK_ONLY.search(phrase) and not re.search(
        r"\b(am|pm)\b", phrase, re.I
    ):
        note = f"No am/pm given, so {result:%H:%M} means the next occurrence"
    return result.strftime("%Y-%m-%dT%H:%M"), note


def parsed_time(text, timezone, existing=""):
    return parse_time(text, timezone, existing)[0]


def operation(text):
    start = re.sub(
        r"^\s*(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?", "", text, flags=re.I
    )
    if re.match(r"(delete|remove|cancel)\b", start, re.I):
        return "delete"
    if re.match(
        r"(update|edit|reschedule|move|rename|change|mark|complete|finish|reopen|undo)\b",
        start,
        re.I,
    ):
        return "update"
    if re.match(r"(show|list|view|what\b|what's\b)", start, re.I):
        return "list"
    return "create"


def infer_kind(text):
    # Recognise plurals and common spelling variants used in the interface request.
    if re.search(
        r"\b(remind(?:er)?s?|remainder[s]?|remember|todo|task[s]?|forget)\b", text, re.I
    ):
        return "Reminders"
    if re.search(
        r"\b(calend[ae]rs?|schedule|meeting[s]?|appointment[s]?|event[s]?)\b",
        text,
        re.I,
    ):
        return "Calendar"
    if re.search(r"\b(note[s]?|notebook)\b", text, re.I):
        return "Notes"
    return None


def target_text(text):
    # Explicit identifiers/quoted titles are preferred to fuzzy matching.
    id_match = re.search(r"(?:#|\bid\s+)\s*(\d+)\b", text, re.I)
    if id_match:
        return ("id", int(id_match.group(1)))
    quoted = re.search(r'"([^"\n]+)"', text)
    if quoted:
        return ("title", quoted.group(1).strip())
    rest = re.sub(
        r"^\s*(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?(?:delete|remove|cancel|update|edit|reschedule|move|rename|change|mark|complete|finish|reopen|undo)\s+(?:(?:the|my)\s+)?(?:(?:calendar\s+)?(?:events?|meetings?|appointments?|reminders?|remainders?|tasks?|notes?)\s*)?",
        "",
        text,
        flags=re.I,
    )
    rest = re.split(
        r"\s+(?:to|for|at)\s+(?=tomorrow|today|tonight|next|in\s+\d|\d)|\s+(?:title|name)\s+to\s+",
        rest,
        maxsplit=1,
        flags=re.I,
    )[0]
    return ("title", rest.strip())


def when(detail, timezone):
    """Human-readable rendering of a stored due time, or the raw note text."""
    if not detail:
        return "no time set"
    if not re.match(r"^\d{4}-\d{2}-\d{2}T", detail):
        return detail
    return (
        datetime.fromisoformat(detail).strftime("%a %d %b %Y at %H:%M")
        + f" ({timezone})"
    )


def review(kind, operation_name, draft, item=None, timezone="", notes=None):
    """Change list + correctness check for a planned draft.

    `changes` is what the UI shows as "here is what will change"; `check` tells
    the user whether the parse looks complete, and says so plainly when it is not.
    """
    notes = list(notes or [])
    changes = []
    timed = kind != "Notes"
    if operation_name == "create":
        changes = [
            {"field": "Action", "before": None, "after": f"Create {kind.lower()}"},
            {"field": "Title", "before": None, "after": draft["title"]},
        ]
        if timed:
            changes.append(
                {"field": "When", "before": None, "after": when(draft["detail"], timezone)}
            )
    elif operation_name == "update":
        changes = [{"field": "Action", "before": None, "after": f"Update {kind.lower()} #{draft['id']}"}]
        for field, key in (("Title", "title"), ("When", "detail"), ("Done", "done")):
            if key == "done" and item and draft["done"] == item["done"]:
                continue
            before = item[key] if item else None
            after = draft[key]
            if key == "detail":
                before, after = when(before, timezone), when(after, timezone)
            if key == "done":
                before, after = "not done" if not before else "done", (
                    "done" if after else "not done"
                )
            changes.append(
                {
                    "field": field,
                    "before": before,
                    "after": after,
                    "changed": before != after,
                }
            )
    else:
        changes = [
            {"field": "Action", "before": None, "after": f"Delete {kind.lower()} #{draft['id']}"},
            {"field": "Title", "before": draft["title"], "after": None},
        ]
        if timed:
            changes.append(
                {
                    "field": "When",
                    "before": when(draft["detail"], timezone),
                    "after": None,
                }
            )
    changes.append(
        {
            "field": "Storage",
            "before": None,
            "after": "Nothing is written until you confirm this review",
        }
    )
    warnings = []
    if timed and operation_name in ("create", "update") and not draft["detail"]:
        warnings.append("No time was found in your message, so no due time is set.")
    if timed and draft["detail"] and re.match(r"^\d{4}-\d{2}-\d{2}T", draft["detail"]):
        due = datetime.fromisoformat(draft["detail"]).replace(tzinfo=ZoneInfo(timezone))
        if due <= datetime.now(ZoneInfo(timezone)):
            warnings.append("The parsed time is already in the past.")
    if operation_name == "update" and item and not any(
        change.get("changed") for change in changes if "changed" in change
    ):
        warnings.append("Nothing differs from the stored item yet; edit a field below.")
    return {
        "changes": changes,
        "check": {
            "ok": not warnings,
            "summary": (
                "These are the changes the assistant understood. Confirm to apply them."
                if not warnings
                else "Review the warnings below before confirming."
            ),
            "notes": notes,
            "warnings": warnings,
        },
    }


def plan(text, kind, items, timezone, create_draft, notes=None):
    op = operation(text)
    if op == "list":
        return {
            "proposal": None,
            "text": "\n".join(
                f"• #{r['id']} {r['title']} — {r['detail']}" for r in items[:30]
            )
            or f"Your {kind.lower()} workspace is empty.",
            "changes": [],
            "check": {
                "ok": True,
                "summary": "Read-only listing. Nothing will be changed.",
                "notes": [],
                "warnings": [],
            },
        }
    if op == "create":
        draft = create_draft()
        draft["operation"] = "create"
        details = review(kind, "create", draft, None, timezone, notes)
        return {
            "proposal": draft,
            "text": "I’ve prepared a draft. Review the title and time, then confirm to create it.",
            **details,
        }
    selector, value = target_text(text)
    if selector == "id":
        matches = [r for r in items if r["id"] == value]
    else:
        normalized = " ".join(str(value).casefold().split())
        if not normalized or normalized in (
            "all",
            "everything",
            "them",
            "all reminders",
            "all events",
        ):
            return {
                "proposal": None,
                "text": "Please specify one item by its ID or a quoted title. Bulk deletion/updates are not supported.",
                "changes": [],
                "check": {
                    "ok": False,
                    "summary": "No item was selected, so nothing can be planned.",
                    "notes": [],
                    "warnings": ["Specify one item by ID or quoted title."],
                },
            }
        matches = [
            r for r in items if " ".join(r["title"].casefold().split()) == normalized
        ]
        if not matches and len(normalized) >= 3:
            matches = [
                r
                for r in items
                if normalized in " ".join(r["title"].casefold().split())
            ]
    if not matches:
        return {
            "proposal": None,
            "text": "I couldn’t find that item in your workspace. List your items, then use an ID such as “delete reminder #12”. Nothing was changed.",
            "changes": [],
            "check": {
                "ok": False,
                "summary": "No matching item, so nothing was planned.",
                "notes": [],
                "warnings": ["No workspace item matched that description."],
            },
        }
    if len(matches) > 1:
        return {
            "proposal": None,
            "text": "More than one item matches. Please choose an ID; nothing was changed:\n"
            + "\n".join(
                f"• #{r['id']} {r['title']} — {r['detail']}" for r in matches[:10]
            ),
            "changes": [],
            "check": {
                "ok": False,
                "summary": "The description was ambiguous.",
                "notes": [],
                "warnings": [f"{len(matches)} items matched; pick one by ID."],
            },
        }
    item = matches[0]
    draft = {
        "operation": op,
        "id": item["id"],
        "kind": kind,
        "title": item["title"],
        "detail": item["detail"],
        "done": item["done"],
        "version": item["version"],
    }
    if op == "update":
        if kind == "Reminders":
            if re.search(
                r"^(?:please\s+)?(?:complete|finish)\b|\b(?:as|to)\s+(?:done|complete|finished)\s*$|^mark\b.*\bdone\s*$",
                text,
                re.I,
            ):
                draft["done"] = True
            elif re.search(
                r"^(?:please\s+)?(?:reopen|undo)\b|\b(?:as|to)\s+(?:incomplete|pending)\s*$",
                text,
                re.I,
            ):
                draft["done"] = False
        date, note = parse_time(text, timezone, item["detail"]) if kind != "Notes" else ("", "")
        if date:
            draft["detail"] = date
        rename = re.search(r'\b(?:title|name)\s+(?:to\s+)?"([^"\n]+)"', text, re.I)
        if not rename:
            rename = re.search(r'\bto\s+"([^"\n]+)"', text, re.I)
        if rename:
            draft["title"] = rename.group(1)
        # Unparsed fields stay unchanged and remain editable in the review dialog.
        details = review(kind, "update", draft, item, timezone, [note] if note else [])
        return {
            "proposal": draft,
            "text": "Review this update. Only the shown fields will change after you confirm; the original item is still unchanged."
            if any(draft[k] != item[k] for k in ("title", "detail", "done"))
            else "I found the item, but couldn’t parse a replacement value. Edit the fields in the draft before confirming. Nothing has changed.",
            **details,
        }
    details = review(kind, "delete", draft, item, timezone)
    return {
        "proposal": draft,
        "text": "Review the item below before confirming deletion. Nothing has been deleted yet.",
        **details,
    }
