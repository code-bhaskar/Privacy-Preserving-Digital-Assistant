"""Deterministic CRUD planning. Returns drafts only; never mutates stored records."""

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import dateparser

TIME_PATTERN = re.compile(
    r"\b(?:tomorrow|today|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|in\s+\d+\s+(?:hours?|minutes?|days?)|at\s+\d{1,2}(?::\d{2})?(?:\s*(?:am|pm))?|\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?)\b.*",
    re.I,
)


def parsed_time(text, timezone, existing=""):
    match = TIME_PATTERN.search(text)
    if not match:
        return ""
    settings = {
        "PREFER_DATES_FROM": "future",
        "TIMEZONE": timezone,
        "RETURN_AS_TIMEZONE_AWARE": True,
    }
    if existing and match.group().lower().startswith("at "):
        settings["RELATIVE_BASE"] = datetime.fromisoformat(existing).replace(
            tzinfo=ZoneInfo(timezone)
        )
    result = dateparser.parse(match.group(), settings=settings)
    if result:
        return result.strftime("%Y-%m-%dT%H:%M")
    return ""


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
        r"\s+(?:to|for|at)\s+(?=tomorrow|today|next|in\s+\d|\d)|\s+(?:title|name)\s+to\s+",
        rest,
        maxsplit=1,
        flags=re.I,
    )[0]
    return ("title", rest.strip())


def plan(text, kind, items, timezone, create_draft):
    op = operation(text)
    if op == "list":
        return {
            "proposal": None,
            "text": "\n".join(
                f"• #{r['id']} {r['title']} — {r['detail']}" for r in items[:30]
            )
            or f"Your {kind.lower()} workspace is empty.",
        }
    if op == "create":
        draft = create_draft()
        draft["operation"] = "create"
        return {
            "proposal": draft,
            "text": "I’ve prepared a draft. Review the title and time, then confirm to create it.",
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
        }
    if len(matches) > 1:
        return {
            "proposal": None,
            "text": "More than one item matches. Please choose an ID; nothing was changed:\n"
            + "\n".join(
                f"• #{r['id']} {r['title']} — {r['detail']}" for r in matches[:10]
            ),
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
        date = parsed_time(text, timezone, item["detail"]) if kind != "Notes" else ""
        if date:
            draft["detail"] = date
        rename = re.search(r'\b(?:title|name)\s+(?:to\s+)?"([^"\n]+)"', text, re.I)
        if not rename:
            rename = re.search(r'\bto\s+"([^"\n]+)"', text, re.I)
        if rename:
            draft["title"] = rename.group(1)
        # Unparsed fields stay unchanged and remain editable in the review dialog.
        return {
            "proposal": draft,
            "text": "Review this update. Only the shown fields will change after you confirm; the original item is still unchanged."
            if any(draft[k] != item[k] for k in ("title", "detail", "done"))
            else "I found the item, but couldn’t parse a replacement value. Edit the fields in the draft before confirming. Nothing has changed.",
        }
    return {
        "proposal": draft,
        "text": "Review the item below before confirming deletion. Nothing has been deleted yet.",
    }
