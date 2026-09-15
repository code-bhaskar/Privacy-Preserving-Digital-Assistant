"""Offline input normalisation: tolerant spelling and spoken number words.

Runs before intent classification and before any time parsing, so a request such
as ``set a remindar at seven thirty pm`` reaches the model and dateparser as
``set a reminder at 7:30 pm``. Everything is deterministic: fixed lexicons, fixed
rewrites, no network access and no model weights. Each change is returned so the
UI can show the user exactly what the assistant understood, and the user can
still edit the resulting draft.

Nothing here executes an action. It only rewrites the text that the rest of the
local pipeline reads.
"""

import re
from functools import lru_cache

# Public domain vocabulary. Both spellings of every variant are listed so a
# valid British/American form is never "corrected".
DOMAIN_WORDS = set(
    """
    add added adding afternoon agenda all am anniversary appointment appointments
    at attend bill birthday book booking both buy by calendar calendars call
    calls cancel cancelled canceling canceling change checklist class complete
    completed conference confirm create dentist description design dinner doctor
    due edit email emails event events exam every finish finished first flight
    forget friday from groceries gym holiday homework hour hours interview item
    items list lunch mark meeting meetings medicine milk minutes monday morning
    move my new next night note notebook notes now office open order pay payment
    pick plan please pm prepare presentation project read reminder reminders
    remind remember remove rename report reschedule review saturday schedule
    scheduled school send set shopping show submission summary summarize
    summarise summarised summarized sunday task tasks thursday time title today
    tomorrow tonight tuesday update view water wednesday week weekday weekend
    work write written year yoga dentist groceries
    """.split()
)

# Explicit, unambiguous misspellings. Applied even to capitalised words, because
# these mappings cannot plausibly be a proper noun.
KNOWN_MISSPELLINGS = {
    "remindar": "reminder",
    "remender": "reminder",
    "remidner": "reminder",
    "remiders": "reminders",
    "remnders": "reminders",
    "rember": "remember",
    "remeber": "remember",
    "calender": "calendar",
    "calander": "calendar",
    "calendarr": "calendar",
    "calenders": "calendars",
    "sumarize": "summarize",
    "summarise": "summarise",
    "summerize": "summarize",
    "sumary": "summary",
    "summarry": "summary",
    "tomorow": "tomorrow",
    "tomarrow": "tomorrow",
    "tommorow": "tomorrow",
    "tommorrow": "tomorrow",
    "todays": "today",
    "tonite": "tonight",
    "apointment": "appointment",
    "appontment": "appointment",
    "appointmant": "appointment",
    "scheduel": "schedule",
    "schedual": "schedule",
    "scedule": "schedule",
    "meating": "meeting",
    "meetng": "meeting",
    "groceris": "groceries",
    "grocery": "groceries",
    "medicin": "medicine",
    "medicne": "medicine",
    "wensday": "wednesday",
    "thrusday": "thursday",
    "thurday": "thursday",
    "tusday": "tuesday",
    "fridy": "friday",
    "mondy": "monday",
    "weakend": "weekend",
    "notez": "note",
    "tasl": "task",
    "taks": "task",
    "delet": "delete",
    "remov": "remove",
    "updat": "update",
    "reschedule": "reschedule",
    "pleas": "please",
    "wat": "what",
    "wht": "what",
    "shw": "show",
    "lis": "list",
}

ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
}
NUM = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fourty|fifty)(?:[\s-](?:one|two|three|four|five|six|"
    r"seven|eight|nine))?"
)


def _distance(left, right):
    """Levenshtein distance, bounded so long inputs stay cheap."""
    if abs(len(left) - len(right)) > 2:
        return 3
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
        if min(previous) > 2:
            return 3
    return previous[-1]


def _suggest(token):
    """Suggest a domain word for a probable misspelling, or None.

    The explicit map is the only path for short or capitalised tokens. The fuzzy
    path additionally refuses any token that is already a common English word
    (checked against the committed public vocabulary), which is what stops
    "eight" becoming "night" or "text" becoming "next".
    """
    lowered = token.lower()
    if lowered in KNOWN_MISSPELLINGS:
        return KNOWN_MISSPELLINGS[lowered]
    if lowered in DOMAIN_WORDS or len(lowered) < 5 or lowered in _english():
        return None
    limit = 1 if len(lowered) <= 7 else 2
    best = None
    for word in DOMAIN_WORDS:
        score = _distance(lowered, word)
        if score <= limit and (best is None or score < best[1]):
            best = (word, score)
    return best[0] if best else None


@lru_cache(maxsize=1)
def _english():
    from fl.text_dp import vocabulary

    return frozenset(vocabulary())


def correct_spelling(text):
    """Fix recognised domain misspellings. Returns (text, [changes])."""
    changes = []
    words = text.split()
    for index, word in enumerate(words):
        core = word.strip(".,;:;!?()\"'")
        if not core or not core.isalpha():
            continue
        if core[0].isupper() and core.lower() not in KNOWN_MISSPELLINGS:
            # Capitalised words may be names; never guess at them.
            continue
        replacement = _suggest(core)
        if not replacement or replacement == core.lower():
            continue
        words[index] = word.replace(core, replacement)
        changes.append({"from": core, "to": replacement, "kind": "spelling"})
    return " ".join(words), changes


def _value(spoken):
    spoken = spoken.replace("-", " ").strip()
    parts = spoken.split()
    if not parts:
        return None
    total = 0
    for part in parts:
        if part in TENS:
            total += TENS[part]
        elif part in ONES:
            total += ONES[part]
        else:
            return None
    return total


def _hour(spoken):
    value = _value(spoken)
    return value if value is not None and 0 <= value <= 24 else None


def spoken_numbers(text):
    """Rewrite spoken numbers inside time/duration phrases. Returns (text, [changes])."""
    changes = []
    result = text

    def rewrite(pattern, handler, flags=re.I):
        nonlocal result

        def apply(match):
            replacement = handler(match)
            if replacement is None:
                return match.group()
            changes.append(
                {"from": match.group().strip(), "to": replacement.strip(), "kind": "time"}
            )
            return replacement

        result = re.sub(pattern, apply, result, flags=flags)

    def clock(hour, minute):
        return f"{hour}:{minute:02d}"

    # half past seven / a quarter past seven / quarter to seven
    def relative_hour(match):
        hour = _hour(match.group("hour"))
        if hour is None:
            return None
        kind = match.group("kind").lower()
        if kind.startswith("half"):
            return clock(hour, 30)
        if " to " in " " + kind + " ":
            return clock((hour - 1) % 24, 45)
        return clock(hour, 15)

    rewrite(
        r"\b(?:a\s+)?(?P<kind>half past|quarter past|quarter to)\s+(?P<hour>" + NUM + r")\b",
        relative_hour,
    )

    # twenty five past nine -> 9:25 ; ten to six -> 5:50
    def minutes_relative(match):
        minute = _value(match.group("minute"))
        hour = _hour(match.group("hour"))
        if minute is None or hour is None or minute > 59:
            return None
        if match.group("rel").lower() == "past":
            return clock(hour, minute)
        return clock((hour - 1) % 24, 60 - minute)

    rewrite(
        r"\b(?P<minute>" + NUM + r")(?:\s+minutes?)?\s+(?P<rel>past|to)\s+(?P<hour>" + NUM + r")\b",
        minutes_relative,
    )

    # seven thirty pm -> 7:30 pm ; at seven thirty -> at 7:30
    def hour_minute(match):
        hour = _hour(match.group("hour"))
        minute = _value(match.group("minute"))
        if hour is None or minute is None or minute > 59:
            return None
        lead = match.groupdict().get("lead")
        suffix = match.groupdict().get("meridiem")
        return (
            (f"{lead} " if lead else "")
            + clock(hour, minute)
            + (f" {suffix.lower()}" if suffix else "")
        )

    minute_words = (
        r"oh one|oh two|oh three|oh four|oh five|oh six|oh seven|oh eight|oh nine|"
        r"five|ten|fifteen|twenty|twenty one|twenty two|twenty three|twenty four|"
        r"twenty five|thirty|forty|forty one|forty two|forty three|forty four|"
        r"forty five|fifty|fifty one|fifty two|fifty three|fifty four|fifty five"
    )
    # A meridiem makes the pair unambiguous anywhere in the sentence.
    rewrite(
        r"\b(?P<hour>" + NUM + r")\s+(?P<minute>" + minute_words + r")"
        r"\s*(?P<meridiem>am|pm|a\.m\.|p\.m\.)\b",
        hour_minute,
    )
    # Without a meridiem, require a time preposition so "twenty apples" is safe.
    rewrite(
        r"\b(?P<lead>at|by|around|before|after)\s+(?P<hour>" + NUM + r")\s+"
        r"(?P<minute>" + minute_words + r")\b",
        hour_minute,
    )

    # seven in the morning / seven in the evening
    def part_of_day(match):
        hour = _hour(match.group("hour"))
        if hour is None:
            return None
        if match.group("part").lower() == "morning":
            return f"{hour} am"
        return clock(hour + 12 if hour < 12 else hour, 0)

    rewrite(
        r"\b(?P<hour>" + NUM + r")\s+in the (?P<part>morning|afternoon|evening|night)\b",
        part_of_day,
    )

    # seven pm / seven o'clock
    def meridiem_hour(match):
        hour = _hour(match.group("hour"))
        if hour is None:
            return None
        suffix = match.group("meridiem")
        if suffix.lower().startswith("o"):
            return clock(hour, 0)
        return f"{hour} {suffix.lower()}"

    rewrite(
        r"\b(?P<hour>" + NUM + r")\s*(?P<meridiem>am|pm|a\.m\.|p\.m\.|o'?clock)\b",
        meridiem_hour,
    )

    # at seven / by seven -> at 7:00 (a bare spoken hour reads as a clock time)
    def bare_hour(match):
        hour = _hour(match.group("hour"))
        if hour is None or hour > 23:
            return None
        return f"{match.group('lead')} {clock(hour, 0)}"

    rewrite(
        r"\b(?P<lead>at|by|around|before|after)\s+(?P<hour>" + NUM + r")\b(?!\s*"
        r"(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|people|times|items?|of)\b)",
        bare_hour,
    )

    # in twenty minutes / for three days / seven hours
    def quantity(match):
        value = _value(match.group("value"))
        if value is None:
            return None
        prefix = (match.group("lead") + " ") if match.group("lead") else ""
        return f"{prefix}{value} {match.group('unit').lower()}"

    rewrite(
        r"\b(?P<lead>in|for|after|within)?\s*(?P<value>" + NUM + r")\s+"
        r"(?P<unit>minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b",
        quantity,
    )
    return result, changes


def normalize(text):
    """Spelling tolerance + spoken numbers. Returns the report used by the API."""
    spelled, spelling = correct_spelling(text)
    rewritten, times = spoken_numbers(spelled)
    rewritten = re.sub(r"[ \t]{2,}", " ", rewritten).strip()
    notes = [
        f"Read “{change['from']}” as “{change['to']}”"
        for change in spelling + times
    ]
    return {
        "text": rewritten,
        "corrections": spelling,
        "rewrites": times,
        "notes": notes,
        "changed": bool(spelling or times),
    }
