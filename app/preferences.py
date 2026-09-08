import json
from sqlalchemy import select, func
from .database import ledger
from fl.privacy import EPSILON_PER_ROUND, DELTA_PER_ROUND, DELTA_CAP

DEFAULTS = {
    "assistant": False,
    "calendar": False,
    "notes": False,
    "summary": False,
    "training": False,
    "cloud": False,
    "epsilon": 3.0,
    "reduced": False,
    "timezone": "Asia/Kolkata",
    "mode": "Default",
}


def prefs(user):
    stored = json.loads(user["preferences"])
    return {key: stored.get(key, value) for key, value in DEFAULTS.items()}


def expenditure(conn, uid):
    values = conn.execute(
        select(
            func.coalesce(func.sum(ledger.c.epsilon), 0),
            func.coalesce(func.sum(ledger.c.delta), 0),
        ).where(ledger.c.user_id == uid)
    ).one()
    return float(values[0]), float(values[1])


def affordable(conn, user):
    epsilon, delta = expenditure(conn, user["id"])
    return (
        epsilon + EPSILON_PER_ROUND <= prefs(user)["epsilon"] + 1e-12
        and delta + DELTA_PER_ROUND <= DELTA_CAP + 1e-12
    )
