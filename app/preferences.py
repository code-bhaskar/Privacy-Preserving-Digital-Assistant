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


def affordable(conn, user, epsilon=EPSILON_PER_ROUND, delta=DELTA_PER_ROUND):
    """Can this account pay for one more release of the given cost?"""
    spent_epsilon, spent_delta = expenditure(conn, user["id"])
    return (
        spent_epsilon + epsilon <= prefs(user)["epsilon"] + 1e-12
        and spent_delta + delta <= DELTA_CAP + 1e-12
    )
