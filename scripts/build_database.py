"""Builds the demo operations database (data/ops.db) from a fixed random seed.

The data is synthetic: fictional customers, orders and support tickets for a
mid-sized B2B supplier. A fixed seed keeps every run identical, so the eval
results in the README can be reproduced.
"""

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "ops.db"

SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    tier TEXT NOT NULL CHECK (tier IN ('Enterprise', 'Business', 'Standard')),
    country TEXT NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TEXT NOT NULL,
    amount_eur REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('delivered', 'shipped', 'processing', 'refunded'))
);

CREATE TABLE tickets (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    opened_at TEXT NOT NULL,
    priority TEXT NOT NULL CHECK (priority IN ('high', 'medium', 'low')),
    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    subject TEXT NOT NULL
);
"""

CUSTOMERS = [
    ("Helios Energy GmbH", "Enterprise", "Germany"),
    ("Nordlicht Logistics", "Enterprise", "Germany"),
    ("Vantara Medical", "Enterprise", "Austria"),
    ("Brightfield Retail", "Business", "Netherlands"),
    ("Alpenblick Hotels", "Business", "Austria"),
    ("Corvid Analytics", "Business", "Germany"),
    ("Lumen Print Studio", "Business", "France"),
    ("Tessera Foods", "Business", "Italy"),
    ("Kalmar Tools", "Standard", "Sweden"),
    ("Isar Bikes", "Standard", "Germany"),
    ("Pomona Garden Supply", "Standard", "Belgium"),
    ("Quill & Co", "Standard", "Ireland"),
    ("Ostara Ceramics", "Standard", "Germany"),
    ("Redwood Dental", "Business", "Germany"),
    ("Silbersee Brewing", "Standard", "Germany"),
    ("Mistral Aero Parts", "Enterprise", "France"),
]

SUBJECTS = [
    "Delivery arrived damaged",
    "Invoice amount does not match order",
    "Request for refund",
    "Tracking shows no movement",
    "Need express delivery for next order",
    "Question about volume discount",
    "Wrong item delivered",
    "Cannot log in to customer portal",
    "Change delivery address",
    "Missing items in shipment",
]


def build(db_path: Path = DB_PATH) -> Path:
    rng = random.Random(7)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    con.executemany(
        "INSERT INTO customers (id, name, tier, country) VALUES (?, ?, ?, ?)",
        [(i + 1, *c) for i, c in enumerate(CUSTOMERS)],
    )

    start = date(2025, 1, 1)
    order_rows = []
    for order_id in range(1, 401):
        cust_id = rng.randint(1, len(CUSTOMERS))
        tier = CUSTOMERS[cust_id - 1][1]
        base = {"Enterprise": 4200, "Business": 1500, "Standard": 400}[tier]
        amount = round(rng.uniform(0.3, 1.8) * base, 2)
        day = start + timedelta(days=rng.randint(0, 600))
        status = rng.choices(
            ["delivered", "shipped", "processing", "refunded"], weights=[70, 12, 10, 8]
        )[0]
        order_rows.append((order_id, cust_id, day.isoformat(), amount, status))
    con.executemany(
        "INSERT INTO orders (id, customer_id, order_date, amount_eur, status) VALUES (?, ?, ?, ?, ?)",
        order_rows,
    )

    ticket_rows = []
    for ticket_id in range(1, 161):
        cust_id = rng.randint(1, len(CUSTOMERS))
        day = start + timedelta(days=rng.randint(200, 630))
        priority = rng.choices(["high", "medium", "low"], weights=[25, 45, 30])[0]
        status = rng.choices(["open", "closed"], weights=[30, 70])[0]
        ticket_rows.append(
            (ticket_id, cust_id, day.isoformat(), priority, status, rng.choice(SUBJECTS))
        )
    con.executemany(
        "INSERT INTO tickets (id, customer_id, opened_at, priority, status, subject) VALUES (?, ?, ?, ?, ?, ?)",
        ticket_rows,
    )
    con.commit()
    con.close()
    return db_path


if __name__ == "__main__":
    path = build()
    print(f"Built {path}")
