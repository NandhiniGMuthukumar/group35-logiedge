"""Short, committed SQLite transactions; a separate connection for each call."""

import hashlib
from contextlib import contextmanager
import json
import sqlite3
import time
from pathlib import Path


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def strict_json(value):
    def reject(constant):raise ValueError("Nonfinite JSON constant: "+constant)
    return json.loads(value,parse_constant=reject)


class EdgeStore:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS events(
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS outbox(
                event_id TEXT PRIMARY KEY REFERENCES events(event_id), attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt REAL NOT NULL DEFAULT 0, last_error TEXT, acknowledged REAL);
            CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.row_factory=sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback();raise
        finally:db.close()

    def add(self,event,remote=False):
        payload=canonical(event)
        with self.connect() as db:
            cursor=db.execute("INSERT OR IGNORE INTO events(event_id,kind,payload,created) VALUES(?,?,?,?)",
                              (event["event_id"],event["kind"],payload,event["created_unix"]))
            inserted=cursor.rowcount==1
            if inserted and remote:
                db.execute("INSERT INTO outbox(event_id) VALUES(?)",(event["event_id"],))
        return inserted

    def put_state(self,value):
        with self.connect() as db:
            db.execute("INSERT INTO state(key,value) VALUES('processor',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                       (canonical(value),))

    def get_state(self):
        with self.connect() as db:row=db.execute("SELECT value FROM state WHERE key='processor'").fetchone()
        return json.loads(row[0]) if row else {}

    def pending(self,now=None):
        now=time.time() if now is None else now
        with self.connect() as db:
            row=db.execute("""SELECT e.payload,o.* FROM outbox o JOIN events e USING(event_id)
                WHERE o.acknowledged IS NULL ORDER BY e.sequence LIMIT 1""").fetchone()
        if row is None or row["next_attempt"]>now:return None
        return dict(row)

    def failed(self,event_id,error,now=None):
        now=time.time() if now is None else now
        with self.connect() as db:
            row=db.execute("SELECT attempts FROM outbox WHERE event_id=?",(event_id,)).fetchone()
            if row is None:raise ValueError("Unknown outbox event")
            attempts=row[0]+1
            delay=min(10,2**min(attempts-1,4))
            db.execute("UPDATE outbox SET attempts=?,next_attempt=?,last_error=? WHERE event_id=?",
                       (attempts,now+delay,str(error)[:500],event_id))

    def acknowledged(self,event_id,now=None):
        with self.connect() as db:
            db.execute("UPDATE outbox SET acknowledged=?,last_error=NULL WHERE event_id=?",
                       (time.time() if now is None else now,event_id))

    def events(self,kind=None,limit=2000):
        with self.connect() as db:
            rows=db.execute("SELECT payload FROM events "+("WHERE kind=? " if kind else "")+
                            "ORDER BY sequence DESC LIMIT ?",(kind,limit) if kind else (limit,)).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]

    def stats(self):
        with self.connect() as db:
            kinds={row[0]:row[1] for row in db.execute("SELECT kind,COUNT(*) FROM events GROUP BY kind")}
            pending=db.execute("SELECT COUNT(*) FROM outbox WHERE acknowledged IS NULL").fetchone()[0]
            delivered=db.execute("SELECT COUNT(*) FROM outbox WHERE acknowledged IS NOT NULL").fetchone()[0]
            attempts=db.execute("SELECT COALESCE(SUM(attempts),0) FROM outbox").fetchone()[0]
        return {"local_event_counts":kinds,"pending_uplink":pending,"acknowledged_uplink":delivered,
                "failed_uplink_attempts":attempts}


class OperationsStore:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS received(event_id TEXT PRIMARY KEY,payload TEXT NOT NULL,sha TEXT NOT NULL,received REAL NOT NULL)")

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.execute("PRAGMA journal_mode=WAL");db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback();raise
        finally:db.close()

    def accept(self,event):
        if (not isinstance(event,dict) or not isinstance(event.get("event_id"),str) or
                not 1<=len(event["event_id"])<=512 or event.get("kind") not in ("alert","heartbeat")):
            raise ValueError("Invalid operations event")
        payload=canonical(event);sha=hashlib.sha256(payload.encode()).hexdigest()
        with self.connect() as db:
            inserted=db.execute("INSERT OR IGNORE INTO received VALUES(?,?,?,?)",(event["event_id"],payload,sha,time.time())).rowcount==1
            existing=db.execute("SELECT sha FROM received WHERE event_id=?",(event["event_id"],)).fetchone()
            if existing[0]!=sha:raise ValueError("An event ID cannot refer to different immutable payloads")
        return inserted

    def stats(self):
        with self.connect() as db:
            rows=db.execute("SELECT payload FROM received ORDER BY received").fetchall()
        values=[json.loads(row[0]) for row in rows]
        return {"unique_received_events":len(values),"counts_by_kind":{
            kind:sum(v["kind"]==kind for v in values) for kind in ("alert","heartbeat")},"last_events":values[-10:]}
