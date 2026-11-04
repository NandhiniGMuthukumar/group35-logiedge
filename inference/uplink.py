"""At-least-once HTTP delivery from an immutable, persistent FIFO outbox."""

import json
import urllib.request


def deliver_once(store,url,opener=None,now=None):
    row=store.pending(now)
    if row is None:return None
    request=urllib.request.Request(url,data=row["payload"].encode(),method="POST",
                                   headers={"Content-Type":"application/json","Idempotency-Key":row["event_id"]})
    # This is a private Docker-network endpoint. Do not route cargo events
    # through a host's ambient HTTP proxy configuration.
    opener=opener or urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    try:
        with opener(request,timeout=2) as response:
            result=json.loads(response.read(65536))
            if response.status not in (200,201) or result.get("event_id")!=row["event_id"] or result.get("stored") is not True:
                raise ValueError("Operations centre did not acknowledge this event ID")
    except Exception as error:
        store.failed(row["event_id"],error,now)
        return {"event_id":row["event_id"],"status":"pending","error":str(error)}
    store.acknowledged(row["event_id"],now)
    return {"event_id":row["event_id"],"status":"acknowledged"}
