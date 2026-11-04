"""Operations-centre simulator on the private Docker network."""

import argparse
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

from inference.storage import OperationsStore,canonical,strict_json


def make_server(host,port,store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass

        def reply(self,status,value):
            body=canonical(value).encode()
            self.send_response(status);self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)

        def do_GET(self):
            if self.path=="/health":self.reply(200,{"status":"ready"})
            elif self.path=="/stats":self.reply(200,store.stats())
            else:self.reply(404,{"error":"unknown endpoint"})

        def do_POST(self):
            if self.path!="/events":self.reply(404,{"error":"unknown endpoint"});return
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 1<=length<=32768:raise ValueError("Invalid body length")
                event=strict_json(self.rfile.read(length))
                if not isinstance(event,dict):raise ValueError("An event must be a JSON object")
                if self.headers.get("Idempotency-Key")!=event.get("event_id"):raise ValueError("Idempotency key mismatch")
                inserted=store.accept(event)
                self.reply(201 if inserted else 200,{"event_id":event["event_id"],"stored":True,"duplicate":not inserted})
            except (ValueError,KeyError,TypeError) as error:self.reply(409,{"error":str(error)})
    return ThreadingHTTPServer((host,port),Handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",default="/state/operations.sqlite")
    parser.add_argument("--host",default="0.0.0.0");parser.add_argument("--port",type=int,default=8080)
    args=parser.parse_args()
    server=make_server(args.host,args.port,OperationsStore(args.db))
    print("Operations-centre simulator ready; durable event IDs enabled",flush=True)
    server.serve_forever()


if __name__=="__main__":main()
