"""MQTT sensors -> unchanged C2 pipeline -> local TFLite -> MQTT + SQLite."""

import json
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

from inference.processor import Processor
from inference.runtime import LocalModel
from inference.storage import EdgeStore,canonical,strict_json
from inference.uplink import deliver_once


def main():
    import paho.mqtt.client as mqtt
    truck=os.environ.get("TRUCK_ID","TRUCK_001")
    store=EdgeStore(os.environ.get("EDGE_DB","/state/edge.sqlite"))
    model=LocalModel(os.environ.get("MODEL_PATH","/app/model.tflite"),
                     os.environ.get("STATS_PATH","/app/data_pipeline/training_stats.npy"),
                     os.environ.get("MODEL_REGISTRY","/app/runtime_registry.json"))
    processor=Processor(model,store,truck)
    stop=threading.Event();connected=threading.Event();fatal=[];incoming=queue.Queue(maxsize=1024)
    health_path=Path(os.environ.get("HEALTH_PATH","/state/health.json"))
    client=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,client_id=f"logibridge-inference-{truck}",
                       clean_session=False,protocol=mqtt.MQTTv311,manual_ack=True)
    client.reconnect_delay_set(min_delay=1,max_delay=5)
    prefix=f"logibridge/trucks/{truck}"

    def on_connect(client,userdata,flags,reason_code,properties):
        if reason_code.is_failure:
            print("MQTT connection rejected:",reason_code,flush=True);return
        client.subscribe([(f"{prefix}/{stream}",1) for stream in ("temperature","vibration_rms","door_event")])
        connected.set();print("MQTT sensors subscribed on localhost; QoS 1, manual consumer acknowledgement",flush=True)

    def on_disconnect(client,userdata,flags,reason_code,properties):connected.clear()

    def on_message(client,userdata,message):
        if len(message.payload)>8192:
            print("Rejected oversized sensor message",flush=True);client.ack(message.mid,message.qos);return
        try:incoming.put_nowait(message)
        except queue.Full:
            fatal.append("Sensor processing queue is full; leaving unacknowledged broker messages for restart")
            stop.set()

    client.on_connect=on_connect;client.on_disconnect=on_disconnect;client.on_message=on_message

    def publisher(event):
        topic=f"{prefix}/"+("inference" if event["kind"]=="inference" else "alerts")
        if event["kind"]=="alert":
            print("[LOGIBRIDGE LOCAL ALERT] "+canonical(event),flush=True)
        else:
            print(f"[INFERENCE] model={event['model_label']} operational={event['operational_label']} "
                  f"variant={model.variant} window_end={event['window_end_unix']:.3f} "
                  f"clipped_features={event['clipped_feature_count']}",flush=True)
        info=client.publish(topic,canonical(event),qos=1,retain=False)
        if info.rc==mqtt.MQTT_ERR_SUCCESS:
            info.wait_for_publish(timeout=2)
            if not info.is_published():print("Local MQTT result is still unacknowledged; SQLite record is durable",flush=True)
        else:print("Local MQTT publish failed; SQLite record is durable:",info.rc,flush=True)

    def worker():
        try:
            while not stop.is_set():
                try:message=incoming.get(timeout=.5)
                except queue.Empty:continue
                try:
                    record=strict_json(message.payload)
                    if not isinstance(record,dict):raise ValueError("Sensor payload must be an object")
                    events=processor.consume(record,message.topic)
                except (ValueError,KeyError,TypeError) as error:
                    print("Rejected sensor record:",error,flush=True);events=[]
                for event in events:publisher(event)
                # Processing/SQLite commits happen before this consumer ACK.
                client.ack(message.mid,message.qos)
                incoming.task_done()
        except Exception as error:
            fatal.append("Inference worker failed: "+repr(error));stop.set()

    def sync_worker():
        url=os.environ.get("OPERATIONS_URL","http://operations:8080/events")
        try:
            while not stop.is_set():
                result=deliver_once(store,url)
                if result and result["status"]=="acknowledged":
                    print("[UPLINK ACK] "+result["event_id"],flush=True)
                elif result:
                    print("[UPLINK PENDING] "+result["event_id"]+" | "+result["error"],flush=True)
                stop.wait(.1 if result and result["status"]=="acknowledged" else .5)
        except Exception as error:
            fatal.append("Durable uplink worker failed: "+repr(error));stop.set()

    def shutdown(*args):stop.set()
    signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
    client.connect_async(os.environ.get("MQTT_HOST","localhost"),int(os.environ.get("MQTT_PORT","1883")),keepalive=30)
    client.loop_start()
    workers=[threading.Thread(target=worker,daemon=True),threading.Thread(target=sync_worker,daemon=True)]
    for thread in workers:thread.start()
    print(f"Model ready: {model.variant} | SHA256={model.sha256} | six fixed normalized features | built-in CPU / one thread",flush=True)
    try:
        while not stop.is_set():
            health={"updated_unix":time.time(),"model_ready":True,"mqtt_connected":connected.is_set(),
                    "variant":model.variant,"model_sha256":model.sha256,"records":processor.records,
                    "windows":processor.windows,"current_run_id":processor.run_id,"pending_input":incoming.qsize(),
                    "current_run_prediction_ready":processor.prediction_ready,
                    "rejected_feature_windows":len(processor.pipeline.rejected_windows) if processor.pipeline else 0,
                    "operational_severity":processor.severity,**store.stats()}
            temp=health_path.with_suffix(".tmp");temp.write_text(canonical(health));temp.replace(health_path)
            stop.wait(2)
    finally:
        stop.set()
        for thread in workers:thread.join(timeout=3)
        client.disconnect();client.loop_stop()
    if fatal:
        print("Service stopped:",fatal,file=sys.stderr,flush=True);return 1
    return 0


if __name__=="__main__":raise SystemExit(main())
