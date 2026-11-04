"""Shared C2 preprocessing, model inference and explicit temperature safety."""

import math
import re
import time

from data_pipeline.preprocessing import FeaturePipeline,FEATURE_NAMES
from inference.runtime import LABELS


class Processor:
    def __init__(self,model,store,truck_id,clock=time.time):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",truck_id):raise ValueError("Invalid truck ID")
        self.model,self.store,self.truck_id,self.clock=model,store,truck_id,clock
        self.pipeline=None;self.run_id=None;self.origin=None
        state=store.get_state()
        self.last_model_class=state.get("last_model_class",0)
        self.severity=state.get("operational_severity",0)
        self.raw_temperature=None;self.last_heartbeat=state.get("last_heartbeat",0)
        self.prediction_ready=False
        self.records=0;self.windows=0

    def event(self,kind,event_id,**fields):
        return {"schema_version":1,"kind":kind,"event_id":event_id,"truck_id":self.truck_id,
                "run_id":self.run_id,"created_unix":self.clock(),"model_variant":self.model.variant,
                "model_sha256":self.model.sha256,**fields}

    def physical_severity(self):
        deviation=abs(self.raw_temperature-4.0) if self.raw_temperature is not None else 0
        return 2 if deviation>3 else (1 if deviation>1 else 0)

    def _safety(self,record_id,source_timestamp):
        physical=self.physical_severity()
        level=max(physical,self.last_model_class)
        if level==self.severity:return []
        self.severity=level
        reasons=[]
        if physical:reasons.append("temperature outside 4C setpoint by "+("more than 3C" if physical==2 else "more than 1C"))
        if self.last_model_class:reasons.append("latest model class "+LABELS[self.last_model_class])
        if not level:reasons.append("temperature and latest model class returned to Normal")
        event=self.event("alert",f"alert:{self.truck_id}:{record_id}:{level}",severity=level,severity_label=LABELS[level],
                         model_class=self.last_model_class if self.prediction_ready else None,
                         model_label=LABELS[self.last_model_class] if self.prediction_ready else "WARMING_UP",
                         latched_model_class=self.last_model_class,
                         latest_raw_temperature_c=self.raw_temperature,reasons=reasons,
                         source_timestamp_unix=source_timestamp,
                         source_to_local_alert_ms=1000*(self.clock()-source_timestamp))
        return [event] if self.store.add(event,remote=True) else []

    def consume(self,record,topic):
        if record.get("schema_version")!=1:raise ValueError("Unsupported sensor schema")
        stream=record.get("stream")
        if stream not in ("temperature","vibration_rms","door_event"):raise ValueError("Unknown sensor stream")
        if record.get("truck_id")!=self.truck_id or topic!=f"logibridge/trucks/{self.truck_id}/{stream}":
            raise ValueError("Sensor topic/payload truck or stream mismatch")
        timestamp=float(record["timestamp_unix"]);elapsed=float(record["elapsed_s"])
        if not math.isfinite(timestamp) or not math.isfinite(elapsed) or elapsed<0:raise ValueError("Invalid source time")
        run_id=record.get("run_id");message_id=record.get("message_id")
        if (not isinstance(run_id,str) or not 1<=len(run_id)<=128 or
                not isinstance(message_id,str) or not 1<=len(message_id)<=256):raise ValueError("Invalid message/run ID")
        if stream!="door_event" and (isinstance(record.get("value"),bool) or
                not isinstance(record.get("value"),(int,float)) or not math.isfinite(record["value"])):
            raise ValueError("Numeric sensor values must be finite")
        if run_id!=self.run_id:
            self.run_id=run_id;self.origin=timestamp-elapsed
            self.pipeline=FeaturePipeline(self.origin,strict=False)
            self.raw_temperature=None
            self.prediction_ready=False
            # Metadata only for a controlled-run SLA report; never an input to
            # filtering, model classification or the temperature safety rule.
            self.store.add(self.event("run_start",f"run:{self.truck_id}:{run_id}",origin_unix=self.origin,
                announced_simulator_mode=record.get("anomaly","unspecified")))
        before=self.pipeline.duplicates_ignored
        windows=self.pipeline.add_record(record)
        if self.pipeline.duplicates_ignored!=before:return []
        self.records+=1
        emitted=[]
        if stream=="temperature":
            self.raw_temperature=float(record["value"])
            emitted.extend(self._safety(message_id,timestamp))
        for window in windows:
            prediction=self.model.predict(window.features)
            self.last_model_class=prediction["model_class"]
            self.prediction_ready=True
            event=self.event("inference",f"inference:{self.truck_id}:{run_id}:{round(window.window_end_unix*1000)}",
                **prediction,model_label=LABELS[self.last_model_class],feature_names=list(FEATURE_NAMES),
                operational_severity=max(self.physical_severity(),self.last_model_class),
                operational_label=LABELS[max(self.physical_severity(),self.last_model_class)],
                raw_features=list(window.features),window_start_unix=window.window_start_unix,
                window_end_unix=window.window_end_unix,door_state=window.door_state_at_end,
                temperature_samples=window.temperature_samples,vibration_samples=window.vibration_samples,
                normality_confidence=prediction["probabilities"][0],confidence_max=max(prediction["probabilities"]))
            if self.store.add(event):emitted.append(event);self.windows+=1
            emitted.extend(self._safety(event["event_id"],window.raw_support_end_unix))
            if self.clock()-self.last_heartbeat>=60:
                heartbeat=self.event("heartbeat",f"heartbeat:{event['event_id']}",
                    operational_severity=self.severity,model_label=LABELS[self.last_model_class],
                    window_end_unix=window.window_end_unix)
                self.store.add(heartbeat,remote=True);self.last_heartbeat=self.clock()
        self.store.put_state({"last_model_class":self.last_model_class,"operational_severity":self.severity,
                              "last_heartbeat":self.last_heartbeat})
        return emitted
