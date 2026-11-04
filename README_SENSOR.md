# LogiEdge sensor simulator — Component C1

## Verified progress before this checkpoint

The team's laptop has Windows 11, Ubuntu-24.04 running in WSL2, Docker and Compose available inside Ubuntu, Git 2.43.0, a Python 3.11.18 container, and the project directory /home/nandhini/logibridge. The user demonstrated that Mosquitto 2.0.22 is healthy and that a setup message is received. Project training and model evaluation have not started.

This archive adds simulator source, its Docker image recipe, a capture-inspection utility, offline signal tests, and an updated Compose file. The MQTT service configuration is preserved. The simulator service uses network_mode: service:mqtt; it therefore shares the broker's loopback interface and can connect to localhost:1883. No broker port is published to the surrounding network.

## Install on the demonstration laptop

Download LogiEdge_Sensor_Simulator.zip to C:\Users\Nandhini G\Downloads. Then, in Ubuntu, run the following commands one at a time:

```bash
cd ~/logibridge
sudo apt install -y unzip
cp compose.yaml compose.before-simulator.yaml
unzip -o "/mnt/c/Users/Nandhini G/Downloads/LogiEdge_Sensor_Simulator.zip" -d .
docker compose config -q
docker compose build simulator
```

The archive's files belong directly under the project root. It updates compose.yaml and adds files under data_pipeline, deployment, and tests.

The launcher uses your Ubuntu UID and GID so captures remain editable by your account. Call it using bash; an executable-bit change is unnecessary.

## First test: 45 real seconds of normal data

```bash
bash deployment/run_simulator.sh --anomaly none --duration 45 --seed 101
```

A healthy run should acknowledge about 45 temperature messages, 23 vibration messages, and 3 door events. The ideal counts are 45/23/3 because readings start at time zero; a heavily stalled run reports missed deadlines rather than creating backdated catch-up readings. Review any nonzero missed-deadline count before collecting training data.

Inspect the saved records:

```bash
docker compose run --rm --entrypoint python simulator -m data_pipeline.inspect_capture --latest
```

This prints actual saved counts, median sample intervals, signal means and standard deviations, door-state counts, and one example per stream. A small smoke run will not have exactly the target population means or standard deviations.

Expected median intervals are close to 1.000 seconds for temperature and 2.000 seconds for vibration. A 45-second run includes an initial CLOSE state, OPEN at 30 seconds, and CLOSE at 35 seconds.

## Optional live subscriber

In one Ubuntu terminal at the project root, start:

```bash
docker compose exec -T mqtt mosquitto_sub -h 127.0.0.1 -q 1 \
  -t 'logibridge/trucks/TRUCK_001/+' -v
```

Keep that terminal open. In a second Ubuntu terminal, run the simulator. The first terminal will display each published sensor reading. After the run, use Ctrl+C in the subscriber terminal to stop viewing.

## Remaining C1 checks, after the normal run works

Use distinct seeds and inspect each capture immediately after its run:

```bash
bash deployment/run_simulator.sh --anomaly temp_drift --duration 45 --seed 102
docker compose run --rm --entrypoint python simulator -m data_pipeline.inspect_capture --latest

bash deployment/run_simulator.sh --anomaly vibration --duration 45 --seed 103
docker compose run --rm --entrypoint python simulator -m data_pipeline.inspect_capture --latest

bash deployment/run_simulator.sh --anomaly combined --duration 45 --seed 104
docker compose run --rm --entrypoint python simulator -m data_pipeline.inspect_capture --latest
```

Do not start two simulators for the same truck simultaneously. Their messages would represent different runs on one sensor topic.

## Required behaviour and implementation

| Requirement | Implementation |
| --- | --- |
| Temperature at 1 Hz | One scheduled temperature reading every second |
| Normal temperature | Gaussian mean 4.0 degrees C, standard deviation 0.3 |
| Temperature drift | Additional 0.08 degrees C per generated temperature reading |
| Vibration RMS at 0.5 Hz | One scheduled RMS reading every two seconds |
| Normal vibration | Gaussian mean 0.45 g, standard deviation 0.05 |
| Vibration anomaly | Gaussian mean 1.2 g, standard deviation 0.15 |
| Door events | Timestamped OPEN/CLOSE; initial CLOSE, open every 30 seconds for 5 seconds |
| Anomaly CLI | none, temp_drift, vibration, combined |
| Local broker | localhost:1883 in the shared truck container network |
| Sensor topics | logibridge/trucks/TRUCK_001/temperature, vibration_rms, door_event |
| Delivery | MQTT 3.1.1, QoS 1, non-retained sensor messages |
| Traceability | Unique run/message IDs, per-stream sequence numbers, UTC and Unix timestamps |
| Saved data | JSONL readings plus a run manifest under data/raw |
| Reproducibility | Explicit seed, or a generated seed recorded in the manifest |

The door timing is a declared demonstration assumption: the brief specifies discrete events but does not prescribe an event frequency.

The anomaly and purpose fields are metadata. The later model must receive only the six specified numerical features. Door events will provide context; they are not a seventh model feature.

The manifest's counts describe messages acknowledged by the local MQTT broker and saved by the simulator. They do not certify delivery to an operations-centre backend.

## Data collection boundaries

These 45-second runs use purpose=smoke_test and are not the required training dataset. Later we will explicitly collect:
- 10 independent clean minutes for fixed normalization statistics;
- 20 Normal, 15 temp_drift, and 15 combined minutes for training scenarios;
- separate clean Normal windows for the PSI reference.

Use different seeds for independent captures and prevent overlapping-window leakage between training and validation. The simulator defaults to a newly generated recorded seed when --seed is omitted.

The brief's +0.08 degrees C per second eventually exceeds a 3-degree temperature deviation in about 38 seconds. Consequently the prescribed 15-minute temp_drift training scenario is not physically Warning throughout its duration. This implementation preserves the specified sensor behaviour and records the anomaly mode, without assigning a physical severity label. Resolve this interpretation when designing training labels and any safety override.

The 0.5 Hz vibration_rms stream implements C1. The report's hypothetical 500 Hz three-axis raw-bandwidth calculation is a separate requirement.

## Source walkthrough for the team

- SensorSignals generates the stochastic readings. Separate random generators avoid coupling temperature values to vibration sampling.
- event_schedule sets independent deadlines for 1 Hz, 0.5 Hz, and door events.
- MqttPublisher uses the Paho callback API version 2 and waits for the broker's QoS 1 acknowledgement.
- run attaches timestamps and IDs, publishes records, saves acknowledged readings, and writes a manifest.
- inspect_capture checks consistency and reports actual capture statistics.
- run_simulator.sh chooses the project directory and invokes Compose with your UID/GID.

Ganesan should be able to explain these pieces for the sensor-pipeline portion of the demonstration.

## Verification

Eight offline unit tests passed while preparing this package: normal signal distributions; exact drift and separation of anomaly modes; both vibration anomaly modes; sampling rates and door transitions; continuous/fractional-duration scheduling; CLI modes and UTC formatting; agreement between saved records and acknowledged counts; and failure handling for an unacknowledged reading. The two capture-bookkeeping tests use an in-memory publisher, not a real MQTT broker. Python source compilation, launcher syntax, and Compose YAML parsing were checked.

The authoring runtime has no Docker daemon. Live container build and Mosquitto publication must be confirmed on the team's laptop with the commands above. Existing broker success is established by the user's supplied terminal output.

Offline tests can also be run in the simulator image:

```bash
docker compose run --rm --entrypoint python simulator -m unittest discover -s tests -v
```

## Current rubric status

This package implements C1 source for verification. Complete sensor-pipeline evidence still requires all four live anomaly runs and the C2/C3 filtering, six-feature extraction, fixed normalization, and feature-fusion implementation. This simulator image is distinct from the later D2 inference-service image and OTA cache demonstration.

## Official references

- https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html
- https://docs.docker.com/reference/compose-file/services/
- https://docs.docker.com/compose/how-tos/profiles/
- https://hub.docker.com/_/eclipse-mosquitto

