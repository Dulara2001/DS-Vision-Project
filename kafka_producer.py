# kafka_producer.py

import json
import threading
from datetime import datetime, timezone
from confluent_kafka import Producer
from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_SECURITY_PROTOCOL,
    KAFKA_SASL_MECHANISM,
    KAFKA_USERNAME,
    KAFKA_PASSWORD,
    KAFKA_TOPIC,
)

KAFKA_CONFIG = {
    "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
    "security.protocol": KAFKA_SECURITY_PROTOCOL,
    "sasl.mechanism":    KAFKA_SASL_MECHANISM,
    "sasl.username":     KAFKA_USERNAME,
    "sasl.password":     KAFKA_PASSWORD,
}

KAFKA_TOPIC = KAFKA_TOPIC

_producer = None
_producer_lock = threading.Lock()

def _get_producer():
    global _producer
    if _producer is None:
        with _producer_lock:
            if _producer is None:
                _producer = Producer(KAFKA_CONFIG)
                print("[Kafka] ✅ Producer connected.")
    return _producer

def _delivery_report(err, msg):
    if err:
        print(f"[Kafka] ❌ Delivery failed: {err}")
    else:
        print(f"[Kafka] ✅ Sent to partition {msg.partition()} offset {msg.offset()}")

def publish_detection(camera_id, global_id, gender=None, age=None, race=None, is_returning=False):
    """Fire and forget — never blocks the main pipeline."""
    threading.Thread(
        target=_publish_async,
        args=(camera_id, global_id, gender, age, race, is_returning),
        daemon=True
    ).start()

def _publish_async(camera_id, global_id, gender=None, age=None, race=None, is_returning=False):
    try:
        payload = {
            "cameraId":  camera_id,
            "id":        global_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "isReturning": "true" if is_returning else "false",
        }
        if gender is not None:
            payload["gender"] = gender   # "Male" / "Female"
        if race is not None:
            payload["race"] = race       # "Local" / "Foreigner"
        if age is not None:
            payload["age"] = age         # "Kid" / "Adult"

        producer = _get_producer()
        producer.produce(
            topic=KAFKA_TOPIC,
            value=json.dumps(payload).encode("utf-8"),
            on_delivery=_delivery_report
        )
        producer.poll(1)

    except Exception as e:
        print(f"[Kafka] ❌ Failed to publish: {e}")

def flush_producer():
    global _producer
    if _producer:
        print("[Kafka] Flushing remaining messages...")
        _producer.flush(timeout=10)
        print("[Kafka] ✅ Flush complete.")