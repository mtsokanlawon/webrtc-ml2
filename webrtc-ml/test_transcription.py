import base64
import threading
import subprocess
from queue import Queue, Empty
from datetime import datetime

from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingParameters,
    StreamingEvents,
    BeginEvent,
    TurnEvent,
    TerminationEvent,
    StreamingError,
)

ASSEMBLYAI_API_KEY = "433a0641544a46cfab94eb5b44ec4f1f"
WAV_FILE = "sample.wav"   # must be mono, 16kHz PCM16

def test_transcription():
    transcripts = []
    audio_queue = Queue()

    client = StreamingClient(
        StreamingClientOptions(
            api_key=ASSEMBLYAI_API_KEY,
            api_host="streaming.assemblyai.com",
        )
    )

    # --- Event Handlers ---
    def on_begin(client, event: BeginEvent):
        print(f"🔗 Session started: {event.id}")

    def on_turn(client, event: TurnEvent):
        print(f"📝 TURN RAW: {event}")
        if event.transcript and event.end_of_turn:
            print(f"✅ Transcript: {event.transcript}")
            transcripts.append(event.transcript)

    def on_terminated(client, event: TerminationEvent):
        print(f"🔚 Session terminated: {event.audio_duration_seconds} sec processed")

    def on_error(client, error: StreamingError):
        print(f"❌ Error: {error}")

    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    # --- Streaming loop ---
    def streaming_loop():
        client.connect(
            StreamingParameters(
                sample_rate=16000,
                format_turns=True,
            )
        )

        def generator():
            with open(WAV_FILE, "rb") as f:
                while True:
                    chunk = f.read(3200)  # 100ms chunks
                    if not chunk:
                        break
                    print(f"Sending {len(chunk)} bytes")
                    yield chunk

        client.stream(generator())
        client.disconnect(terminate=True)

    thread = threading.Thread(target=streaming_loop, daemon=True)
    thread.start()
    thread.join()

    return transcripts


if __name__ == "__main__":
    results = test_transcription()
    print("\nFinal transcripts:", results)
