"""
Configure client scheduling and Gabriel communication
"""

# Set how many turns each queue gets during reintegration
LIVE_FRAME_WEIGHT = 2
STORED_FRAME_WEIGHT = 1

# Set the minimum time between image submissions
SEND_INTERVAL_SECONDS = 0.75

# Set how often the transport checks for work and timeouts
LOOP_INTERVAL_SECONDS = 0.02

# Set the Gabriel server and the engine that receives our images
GABRIEL_ENDPOINT = "ws://localhost:9099"
GABRIEL_ENGINE_ID = "image_receiver"
GABRIEL_PRODUCER_NAME = "images"

# Set how long to wait for startup and image receipts
CONNECTION_TIMEOUT_SECONDS = 10.0
RECEIPT_TIMEOUT_SECONDS = 10.0

# Wait between failed connection attempts
RECONNECT_INTERVAL_SECONDS = 2.0