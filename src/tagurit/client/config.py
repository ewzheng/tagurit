"""
Configure client scheduling and Gabriel communication
"""

# Set how many turns each queue gets during reintegration
LIVE_FRAME_WEIGHT = 2
STORED_FRAME_WEIGHT = 1

# Move one waiting live frame into the bank after each disconnected interval
LIVE_TO_BANK_INTERVAL_SECONDS = 5.0

# Set the minimum time between image submissions
SEND_INTERVAL_SECONDS = 0.0

# Maximum reserved images, including encoding and awaiting receipts
INFLIGHT_WINDOW = 2

# Set how often the client checks for work and timeouts
LOOP_INTERVAL_SECONDS = 0.02

# Set the Gabriel server and the engine that receives our images
GABRIEL_ENDPOINT = "ws://localhost:9099"
GABRIEL_ENGINE_ID = "image_receiver"
GABRIEL_PRODUCER_NAME = "images"

# Transmit each selected JPEG as an independent H.264 frame
IMAGE_CODEC = "h264"
VIDEO_CRF = 32

# Set how long to wait for startup and image receipts
CONNECTION_TIMEOUT_SECONDS = 10.0
RECEIPT_TIMEOUT_SECONDS = 10.0

# Wait between failed connection attempts
RECONNECT_INTERVAL_SECONDS = 2.0
