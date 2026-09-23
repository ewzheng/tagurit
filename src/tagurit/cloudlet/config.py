"""
Configure the Gabriel image receiver
"""

# Set the port used by the Gabriel server
SERVER_PORT = 9099

# Name the engine that clients will target
ENGINE_ID = "image_receiver"

# Set the number of frames allowed in flight per producer
NUM_TOKENS = 1

# Set how many frames can wait inside the local Gabriel server
INPUT_QUEUE_MAXSIZE = 2