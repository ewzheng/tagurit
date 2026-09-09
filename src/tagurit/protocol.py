"""Data contract shared by every pipeline stage.

Defines what an image, its tags, and its priority look like in memory, and how
they are encoded on the wire to the cloudlet. Every other module talks to its
neighbours through the types defined here. Keep this file free of ML imports.
"""
