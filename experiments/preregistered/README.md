# Frozen preregistrations

Immutable manifests created by `experiments/preregister.py` live here. Each
contains the complete protocol, its SHA-256 fingerprint, the exact clean Git
revision, freeze time, and optional public registration URL.

Never edit a frozen manifest. Amendments receive a new study identifier and a
new manifest, with the reason documented before seeing amended confirmatory
outcomes.

The source revision is the clean commit containing the implementation and
candidate protocol. The manifest itself may be added in one later
manifest-only commit; the confirmatory runner rejects any other changed path.
