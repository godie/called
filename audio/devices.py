"""Audio device enumeration and selection."""

import logging
from typing import Optional

import sounddevice as sd

logger = logging.getLogger("realtime-transcriber.devices")


def list_input_devices() -> list[dict]:
    """List all available input audio devices.

    Returns:
        List of device info dictionaries for input-capable devices.
    """
    devices: list[dict] = []
    for idx, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            devices.append({
                "index": idx,
                "name": device["name"],
                "channels": device["max_input_channels"],
                "default_samplerate": device["default_samplerate"],
                "hostapi": device["hostapi"],
                "is_default": device.get("isdefault", False),
            })
    return devices


def find_device_by_name(name_substring: str) -> Optional[int]:
    """Find a device index by partial name match.

    Args:
        name_substring: Part of the device name to search for.

    Returns:
        Device index if found, None otherwise.
    """
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if name_substring.lower() in device["name"].lower():
            logger.info("Found device '%s' at index %d", device["name"], idx)
            return idx
    logger.warning("No device found matching '%s'", name_substring)
    return None


def get_default_input_device() -> Optional[int]:
    """Get the default input device index.

    Returns:
        Index of the default input device, or None if not found.
    """
    try:
        default = sd.default.device[0]
        if default is not None:
            return default
    except Exception:
        pass

    # Fallback: find first input device
    devices = list_input_devices()
    if devices:
        return devices[0]["index"]
    return None


def validate_device(device_index: int, sample_rate: int, channels: int) -> bool:
    """Check if a device supports the required format.

    Args:
        device_index: Device index to validate.
        sample_rate: Required sample rate.
        channels: Required number of channels.

    Returns:
        True if the device is valid.
    """
    try:
        devices = sd.query_devices()
        if device_index >= len(devices):
            logger.error("Device index %d out of range", device_index)
            return False

        device = devices[device_index]
        if device["max_input_channels"] < channels:
            logger.error(
                "Device '%s' has %d input channels, need %d",
                device["name"],
                device["max_input_channels"],
                channels,
            )
            return False

        logger.info(
            "Device '%s' validated (sample_rate=%d, channels=%d)",
            device["name"],
            sample_rate,
            channels,
        )
        return True
    except Exception as exc:
        logger.error("Failed to validate device %d: %s", device_index, exc)
        return False


def print_available_devices() -> None:
    """Print all available audio devices to stdout."""
    devices = sd.query_devices()
    print("\n=== Available Audio Devices ===\n")
    for idx, device in enumerate(devices):
        print(f"  [{idx}] {device['name']}")
        print(f"       Input channels:  {device['max_input_channels']}")
        print(f"       Output channels: {device['max_output_channels']}")
        print(f"       Default rate:    {device['default_samplerate']}")
        print(f"       Host API:        {device['hostapi']}")
        print()
