"""
LLM Safety Alert Generation
After command classification, an LLM generates a structured, human-readable safety alert
conditioned on the predicted command and its confidence score.
"""

ALERT_PROMPT_TEMPLATE = """\
You are a safety alert system in an industrial facility.
A speech command has been detected under extreme noise conditions.

Detected Command : {command}
Confidence Score : {confidence:.2f} / 1.00
Noise Level      : {snr_db} dB SNR
Sensor State     : {sensor_state}

Generate a structured safety alert with:
- Severity level (CRITICAL / WARNING / INFO)
- Recommended immediate action
- Verification flag if confidence < 0.5
"""


def generate_alert(
    command: str,
    confidence: float,
    snr_db: float,
    sensor_state: str,
    llm_client,
    max_tokens: int = 300,
) -> str:
    """
    Generate a structured safety alert via an LLM.

    Args:
        command:      predicted command string
        confidence:   0.0–1.0 calibrated confidence score
        snr_db:       estimated signal-to-noise ratio in dB
        sensor_state: human-readable summary of current sensor readings
        llm_client:   any client with a .complete(prompt, max_tokens) method
                      (e.g., OpenAI, llama-cpp wrapper)
        max_tokens:   maximum tokens for LLM response
    Returns:
        Structured alert string
    """
    prompt = ALERT_PROMPT_TEMPLATE.format(
        command=command,
        confidence=confidence,
        snr_db=snr_db,
        sensor_state=sensor_state,
    )
    return llm_client.complete(prompt, max_tokens=max_tokens)
