"""Laya question sets. Changing one changes the experiment, so version them by name."""

ZERO_SHOT_V1 = {
    "emotion": {
        "type": "choice",
        "instructions": "Which emotion does the speaker express in their line?",
        "criteria": {
            "neutral": "calm, matter-of-fact, no strong feeling",
            "joy": "happy, excited, amused, pleased",
            "surprise": "astonished, shocked, caught off guard",
            "anger": "annoyed, frustrated, irritated, furious",
            "sadness": "sad, hurt, disappointed, regretful",
            "disgust": "repulsed, grossed out, contemptuous",
            "fear": "scared, anxious, nervous, worried",
        },
    },
    "sentiment": {
        "type": "choice",
        "instructions": "What is the overall sentiment of the speaker's line?",
        "criteria": {"positive": "good feeling", "negative": "bad feeling", "neutral": "neither"},
    },
}


# Fine-tuning trains on exactly the zero-shot questions, so the only thing that changes between
# the baseline and the fine-tuned run is the weights.
MELD_V1 = ZERO_SHOT_V1


def state_for(utterance):
    """What Laya sees for one utterance: the previous line as context, then the line itself."""
    return {"previous_line": utterance["previous"], "speaker": utterance["speaker"],
            "line": utterance["text"]}
