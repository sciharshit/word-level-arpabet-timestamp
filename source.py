import whisper_timestamped as whisper
import nltk
from nltk.corpus import cmudict
import json
from g2p_en import G2p

# Download the cmudict corpus
nltk.download('cmudict')

# Load the ARPAbet dictionary
arpabet_dict = cmudict.dict()

# Initialize g2p model
g2p = G2p()

def word_to_arpabet(word, start_time, end_time):
    word = word.lower().strip(".,!-")
    arpabet = None
    # If word not in arpabet_dict, use G2P
    if word not in arpabet_dict:
        arpabet = g2p(word)
    else:
        # The arpabet dictionary returns a list of lists where each sub-list is a different pronunciation
        # This uses the first pronunciation only
        arpabet = arpabet_dict.get(word, [[]])[0]
    return [[phoneme, calculate_phoneme_time(phoneme, start_time, end_time, len(arpabet), i)] for i, phoneme in enumerate(arpabet)]


def calculate_phoneme_time(phoneme, start_time, end_time, total_phonemes, phoneme_index):
    if total_phonemes == 0:
        return {"start": start_time, "end": end_time}
    total_duration = end_time - start_time
    phoneme_duration = total_duration / total_phonemes
    phoneme_start = start_time + (phoneme_index * phoneme_duration)
    phoneme_end = phoneme_start + phoneme_duration
    return {"start": phoneme_start, "end": phoneme_end}

audio = whisper.load_audio("audio.wav")
print("Loaded audio")

model = whisper.load_model("tiny", device="cpu")
print("Loaded model")

result = whisper.transcribe(model, audio, language="en")
print("Transcription result:", result)

for segment in result["segments"]:
    for word in segment["words"]:
        word["arpabet"] = word_to_arpabet(word["text"], word["start"], word["end"])

json_result = json.dumps(result, indent = 2, ensure_ascii = False)
print("JSON result:", json_result)

# Save to a JSON file
with open("output1.json", "w", encoding='utf-8') as json_file:
    json.dump(result, json_file, ensure_ascii=False, indent=2)
