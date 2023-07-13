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

def word_to_arpabet(word):
    word = word.lower().strip(".,")
    # If word not in arpabet_dict, use G2P
    if word not in arpabet_dict:
        arpabet = g2p(word)
        return [[phoneme] for phoneme in arpabet]
    return arpabet_dict.get(word, [])

audio = whisper.load_audio("audio.wav")
print("Loaded audio") # Debugging print statement

model = whisper.load_model("tiny", device="cpu")
print("Loaded model") # Debugging print statement

result = whisper.transcribe(model, audio, language="en")
print("Transcription result:", result) # Debugging print statement

# Modify the result to include ARPAbet
for segment in result["segments"]:
    for word in segment["words"]:
        word["arpabet"] = word_to_arpabet(word["text"])

json_result = json.dumps(result, indent = 2, ensure_ascii = False)
print("JSON result:", json_result) # Debugging print statement

# Save to a JSON file
with open("output.json", "w", encoding='utf-8') as json_file:
    json.dump(result, json_file, ensure_ascii=False, indent=2)
