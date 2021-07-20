import unicodedata

texts = ['क्ष, त्र, ज्ञ और श्र हिन्दी के संयुक्त व्यंजन हैं', 'ता']
normalizetion_type = 'NFKC'
for text in texts:
    noramalized_text = unicodedata.normalize(normalizetion_type, text)
    print("Unnormalized text")
    for char in text:
        print(char, unicodedata.name(char))
    print("Normalized text")
    for char in noramalized_text:
        print(char, unicodedata.name(char))
