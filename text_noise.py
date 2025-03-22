import random
import string

# Interactively add noise to text
if __name__ == "__main__":
    print("Type text to add noise to it. Type /exit to exit.")

    # Probabilities of each type of character noise
    randomLetterProb = 0.07
    randomDigitProb = 0.03
    randomPunctuationProb = 0.02

    while True:
        text = input("> ")
        if text == "/exit":
            break

        for i in range(len(text)):
            randomVariable = random.random()

            if randomVariable < randomLetterProb:
                text = text[:i] + random.choice(string.ascii_letters) + text[(i + 1) :]
            elif randomVariable < randomLetterProb + randomDigitProb:
                text = text[:i] + random.choice(string.digits) + text[(i + 1) :]
            elif (
                randomVariable
                < randomLetterProb + randomDigitProb + randomPunctuationProb
            ):
                text = text[:i] + random.choice(string.punctuation) + text[(i + 1) :]

        print(text + "\n")
