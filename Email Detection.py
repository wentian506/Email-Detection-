
from transformers import pipeline

print("Loading spam detection model...")

# Pre-trained spam detection model
model = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection"
)

print("Spam detector is ready!")


def detect_spam(email_text):

    if not email_text.strip():
        print("Please enter some email text.")
        return

    # Split long emails into smaller sections
    chunks = [
        email_text[i:i + 1500]
        for i in range(0, len(email_text), 1500)
    ]

    spam_scores = []

    for chunk in chunks:
        result = model(
            chunk,
            truncation=True,
            max_length=512
        )[0]

        label = result["label"].upper()
        score = result["score"]

        # LABEL_1 = spam, LABEL_0 = not spam
        if label in ("LABEL_1", "SPAM"):
            spam_scores.append(score)
        else:
            spam_scores.append(1 - score)

    # Average spam score across all sections
    spam_probability = sum(spam_scores) / len(spam_scores)

    print("\n========== RESULT ==========")

    if spam_probability >= 0.5:
        print("Email Status: SPAM")
    else:
        print("Email Status: NOT SPAM")

    print(f"Spam Score: {spam_probability * 100:.2f}%")
    print("============================")


def main():

    print("\n===== EMAIL SPAM DETECTOR =====")
    print("Paste your complete email below.")
    print("Type END on a new line to check it.")
    print("Type EXIT on a new line to close.\n")

    while True:

        email_lines = []

        while True:
            try:
                line = input()
            except EOFError:
                return

            if line.strip().upper() == "END":
                break

            if line.strip().upper() == "EXIT":
                print("Program closed.")
                return

            email_lines.append(line)

        email_text = "\n".join(email_lines)

        detect_spam(email_text)

        print("\nPaste another email or type EXIT.")


if __name__ == "__main__":
    main()
