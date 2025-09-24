# Step 1: Install transformers (if not installed)
# pip install transformers

from transformers import pipeline

# Step 2: Load a pre-trained model (text generation)
generator = pipeline("text-generation", model="gpt2")

# Step 3: Give it a prompt
result = generator("Once upon a time", max_length=50, num_return_sequences=1)

# Step 4: Print the output
print(result[0]['generated_text'])
