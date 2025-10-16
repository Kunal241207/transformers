from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
The Transformer architecture revolutionized natural language processing by enabling models 
to process text in parallel rather than sequentially, improving both efficiency and accuracy. 
Today, transformer-based models dominate tasks like translation, summarization, and question answering.
"""

summary = summarizer(text, max_length=50, min_length=20, do_sample=False)
print(summary[0]['summary_text'])
