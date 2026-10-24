"""Question 2: 5-class Amazon review rating with a BiGRU baseline and fine-tuned RoBERTa.

    data          loads the reviews and puts each title back in front of its body
    classic       BiGRU pipeline: clean, lowercase, spaCy tokens, Vocab, GloVe matrix
    roberta_data  RoBERTa pipeline: byte-level BPE, truncation, dynamic padding
"""
