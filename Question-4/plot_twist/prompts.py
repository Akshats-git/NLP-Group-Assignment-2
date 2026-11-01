"""The story openings used in Part 3 (decoding comparison) and Part 4 (Plot-Twist challenge).

Written before any model was trained or sampled, and not changed after seeing outputs.
They are in TinyStories register (short, concrete, child-level vocabulary) because the models
only know that world; the plot-twist openings keep the brief's example ideas.
"""

# Part 3: one per category named in the brief.
DECODING_PROMPTS = {
    "mystery": "One night, Tom heard a strange noise in the garden. He took a flashlight and went outside to see what it was.",
    "discovery": "Lily was digging in the sand at the beach when she found a shiny box with a tiny key on top.",
    "unusual_character": "There was once a little dragon who was afraid of fire. Every day he hid behind a big rock.",
    "problem": "Ben's kite was stuck high up in a tall tree and he could not reach it. He looked around for help.",
    "unexpected_ending": "Mia and her dog went for a walk in the park. Everything was quiet and sunny until the dog stopped and started to bark.",
}

# Part 4: openings designed to lead to a surprise (brief's examples, rewritten in the corpus register).
PLOT_TWIST_PROMPTS = {
    "exam_paper": "A girl named Sara found her test paper on the desk. When she opened it, she saw that it already had the answers to the questions for tomorrow.",
    "future_robot": "An old robot sat alone in the garage. One day, a little light on his chest began to blink and he got a message from himself in the future.",
    "detective": "The detective walked into the empty house. Every clock on the wall had stopped at midnight. Suddenly, the phone began to ring.",
    "forgotten_day": "One morning, everyone in the village woke up and nobody could remember what happened the day before.",
    "future_letter": "Jack found a letter in his mailbox with his name on it. The letter was written one hundred years in the future.",
}

ALL_PROMPTS = {**{f"decode_{k}": v for k, v in DECODING_PROMPTS.items()},
               **{f"twist_{k}": v for k, v in PLOT_TWIST_PROMPTS.items()}}

# Named characters in each opening (lowercase), used to check whether a continuation keeps them consistent.
PROMPT_NAMES = {"decode_mystery": ["tom"], "decode_discovery": ["lily"], "decode_problem": ["ben"],
                "decode_unexpected_ending": ["mia"], "twist_exam_paper": ["sara"], "twist_future_letter": ["jack"]}
