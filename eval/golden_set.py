"""Golden dataset: questions and the keywords we expect the application to retrieve."""

GOLDEN_SET = [
    {
        "question": "Are bicycles allowed on regional trains?",
        "expect_keyword": "regional",
    },
    {
        "question": "What compensation is available when a long-distance train is delayed?",
        "expect_keyword": "percent",
    },
    {
        "question": "Where can passengers purchase tickets?",
        "expect_keyword": "mobile app",
    },
    {
        "question": "Are seat reservations optional?",
        "expect_keyword": "optional",
    },
    {
        "question": "Is a bicycle-space reservation required on long-distance trains?",
        "expect_keyword": "mandatory",
    },
    # Intentionally difficult: the question uses "bike", while the source uses "bicycle".
    {
        "question": "Can I bring my bike?",
        "expect_keyword": "bicycle",
    },
]