-- Seed data for the "scientists" table: a small structured dataset of computer
-- science / AI pioneers, used to demo a Postgres-backed lookup tool alongside
-- the document retriever, Wikipedia, and GitHub tools.

DROP TABLE IF EXISTS scientists;

CREATE TABLE scientists (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    field TEXT NOT NULL,
    known_for TEXT NOT NULL,
    birth_year INTEGER
);

INSERT INTO scientists (name, field, known_for, birth_year) VALUES
    ('Alan Turing', 'Computer Science', 'Turing machine; breaking the Enigma cipher at Bletchley Park', 1912),
    ('Ada Lovelace', 'Mathematics', 'Wrote the first published algorithm intended for a computing machine', 1815),
    ('John von Neumann', 'Mathematics', 'Von Neumann computer architecture; foundational work in game theory', 1903),
    ('Claude Shannon', 'Information Theory', 'Founded information theory; "A Mathematical Theory of Communication"', 1916),
    ('John McCarthy', 'Computer Science', 'Coined the term "Artificial Intelligence"; invented the Lisp language', 1927),
    ('Marvin Minsky', 'Computer Science', 'Co-founded the MIT Artificial Intelligence Laboratory', 1927),
    ('Geoffrey Hinton', 'Machine Learning', 'Backpropagation and deep learning pioneer; 2018 Turing Award', 1947),
    ('Yann LeCun', 'Machine Learning', 'Convolutional neural networks; 2018 Turing Award', 1960),
    ('Yoshua Bengio', 'Machine Learning', 'Deep learning and neural sequence modeling; 2018 Turing Award', 1964),
    ('Andrew Ng', 'Machine Learning', 'Co-founded Google Brain and Coursera; deep learning education', 1976),
    ('Fei-Fei Li', 'Computer Vision', 'Created the ImageNet dataset that catalyzed deep learning for vision', 1976),
    ('Ian Goodfellow', 'Machine Learning', 'Invented Generative Adversarial Networks (GANs)', 1985),
    ('Demis Hassabis', 'AI Research', 'Co-founded DeepMind; led development of AlphaFold', 1976),
    ('Timnit Gebru', 'AI Ethics', 'AI ethics researcher; co-founded the DAIR research institute', 1983),
    ('Ashish Vaswani', 'Machine Learning', 'Lead author of "Attention Is All You Need", introducing the Transformer', NULL);
