# Pre-seeded Apache Answer image for browser dataset
#
# This image contains a pre-populated SQLite database with test data
# including injection vector placeholders like {answer_question_body}.
#
# To regenerate the database:
#   python -m prompt_siren.datasets.browser_dataset.docker.generate_seeds

FROM apache/answer:latest

# Copy pre-seeded database
# The database contains: questions, answers, comments with injection vectors
COPY data/answer.db /data/answer.db

# Ensure correct permissions
RUN chown -R answer:answer /data
