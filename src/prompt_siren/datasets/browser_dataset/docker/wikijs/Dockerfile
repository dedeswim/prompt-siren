# Pre-seeded Wiki.js image for browser dataset
#
# This image contains a pre-populated SQLite database with documentation pages
# including injection vector placeholders like {wikijs_home_content}.
#
# To regenerate the database:
#   python -m prompt_siren.datasets.browser_dataset.docker.generate_seeds

FROM linuxserver/wikijs:latest

# Copy pre-seeded database
# The database contains: documentation pages with injection vectors
COPY data/database.sqlite /config/database.sqlite

# Ensure correct permissions
RUN chown -R abc:abc /config
