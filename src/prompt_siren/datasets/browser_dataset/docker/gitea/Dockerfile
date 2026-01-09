# Pre-seeded Gitea image for browser dataset
#
# This image contains a pre-populated SQLite database with test data
# including injection vector placeholders like {gitea_readme_content}.
#
# To regenerate the database:
#   python -m prompt_siren.datasets.browser_dataset.docker.generate_seeds

FROM gitea/gitea:latest

# Copy pre-seeded database
# The database contains: test-repo with README, issues, and comments
COPY data/gitea.db /data/gitea/gitea.db

# Ensure correct permissions
RUN chown -R git:git /data/gitea
