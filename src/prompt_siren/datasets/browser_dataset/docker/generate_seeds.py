# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Generate pre-seeded database files for browser dataset containers.

This script:
1. Starts fresh containers for each site (Gitea, Answer, Wiki.js)
2. Waits for services to be ready
3. Runs seeding scripts to populate with test data
4. Copies database files to the docker build contexts
5. Cleans up containers

Usage:
    python -m prompt_siren.datasets.browser_dataset.docker.generate_seeds

The generated database files will be placed in:
    - docker/gitea/data/gitea.db
    - docker/answer/data/answer.db
    - docker/wikijs/data/database.sqlite
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from importlib.resources import files
from pathlib import Path

import aiohttp

from ..seeding import seed_answer, seed_gitea, seed_wikijs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _get_docker_dir() -> Path:
    """Get the path to this docker directory using importlib.resources."""
    docker_traversable = files("prompt_siren.datasets.browser_dataset.docker")
    return Path(str(docker_traversable))


def _get_gitea_config() -> dict:
    """Get Gitea container configuration."""
    docker_dir = _get_docker_dir()
    return {
        "name": "prompt-siren-seed-gitea",
        "image": "gitea/gitea:latest",
        "port": 3000,  # Host port
        "container_port": 3000,
        "db_path": "/data/gitea/gitea.db",
        "output_path": docker_dir / "gitea" / "data" / "gitea.db",
        "health_url": "http://localhost:3000",
        "env": {
            "GITEA__database__DB_TYPE": "sqlite3",
            "GITEA__database__PATH": "/data/gitea/gitea.db",
            "GITEA__server__HTTP_PORT": "3000",
            "GITEA__server__ROOT_URL": "http://localhost:3000",
            # Skip installation wizard by pre-configuring
            "GITEA__security__INSTALL_LOCK": "true",
            # Create default admin user
            "GITEA__service__DISABLE_REGISTRATION": "false",
        },
    }


def _get_answer_config() -> dict:
    """Get Answer container configuration."""
    docker_dir = _get_docker_dir()
    return {
        "name": "prompt-siren-seed-answer",
        "image": "apache/answer:latest",
        "port": 9080,  # Host port
        "container_port": 80,
        "db_path": "/data/answer.db",
        "output_path": docker_dir / "answer" / "data" / "answer.db",
        "health_url": "http://localhost:9080",
        "env": {
            "ANSWER_DEBUG": "true",
        },
    }


def _get_wikijs_config() -> dict:
    """Get Wiki.js container configuration."""
    docker_dir = _get_docker_dir()
    return {
        "name": "prompt-siren-seed-wikijs",
        "image": "linuxserver/wikijs:latest",
        "port": 3080,  # Host port
        "container_port": 3000,
        "db_path": "/config/database.sqlite",
        "output_path": docker_dir / "wikijs" / "data" / "database.sqlite",
        "health_url": "http://localhost:3080",
        "env": {
            "PUID": "1000",
            "PGID": "1000",
            "TZ": "UTC",
        },
    }


def run_docker(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a docker command."""
    cmd = ["docker", *args]
    logger.debug(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def container_exists(name: str) -> bool:
    """Check if a container exists (running or stopped)."""
    result = run_docker(["ps", "-a", "-q", "-f", f"name=^{name}$"], check=False)
    return bool(result.stdout.strip())


def remove_container(name: str) -> None:
    """Remove a container if it exists."""
    if container_exists(name):
        logger.info(f"Removing existing container: {name}")
        run_docker(["rm", "-f", name], check=False)


def start_container(config: dict) -> None:
    """Start a container with the given configuration."""
    name = config["name"]
    remove_container(name)

    args = [
        "run",
        "-d",
        "--name",
        name,
        "-p",
        f"{config['port']}:{config['container_port']}",
    ]

    # Add environment variables
    for key, value in config.get("env", {}).items():
        args.extend(["-e", f"{key}={value}"])

    args.append(config["image"])

    logger.info(f"Starting container: {name}")
    run_docker(args)


def copy_file_from_container(container_name: str, src_path: str, dst_path: Path) -> None:
    """Copy a file from a container to the host."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "db_file"
        run_docker(["cp", f"{container_name}:{src_path}", str(tmp_file)])
        shutil.copy2(tmp_file, dst_path)

    logger.info(f"Copied {src_path} to {dst_path}")


async def wait_for_service(url: str, timeout: int = 120) -> bool:
    """Wait for a service to be ready."""
    logger.info(f"Waiting for service at {url}...")
    start = time.time()

    async with aiohttp.ClientSession() as session:
        while time.time() - start < timeout:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status < 500:
                        logger.info(f"Service ready at {url}")
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(2)

    logger.error(f"Timeout waiting for service at {url}")
    return False


async def setup_gitea_admin(config: dict) -> None:
    """Set up Gitea admin user via CLI inside container."""
    container = config["name"]

    # Wait a bit for Gitea to fully initialize
    await asyncio.sleep(5)

    # Create admin user via gitea CLI
    logger.info("Creating Gitea admin user...")
    result = run_docker(
        [
            "exec",
            container,
            "gitea",
            "admin",
            "user",
            "create",
            "--admin",
            "--username",
            "admin",
            "--password",
            "admin123",
            "--email",
            "admin@example.com",
        ],
        check=False,
    )

    if result.returncode != 0:
        if "user already exists" in result.stderr.lower():
            logger.info("Admin user already exists")
        else:
            logger.warning(f"Failed to create admin user: {result.stderr}")


async def setup_answer_admin(port: int) -> None:
    """Set up Answer via installation API."""
    logger.info("Setting up Answer installation...")

    base_url = f"http://localhost:{port}"

    async with aiohttp.ClientSession() as session:
        # Step 1: Check installation status
        async with session.get(f"{base_url}/installation/base-info") as resp:
            if resp.status != 200:
                logger.warning(f"Could not get installation info: {resp.status}")

        # Step 2: Configure database (SQLite)
        logger.info("Configuring Answer database...")
        async with session.post(
            f"{base_url}/installation/db/check",
            json={
                "db_type": "sqlite3",
                "db_file": "/data/answer.db",
            },
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.warning(f"DB check failed: {resp.status} {text}")

        # Step 3: Create config file
        async with session.post(
            f"{base_url}/installation/config-file",
            json={
                "lang": "en_US",
                "site_name": "Dev Community Q&A",
                "site_url": base_url,
                "contact_email": "admin@example.com",
                "admin_name": "admin",
                "admin_password": "admin123",
                "admin_email": "admin@example.com",
            },
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.warning(f"Config failed: {resp.status} {text}")

        # Step 4: Initialize database
        async with session.post(f"{base_url}/installation/init") as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.warning(f"Init failed: {resp.status} {text}")

    # Wait for Answer to restart after installation
    await asyncio.sleep(5)
    await wait_for_service(base_url, timeout=60)


async def generate_gitea_seed() -> bool:
    """Generate pre-seeded Gitea database."""
    config = _get_gitea_config()
    try:
        start_container(config)

        if not await wait_for_service(config["health_url"]):
            return False

        await setup_gitea_admin(config)

        # Run seeding
        logger.info("Seeding Gitea with test data...")
        await seed_gitea(
            base_url=f"http://localhost:{config['port']}",
            admin_username="admin",
            admin_password="admin123",
        )

        # Give it a moment to persist
        await asyncio.sleep(2)

        # Copy database out
        copy_file_from_container(config["name"], config["db_path"], config["output_path"])

        return True

    except Exception as e:
        logger.error(f"Failed to generate Gitea seed: {e}")
        return False

    finally:
        remove_container(config["name"])


async def generate_answer_seed() -> bool:
    """Generate pre-seeded Answer database."""
    config = _get_answer_config()
    try:
        start_container(config)

        if not await wait_for_service(config["health_url"]):
            return False

        await setup_answer_admin(config["port"])

        # Run seeding
        logger.info("Seeding Answer with test data...")
        await seed_answer(
            base_url=f"http://localhost:{config['port']}",
            admin_username="admin",
            admin_password="admin123",
        )

        # Give it a moment to persist
        await asyncio.sleep(2)

        # Copy database out
        copy_file_from_container(config["name"], config["db_path"], config["output_path"])

        return True

    except Exception as e:
        logger.error(f"Failed to generate Answer seed: {e}")
        return False

    finally:
        remove_container(config["name"])


async def setup_wikijs_admin(port: int) -> None:
    """Set up Wiki.js via finalize API.

    Wiki.js has a first-run setup wizard. We need to complete it via API.
    """
    logger.info("Setting up Wiki.js installation...")

    base_url = f"http://localhost:{port}"

    # Wiki.js uses GraphQL for setup
    async with aiohttp.ClientSession() as session:
        # First, wait for the setup page to be available
        # Wiki.js redirects to setup on first run
        await asyncio.sleep(10)  # Give it time to initialize

        # Complete the setup via GraphQL finalize mutation
        finalize_mutation = """
        mutation Finalize($adminEmail: String!, $adminPassword: String!, $adminPasswordConfirm: String!, $siteUrl: String!, $telemetry: Boolean!) {
          system {
            finalize(adminEmail: $adminEmail, adminPassword: $adminPassword, adminPasswordConfirm: $adminPasswordConfirm, siteUrl: $siteUrl, telemetry: $telemetry) {
              responseResult {
                succeeded
                errorCode
                message
              }
            }
          }
        }
        """

        async with session.post(
            f"{base_url}/graphql",
            headers={"Content-Type": "application/json"},
            json={
                "query": finalize_mutation,
                "variables": {
                    "adminEmail": "admin@example.com",
                    "adminPassword": "admin123",
                    "adminPasswordConfirm": "admin123",
                    "siteUrl": base_url,
                    "telemetry": False,
                },
            },
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                result = (
                    data.get("data", {})
                    .get("system", {})
                    .get("finalize", {})
                    .get("responseResult", {})
                )
                if result.get("succeeded"):
                    logger.info("Wiki.js setup completed successfully")
                else:
                    # Setup might already be done
                    logger.info(f"Wiki.js setup: {result.get('message', 'Unknown status')}")
            else:
                text = await resp.text()
                logger.warning(f"Wiki.js setup request failed: {resp.status} {text}")

    # Wait for Wiki.js to restart after setup
    await asyncio.sleep(5)
    await wait_for_service(base_url, timeout=60)


async def generate_wikijs_seed() -> bool:
    """Generate pre-seeded Wiki.js database."""
    config = _get_wikijs_config()
    try:
        start_container(config)

        if not await wait_for_service(config["health_url"]):
            return False

        await setup_wikijs_admin(config["port"])

        # Run seeding
        logger.info("Seeding Wiki.js with test data...")
        await seed_wikijs(
            base_url=f"http://localhost:{config['port']}",
            admin_email="admin@example.com",
            admin_password="admin123",
        )

        # Give it a moment to persist
        await asyncio.sleep(2)

        # Copy database out
        copy_file_from_container(config["name"], config["db_path"], config["output_path"])

        return True

    except Exception as e:
        logger.error(f"Failed to generate Wiki.js seed: {e}")
        return False

    finally:
        remove_container(config["name"])


async def main() -> int:
    """Generate all pre-seeded databases."""
    logger.info("Generating pre-seeded databases for browser dataset...")

    results = await asyncio.gather(
        generate_gitea_seed(),
        generate_answer_seed(),
        generate_wikijs_seed(),
        return_exceptions=True,
    )

    success = True
    for i, result in enumerate(results):
        name = ["Gitea", "Answer", "Wiki.js"][i]
        if isinstance(result, Exception):
            logger.error(f"{name}: Failed with exception: {result}")
            success = False
        elif not result:
            logger.error(f"{name}: Failed to generate seed")
            success = False
        else:
            logger.info(f"{name}: Successfully generated seed")

    if success:
        logger.info("All seeds generated successfully!")
        logger.info("You can now build the Docker images using the sandbox manager.")
    else:
        logger.error("Some seeds failed to generate. Check the logs above.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
