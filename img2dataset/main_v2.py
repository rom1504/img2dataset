"""
New CLI entry point with producer/consumer mode support.

This extends the traditional img2dataset CLI with service-oriented commands.
"""

import fire
from .main import download  # Traditional download function
from .cli.service import start_service, enqueue, materialize
from .server.status_server import run_status_server


class Img2DatasetCLI:
    """
    img2dataset CLI with producer/consumer mode support.

    Commands:
        download: Traditional download mode (backward compatible)
        service: Start service mode (segment appender)
        enqueue: Enqueue URLs to ingest queue
        materialize: Materialize dataset from segments
        status: Run web status server
    """

    def download(self, *args, **kwargs):
        """
        Traditional download mode (backward compatible).

        Downloads images from URLs and creates dataset in specified format.
        Run `img2dataset download --help` for full options.
        """
        return download(*args, **kwargs)

    def service(
        self,
        output_folder: str = "output",
        max_segment_size: int = 4 * 1024 * 1024 * 1024,
        fetch_retries: int = 3,
        fetch_timeout: int = 10,
        user_agent_token: str = None,
        max_items: int = None
    ):
        """
        Start service mode (segment appender).

        This runs a local segment appender that consumes URLs from the
        ingest.items queue and writes to segments.

        Args:
            output_folder: Output directory for segments, index, and bus
            max_segment_size: Maximum segment size in bytes (default: 4GB)
            fetch_retries: Number of HTTP retry attempts (default: 3)
            fetch_timeout: HTTP timeout in seconds (default: 10)
            user_agent_token: Optional token for User-Agent string
            max_items: Maximum items to process (default: unlimited)
        """
        return start_service(
            output_folder=output_folder,
            max_segment_size=max_segment_size,
            fetch_retries=fetch_retries,
            fetch_timeout=fetch_timeout,
            user_agent_token=user_agent_token,
            max_items=max_items
        )

    def enqueue(
        self,
        url_list: str,
        output_folder: str = "output",
        input_format: str = "txt",
        url_col: str = "url"
    ):
        """
        Enqueue URLs to ingest queue.

        Args:
            url_list: Path to URL list file
            output_folder: Output directory (for event bus)
            input_format: Input format (txt, csv, json, parquet)
            url_col: Column name for URLs (for structured formats)
        """
        return enqueue(
            url_list=url_list,
            output_folder=output_folder,
            input_format=input_format,
            url_col=url_col
        )

    def materialize(
        self,
        output_folder: str = "output",
        manifest_path: str = "manifest.json",
        output_format: str = "manifest"
    ):
        """
        Materialize dataset from segments.

        Creates a manifest or exports the index in various formats.

        Args:
            output_folder: Output directory (for index)
            manifest_path: Path to output manifest
            output_format: Output format (manifest, parquet)
        """
        return materialize(
            output_folder=output_folder,
            manifest_path=manifest_path,
            output_format=output_format
        )

    def status(self, output_folder: str, port: int = 8080, host: str = "0.0.0.0"):
        """
        Run web status server.

        Provides a simple web UI and REST API to monitor the system status.
        Runs independently and only reads from databases.

        Args:
            output_folder: Output directory to monitor
            port: Port to listen on (default: 8080)
            host: Host to bind to (default: 0.0.0.0)
        """
        return run_status_server(output_folder=output_folder, port=port, host=host)


def main():
    """Main entry point for img2dataset CLI."""
    fire.Fire(Img2DatasetCLI)


if __name__ == "__main__":
    main()
