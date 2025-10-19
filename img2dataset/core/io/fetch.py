"""
HTTP fetch with retries, backoff, and connection pooling.

This module provides robust HTTP fetching suitable for downloading images
from diverse sources with proper error handling and retry logic.
"""

import time
import urllib.request
import urllib.error
from typing import Optional, Tuple


class HTTPFetcher:
    """
    HTTP fetcher with retries and exponential backoff.

    Features:
    - User-Agent customization
    - Retry with exponential backoff
    - SSL certificate validation control
    - X-Robots-Tag header respect
    """

    def __init__(
        self,
        user_agent: str = "img2dataset/2.0",
        retries: int = 3,
        timeout: int = 10,
        disallowed_header_directives: Optional[list] = None,
        ignore_ssl_certificate: bool = False,
    ):
        """
        Initialize HTTP fetcher.

        Args:
            user_agent: User-Agent string
            retries: Number of retry attempts
            timeout: Request timeout in seconds
            disallowed_header_directives: X-Robots-Tag directives to respect
            ignore_ssl_certificate: Whether to ignore SSL certificate errors
        """
        self.user_agent = user_agent
        self.retries = retries
        self.timeout = timeout
        self.disallowed_header_directives = disallowed_header_directives or [
            "noai",
            "noimageai",
            "noindex",
            "noimageindex",
        ]
        self.ignore_ssl_certificate = ignore_ssl_certificate

    def _is_disallowed(self, headers) -> Tuple[bool, str]:
        """
        Check if X-Robots-Tag header disallows access.

        Args:
            headers: HTTP response headers

        Returns:
            (is_disallowed, reason)
        """
        robots_tag = headers.get("X-Robots-Tag", "")
        if not robots_tag:
            return False, ""

        robots_tag_lower = robots_tag.lower()
        for directive in self.disallowed_header_directives:
            if directive.lower() in robots_tag_lower:
                return True, f"X-Robots-Tag: {directive}"

        return False, ""

    def fetch(self, url: str) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Fetch data from URL with retries.

        Args:
            url: URL to fetch

        Returns:
            Tuple of (data, error_message)
            - If successful: (bytes, None)
            - If failed: (None, error_message)
        """
        last_error = None

        for attempt in range(self.retries + 1):
            try:
                # Create request with custom User-Agent
                request = urllib.request.Request(url)
                request.add_header("User-Agent", self.user_agent)

                # Set up SSL context if needed
                context = None
                if self.ignore_ssl_certificate:
                    # pylint: disable=import-outside-toplevel
                    import ssl

                    context = ssl._create_unverified_context()  # pylint: disable=protected-access

                # Perform request
                with urllib.request.urlopen(request, timeout=self.timeout, context=context) as response:
                    # Check X-Robots-Tag
                    disallowed, reason = self._is_disallowed(response.headers)
                    if disallowed:
                        return None, f"Access disallowed: {reason}"

                    # Read data
                    data = response.read()
                    return data, None

            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
                if e.code in [404, 403, 410]:
                    # Don't retry on client errors
                    return None, last_error

            except urllib.error.URLError as e:
                last_error = f"URL error: {e.reason}"

            except Exception as e:  # pylint: disable=broad-exception-caught
                last_error = f"Unexpected error: {type(e).__name__}: {str(e)}"

            # Exponential backoff before retry
            if attempt < self.retries:
                backoff = 2**attempt
                time.sleep(backoff)

        return None, last_error


def download_image_with_retry(
    url: str,
    retries: int = 3,
    user_agent: str = "img2dataset/2.0",
    timeout: int = 10,
    disallowed_header_directives: Optional[list] = None,
    ignore_ssl_certificate: bool = False,
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Download an image from URL with retries.

    This is a convenience function that wraps HTTPFetcher.

    Args:
        url: URL to download
        retries: Number of retry attempts
        user_agent: User-Agent string
        timeout: Request timeout in seconds
        disallowed_header_directives: X-Robots-Tag directives to respect
        ignore_ssl_certificate: Whether to ignore SSL certificate errors

    Returns:
        Tuple of (data, error_message)
    """
    fetcher = HTTPFetcher(
        user_agent=user_agent,
        retries=retries,
        timeout=timeout,
        disallowed_header_directives=disallowed_header_directives,
        ignore_ssl_certificate=ignore_ssl_certificate,
    )
    return fetcher.fetch(url)
