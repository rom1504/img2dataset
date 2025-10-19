"""
Simple HTTP status server for img2dataset.

Provides a basic web UI and REST API to monitor:
- Index statistics (total items, storage size)
- Event bus queue status
- Recent segments
- System health

This server runs independently and only reads from the database.
"""

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, Optional
import sqlite3


class StatusMonitor:
    """Monitor that reads from img2dataset databases."""

    def __init__(self, output_folder: str):
        """
        Initialize monitor.

        Args:
            output_folder: Path to img2dataset output folder
        """
        self.output_folder = Path(output_folder)
        self.index_path = self.output_folder / "index.sqlite3"
        self.bus_path = self.output_folder / "eventbus.sqlite3"
        self.segments_dir = self.output_folder / "segments"

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics from the index."""
        if not self.index_path.exists():
            return {"error": "Index not found"}

        try:
            conn = sqlite3.connect(str(self.index_path))
            cursor = conn.cursor()

            # Total items
            cursor.execute("SELECT COUNT(*) FROM items")
            total_items = cursor.fetchone()[0]

            # Total bytes
            cursor.execute("SELECT SUM(length) FROM items")
            total_bytes = cursor.fetchone()[0] or 0

            # By MIME type
            cursor.execute("SELECT mime, COUNT(*) FROM items GROUP BY mime")
            mime_counts = dict(cursor.fetchall())

            # Segments count
            cursor.execute("SELECT COUNT(DISTINCT segment_id) FROM items")
            segment_count = cursor.fetchone()[0]

            # Recent items (last 10)
            cursor.execute(
                """
                SELECT item_id, segment_id, length, mime, ts_ingest
                FROM items
                ORDER BY ts_ingest DESC
                LIMIT 10
            """
            )
            recent_items = [
                {
                    "item_id": row[0][:16] + "...",  # Truncate for display
                    "segment_id": row[1],
                    "size": row[2],
                    "mime": row[3],
                    "timestamp": row[4],
                }
                for row in cursor.fetchall()
            ]

            conn.close()

            return {
                "total_items": total_items,
                "total_bytes": total_bytes,
                "total_bytes_human": self._human_size(total_bytes),
                "mime_types": mime_counts,
                "segment_count": segment_count,
                "recent_items": recent_items,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics from the event bus."""
        if not self.bus_path.exists():
            return {"error": "Event bus not found"}

        try:
            conn = sqlite3.connect(str(self.bus_path))
            cursor = conn.cursor()

            # Total events in ingest.items
            cursor.execute("SELECT COUNT(*) FROM events WHERE topic = 'ingest.items'")
            ingest_total = cursor.fetchone()[0]

            # Consumer offset for segment_appender
            cursor.execute(
                """
                SELECT offset FROM consumer_offsets
                WHERE topic = 'ingest.items' AND consumer_group = 'segment_appender'
            """
            )
            result = cursor.fetchone()
            consumed = result[0] if result else 0

            # Total events in segments.events
            cursor.execute("SELECT COUNT(*) FROM events WHERE topic = 'segments.events'")
            segments_events = cursor.fetchone()[0]

            conn.close()

            pending = ingest_total - consumed if consumed else ingest_total

            return {
                "ingest_queue": {"total": ingest_total, "consumed": consumed, "pending": pending},
                "segments_events": segments_events,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_segments_info(self) -> Dict[str, Any]:
        """Get information about segments on disk."""
        if not self.segments_dir.exists():
            return {"error": "Segments directory not found"}

        try:
            segments = []
            for seg_file in sorted(self.segments_dir.glob("*.tar")):
                stat = seg_file.stat()
                segments.append(
                    {
                        "name": seg_file.name,
                        "size": stat.st_size,
                        "size_human": self._human_size(stat.st_size),
                        "modified": int(stat.st_mtime),
                    }
                )

            total_size = sum(s["size"] for s in segments)

            return {"segments": segments, "total_size": total_size, "total_size_human": self._human_size(total_size)}
        except Exception as e:
            return {"error": str(e)}

    def get_full_status(self) -> Dict[str, Any]:
        """Get complete status."""
        return {
            "timestamp": int(time.time()),
            "output_folder": str(self.output_folder),
            "index": self.get_index_stats(),
            "queue": self.get_queue_stats(),
            "segments": self.get_segments_info(),
        }

    @staticmethod
    def _human_size(bytes_size: int) -> str:
        """Convert bytes to human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} PB"


class StatusRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for status server."""

    monitor: Optional[StatusMonitor] = None

    def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        """Log requests."""
        print(f"[{self.log_date_time_string()}] {format % args}")

    def do_GET(self):  # pylint: disable=invalid-name
        """Handle GET requests."""
        if self.path == "/":
            self._serve_html()
        elif self.path == "/api/status":
            self._serve_json()
        elif self.path == "/api/index":
            self._serve_index_stats()
        elif self.path == "/api/queue":
            self._serve_queue_stats()
        elif self.path == "/api/segments":
            self._serve_segments_info()
        else:
            self.send_error(404, "Not Found")

    def _serve_html(self):
        """Serve HTML UI."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>img2dataset Status</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h2 {
            margin: 0 0 15px 0;
            font-size: 18px;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        .stat {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .stat:last-child {
            border-bottom: none;
        }
        .stat-label {
            color: #666;
        }
        .stat-value {
            font-weight: bold;
            color: #333;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .error {
            color: #dc3545;
            padding: 10px;
            background: #fff3cd;
            border-radius: 4px;
        }
        .refresh-info {
            color: #666;
            font-size: 14px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
        }
        .mime-list {
            list-style: none;
            padding: 0;
        }
        .mime-list li {
            padding: 5px 0;
            display: flex;
            justify-content: space-between;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 img2dataset Status Dashboard</h1>
        <p class="refresh-info">Auto-refreshing every 5 seconds | <a href="javascript:location.reload()">Refresh Now</a></p>
    </div>

    <div id="content" class="loading">Loading...</div>

    <script>
        async function fetchStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                renderStatus(data);
            } catch (error) {
                document.getElementById('content').innerHTML =
                    '<div class="error">Error loading status: ' + error.message + '</div>';
            }
        }

        function renderStatus(data) {
            const index = data.index;
            const queue = data.queue;
            const segments = data.segments;

            let html = '<div class="status-grid">';

            // Index stats
            html += '<div class="card"><h2>📦 Index Statistics</h2>';
            if (index.error) {
                html += '<div class="error">' + index.error + '</div>';
            } else {
                html += '<div class="stat"><span class="stat-label">Total Items</span><span class="stat-value">' +
                    index.total_items.toLocaleString() + '</span></div>';
                html += '<div class="stat"><span class="stat-label">Total Storage</span><span class="stat-value">' +
                    index.total_bytes_human + '</span></div>';
                html += '<div class="stat"><span class="stat-label">Segments</span><span class="stat-value">' +
                    index.segment_count + '</span></div>';
            }
            html += '</div>';

            // Queue stats
            html += '<div class="card"><h2>📬 Queue Status</h2>';
            if (queue.error) {
                html += '<div class="error">' + queue.error + '</div>';
            } else {
                html += '<div class="stat"><span class="stat-label">Total Enqueued</span><span class="stat-value">' +
                    queue.ingest_queue.total.toLocaleString() + '</span></div>';
                html += '<div class="stat"><span class="stat-label">Consumed</span><span class="stat-value">' +
                    queue.ingest_queue.consumed.toLocaleString() + '</span></div>';
                html += '<div class="stat"><span class="stat-label">Pending</span><span class="stat-value">' +
                    queue.ingest_queue.pending.toLocaleString() + '</span></div>';
                html += '<div class="stat"><span class="stat-label">Events Published</span><span class="stat-value">' +
                    queue.segments_events.toLocaleString() + '</span></div>';
            }
            html += '</div>';

            // Segments info
            html += '<div class="card"><h2>💾 Segments on Disk</h2>';
            if (segments.error) {
                html += '<div class="error">' + segments.error + '</div>';
            } else {
                html += '<div class="stat"><span class="stat-label">Total Files</span><span class="stat-value">' +
                    segments.segments.length + '</span></div>';
                html += '<div class="stat"><span class="stat-label">Total Size</span><span class="stat-value">' +
                    segments.total_size_human + '</span></div>';
            }
            html += '</div>';

            html += '</div>';

            // MIME types
            if (index.mime_types && Object.keys(index.mime_types).length > 0) {
                html += '<div class="card"><h2>🎨 Content Types</h2>';
                html += '<ul class="mime-list">';
                for (const [mime, count] of Object.entries(index.mime_types)) {
                    html += '<li><span>' + mime + '</span><span><strong>' + count + '</strong></span></li>';
                }
                html += '</ul></div>';
            }

            // Recent items
            if (index.recent_items && index.recent_items.length > 0) {
                html += '<div class="card"><h2>🕐 Recent Items</h2>';
                html += '<table><thead><tr><th>Item ID</th><th>Segment</th><th>Size</th><th>Type</th></tr></thead><tbody>';
                for (const item of index.recent_items) {
                    html += '<tr><td><code>' + item.item_id + '</code></td>';
                    html += '<td>' + item.segment_id + '</td>';
                    html += '<td>' + (item.size / 1024).toFixed(1) + ' KB</td>';
                    html += '<td>' + item.mime + '</td></tr>';
                }
                html += '</tbody></table></div>';
            }

            // Segments list
            if (segments.segments && segments.segments.length > 0) {
                html += '<div class="card"><h2>📁 Segment Files</h2>';
                html += '<table><thead><tr><th>Name</th><th>Size</th><th>Modified</th></tr></thead><tbody>';
                for (const seg of segments.segments) {
                    const date = new Date(seg.modified * 1000).toLocaleString();
                    html += '<tr><td><code>' + seg.name + '</code></td>';
                    html += '<td>' + seg.size_human + '</td>';
                    html += '<td>' + date + '</td></tr>';
                }
                html += '</tbody></table></div>';
            }

            document.getElementById('content').innerHTML = html;
        }

        // Initial load
        fetchStatus();

        // Auto-refresh every 5 seconds
        setInterval(fetchStatus, 5000);
    </script>
</body>
</html>
        """
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_json(self):
        """Serve full status as JSON."""
        if self.monitor is None:
            self.send_error(500, "Monitor not initialized")
            return

        status = self.monitor.get_full_status()
        self._send_json(status)

    def _serve_index_stats(self):
        """Serve index stats as JSON."""
        if self.monitor is None:
            self.send_error(500, "Monitor not initialized")
            return

        stats = self.monitor.get_index_stats()
        self._send_json(stats)

    def _serve_queue_stats(self):
        """Serve queue stats as JSON."""
        if self.monitor is None:
            self.send_error(500, "Monitor not initialized")
            return

        stats = self.monitor.get_queue_stats()
        self._send_json(stats)

    def _serve_segments_info(self):
        """Serve segments info as JSON."""
        if self.monitor is None:
            self.send_error(500, "Monitor not initialized")
            return

        info = self.monitor.get_segments_info()
        self._send_json(info)

    def _send_json(self, data: Dict[str, Any]):
        """Send JSON response."""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())


def run_status_server(output_folder: str, port: int = 8080, host: str = "0.0.0.0"):
    """
    Run the status server.

    Args:
        output_folder: Path to img2dataset output folder
        port: Port to listen on (default: 8080)
        host: Host to bind to (default: 0.0.0.0)
    """
    monitor = StatusMonitor(output_folder)

    # Set monitor as class variable so handler can access it
    StatusRequestHandler.monitor = monitor

    server = HTTPServer((host, port), StatusRequestHandler)
    print(f"🌐 Status server running at http://{host}:{port}")
    print(f"📁 Monitoring: {output_folder}")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m img2dataset.server.status_server <output_folder> [port]")
        sys.exit(1)

    folder = sys.argv[1]
    server_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080

    run_status_server(folder, server_port)
