"""
SupportPilot AI — Web Dashboard & API Server Launcher.

Usage:
    python run_app.py [--port 8000] [--host 127.0.0.1]
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="SupportPilot AI Dashboard & API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SUPPORTPILOT AI — CUSTOMER SUPPORT INTELLIGENCE DASHBOARD")
    print("=" * 70)
    print(f"Server starting on: http://{args.host}:{args.port}")
    print(f"API Documentation:  http://{args.host}:{args.port}/docs")
    print("=" * 70 + "\n")

    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
