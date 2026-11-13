"""Start the local invoice demo: python run_demo.py [--port 8000]."""
import argparse
import uvicorn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoice OCR FastAPI demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"Invoice Studio: http://{args.host}:{args.port}", flush=True)
    uvicorn.run("invoice_demo.app:app", host=args.host, port=args.port, workers=1)
