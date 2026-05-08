"""Development entry point for running the MedAI FastAPI application."""

import os

import uvicorn


def main() -> None:
    """Check local artifacts and start the MedAI API server with development reload."""
    if not os.path.exists("models/best_model.pt"):
        print("Model not found. Run training first or place best_model.pt in models/")
    if not os.path.exists("data/medical_kb/medical.index"):
        print("Knowledge base index not found. Run /build-index endpoint after starting server.")

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
