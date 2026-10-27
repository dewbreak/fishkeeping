"""Local OCR by default; --live explicitly opts in to one DeepSeek request."""

import argparse
from pathlib import Path

from fishkeeping.client import MemoryClient
from fishkeeping.config import Settings
from fishkeeping.image_chat import DeepSeekImageChat
from fishkeeping.models import Scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("question")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--user-id", default="image-demo")
    parser.add_argument("--event-id", default="image-demo-1")
    parser.add_argument("--mode", choices=["ocr", "low", "original"], default="ocr")
    parser.add_argument("--layout", choices=["text", "table"], default="text")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    image = args.image.read_bytes()
    with MemoryClient(args.url, Scope(user_id=args.user_id)) as memory:
        result = memory.enqueue_image(
            image,
            args.question,
            args.event_id,
            "image-demo",
            task="document" if args.mode == "ocr" else "scene",
            layout=args.layout,
        )
        print(result.evidence.model_dump_json(indent=2))
        print("Source episode:", result.event.episode.id)
    if not args.live:
        return
    chat = DeepSeekImageChat(Settings())
    try:
        answer = (
            chat.answer_evidence(result.evidence)
            if args.mode == "ocr"
            else chat.answer_image(image, args.question, detail=args.mode)
        )
        print(answer.model_dump_json(indent=2))
    finally:
        chat.close()


if __name__ == "__main__":
    main()
