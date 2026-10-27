"""Local video sampling by default; --live enables one explicit DeepSeek request."""

import argparse
import base64
import json
from pathlib import Path

from fishkeeping.client import MemoryClient
from fishkeeping.config import Settings
from fishkeeping.image_chat import DeepSeekImageChat
from fishkeeping.models import Scope
from fishkeeping.video import MAX_VIDEO_BYTES, VideoProcessor
from fishkeeping.video_models import VideoRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("question")
    parser.add_argument("--url", help="Use a running Fishkeeping API instead of the local decoder")
    parser.add_argument("--user-id", default="video-demo")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--agent-id", default="default")
    parser.add_argument(
        "--event-id", help="Also archive provenance through --url; no media is stored"
    )
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--max-frames", type=int, choices=range(2, 13), default=6)
    parser.add_argument("--strategy", choices=["adaptive", "uniform"], default="adaptive")
    parser.add_argument("--detail", choices=["low", "original"], default="low")
    parser.add_argument("--live", action="store_true", help="Make at most one paid model request")
    parser.add_argument(
        "--allow-review",
        action="store_true",
        help="Explicitly accept sampling warnings for this model request",
    )
    args = parser.parse_args()
    if args.event_id and not args.url:
        parser.error("--event-id requires --url to persist a reliable event")
    if args.video.stat().st_size > MAX_VIDEO_BYTES:
        parser.error("video exceeds the 24 MiB input limit")
    video = args.video.read_bytes()
    scope = Scope(namespace=args.namespace, user_id=args.user_id, agent_id=args.agent_id)
    options = {
        "start_seconds": args.start,
        "end_seconds": args.end,
        "max_frames": args.max_frames,
        "strategy": args.strategy,
    }
    if args.url:
        with MemoryClient(args.url, scope) as memory:
            evidence = memory.analyze_video(video, args.question, **options)
            if args.event_id:
                # First archival decodes again; receipt replays only read saved provenance.
                receipt = memory.enqueue_video(
                    video, args.question, args.event_id, "video-demo", **options
                )
                print("Source episode:", receipt.episode.id)
    else:
        evidence = VideoProcessor().analyze(
            VideoRequest(
                scope=scope,
                video_base64=base64.b64encode(video).decode("ascii"),
                question=args.question,
                **options,
            )
        )
    # Do not dump frame Base64 into the terminal or an application log.
    print(json.dumps(json.loads(evidence.packed), ensure_ascii=False, indent=2))
    if not args.live:
        return
    if evidence.requires_review and not args.allow_review:
        raise SystemExit(
            "Sampling requires review. Narrow the time window or explicitly use --allow-review."
        )
    chat = DeepSeekImageChat(Settings())
    try:
        answer = chat.answer_video(evidence, detail=args.detail, allow_review=args.allow_review)
        print(answer.model_dump_json(indent=2))
    finally:
        chat.close()


if __name__ == "__main__":
    main()
