import logging
import re
from datetime import datetime, timezone

from research_assistant.schemas import ContentItem, FormatMetadata

logger = logging.getLogger(__name__)


def detect_source_type(url: str) -> str:
    if re.search(r"(youtube\.com/watch|youtu\.be/)", url):
        return "youtube"
    if "substack.com" in url:
        return "substack"
    raise NotImplementedError(f"Source type detection not implemented for URL: {url}")


def _extract_subtitle_text(info: dict) -> str | None:
    subtitles = info.get("subtitles") or {}
    auto_subs = info.get("automatic_captions") or {}

    # Prefer manual English subtitles
    for lang_key in ("en", "en-US", "en-GB"):
        if lang_key in subtitles:
            return _download_sub_text(subtitles[lang_key])

    # Fall back to auto-generated
    for lang_key in ("en", "en-US", "en-GB", "en-orig"):
        if lang_key in auto_subs:
            return _download_sub_text(auto_subs[lang_key])

    return None


def _download_sub_text(sub_formats: list[dict]) -> str | None:
    # Prefer vtt or json3 formats
    for fmt in sub_formats:
        ext = fmt.get("ext", "")
        if ext in ("vtt", "json3", "srv1", "srv2", "srv3"):
            # yt-dlp provides the text in 'data' key when using extract_info
            # In practice we use the write_subtitles option and read the file
            if "data" in fmt:
                return _clean_subtitle_text(fmt["data"], ext)
    return None


def _clean_subtitle_text(raw: str, ext: str) -> str:
    if ext == "vtt":
        # Remove VTT header and timestamps
        lines = raw.split("\n")
        text_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
                continue
            if re.match(r"^\d{2}:\d{2}", line):
                continue
            if re.match(r"^[\d\s\-\>:\.]+$", line):
                continue
            # Remove inline tags
            line = re.sub(r"<[^>]+>", "", line)
            if line:
                text_lines.append(line)
        return " ".join(text_lines)
    # For other formats, return as-is (best effort)
    return raw


def extract_youtube(url: str, source_id: str) -> ContentItem:
    import yt_dlp

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title", "Unknown")
    author = info.get("uploader", info.get("channel", "Unknown"))
    upload_date = info.get("upload_date")
    published_at = None
    if upload_date:
        try:
            published_at = datetime.strptime(upload_date, "%Y%m%d").replace(
                tzinfo=timezone.utc
            ).isoformat()
        except ValueError:
            pass

    transcript = _extract_subtitle_text(info)
    if transcript:
        status = "success"
        error = None
    else:
        # Fall back to description as last resort
        description = info.get("description", "")
        if description:
            transcript = f"[No transcript available. Video description:]\n{description}"
            status = "partial"
            error = "No subtitles found; used video description as fallback"
        else:
            transcript = ""
            status = "failed"
            error = "No subtitles or description available"

    return ContentItem(
        source_id=source_id,
        content_type="transcript",
        title=title,
        author=author,
        published_at=published_at,
        raw_text=transcript,
        word_count=len(transcript.split()) if transcript else 0,
        format_metadata=FormatMetadata(
            has_sections=False,
            has_citations=False,
            has_data_tables=False,
            is_paywalled=False,
        ),
        processing_status=status,
        error_detail=error,
    )
