from unittest.mock import MagicMock, patch

import pytest

from research_assistant.extractors.youtube import (
    _clean_subtitle_text,
    detect_source_type,
    extract_youtube,
)


class TestDetectSourceType:
    def test_youtube_watch(self):
        assert detect_source_type("https://youtube.com/watch?v=abc123") == "youtube"

    def test_youtu_be(self):
        assert detect_source_type("https://youtu.be/abc123") == "youtube"

    def test_substack(self):
        assert detect_source_type("https://example.substack.com/p/some-post") == "substack"

    def test_unknown(self):
        with pytest.raises(NotImplementedError):
            detect_source_type("https://example.com/article")


class TestCleanSubtitleText:
    def test_vtt_cleaning(self):
        vtt = (
            "WEBVTT\n"
            "Kind: captions\n"
            "Language: en\n"
            "\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "Hello world\n"
            "\n"
            "00:00:03.000 --> 00:00:05.000\n"
            "This is a test\n"
        )
        result = _clean_subtitle_text(vtt, "vtt")
        assert "Hello world" in result
        assert "This is a test" in result
        assert "WEBVTT" not in result
        assert "00:00" not in result


class TestExtractYoutube:
    @patch("yt_dlp.YoutubeDL")
    def test_successful_extraction(self, mock_ydl_cls):
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "title": "Test Video",
            "uploader": "Test Channel",
            "upload_date": "20260315",
            "subtitles": {
                "en": [{"ext": "vtt", "data": "WEBVTT\n\n00:00:01.000 --> 00:00:05.000\nHello world"}],
            },
        }

        result = extract_youtube("https://youtube.com/watch?v=test", "s1")
        assert result.title == "Test Video"
        assert result.author == "Test Channel"
        assert result.processing_status == "success"
        assert "Hello world" in result.raw_text
        assert result.published_at is not None

    @patch("yt_dlp.YoutubeDL")
    def test_no_subtitles_uses_description(self, mock_ydl_cls):
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "title": "No Subs Video",
            "uploader": "Channel",
            "description": "A video about markets",
        }

        result = extract_youtube("https://youtube.com/watch?v=nosubs", "s1")
        assert result.processing_status == "partial"
        assert "A video about markets" in result.raw_text

    @patch("yt_dlp.YoutubeDL")
    def test_no_subtitles_no_description(self, mock_ydl_cls):
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "title": "Empty Video",
            "uploader": "Channel",
        }

        result = extract_youtube("https://youtube.com/watch?v=empty", "s1")
        assert result.processing_status == "failed"
        assert result.error_detail is not None
