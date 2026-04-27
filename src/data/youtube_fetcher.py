from youtube_comment_downloader import YoutubeCommentDownloader
from typing import List
import logging

logger = logging.getLogger(__name__)


class YouTubeFetcher:
    """
    Fetch YouTube comments without API key.
    """

    def __init__(self, max_comments: int = 200):
        self.downloader = YoutubeCommentDownloader()
        self.max_comments = max_comments

    def fetch_comments(self, video_url: str) -> List[str]:
        """
        Returns a list of comment strings.
        """
        try:
            comments = self.downloader.get_comments_from_url(video_url)
        except Exception as e:
            logger.error("Failed to fetch comments: %s", e)
            return []

        results: List[str] = []

        for i, c in enumerate(comments):
            if i >= self.max_comments:
                break

            text = c.get("text", "").strip()

            # Filter weak/noisy comments
            if text and len(text) > 15:
                results.append(text)

        logger.info("Fetched %d comments", len(results))
        return results