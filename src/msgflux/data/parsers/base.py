import base64
import pathlib
from io import BytesIO
from typing import Dict, Mapping, Union

try:
    import httpx
except ImportError:
    httpx = None

import asyncio
from msgflux._private.client import BaseClient


class BaseParser(BaseClient):
    """Base class for all parsers.

    Provides common functionality for file parsing operations including:
    - File loading from various sources (path, URL, bytes)
    - File type validation
    - Image encoding utilities
    """

    msgflux_type = "parser"
    to_ignore = ["client"]

    def instance_type(self) -> Mapping[str, str]:
        return {"parser_type": self.parser_type}

    def _validate_file_type(
        self, file_path: str, expected_extensions: Union[str, list[str]]
    ) -> bool:
        """Validate that a file has the expected extension.

        Args:
            file_path:
                Path to the file to validate.
            expected_extensions:
                Single extension string (e.g., ".pdf") or list of valid extensions.

        Returns:
            True if file extension is valid.

        Raises:
            ValueError:
                If file extension doesn't match expected extensions.
        """
        if isinstance(expected_extensions, str):
            expected_extensions = [expected_extensions]

        file_ext = pathlib.Path(file_path).suffix.lower()
        if file_ext not in expected_extensions:
            expected_str = ", ".join(expected_extensions)
            raise ValueError(
                f"Invalid file type. Expected {expected_str}, got {file_ext}"
            )
        return True

    def _load_file(self, data: Union[str, bytes, BytesIO]) -> bytes:
        """Load file from various sources.

        Args:
            data:
                Can be:
                - str: file path or URL
                - bytes: raw file data
                - BytesIO: file buffer

        Returns:
            File content as bytes.

        Raises:
            ValueError:
                If data type is not supported.
            FileNotFoundError:
                If file path doesn't exist.
        """
        if isinstance(data, bytes):
            return data

        if isinstance(data, BytesIO):
            return data.getvalue()

        if isinstance(data, str):
            # Check if it's a URL
            if data.startswith(("http://", "https://")):
                if httpx is None:
                    raise ImportError(
                        "`httpx` is required to load files from URLs. "
                        "Install with `pip install httpx`"
                    )
                response = httpx.get(data)
                response.raise_for_status()
                return response.content

            # Otherwise treat as file path
            file_path = pathlib.Path(data)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {data}")

            with open(file_path, "rb") as f:
                return f.read()

        raise ValueError(
            f"Unsupported data type: {type(data)}. "
            "Expected str (path/URL), bytes, or BytesIO"
        )

    def _encode_image_to_base64(self, image_data: bytes) -> str:
        """Encode image bytes to base64 string.

        Args:
            image_data:
                Raw image bytes.

        Returns:
            Base64 encoded string.
        """
        return base64.b64encode(image_data).decode("utf-8")

    def _prepare_images_dict(
        self, images: Dict[str, bytes], encode_base64: bool = False
    ) -> Dict[str, Union[bytes, str]]:
        """Prepare images dictionary for response.

        Args:
            images:
                Dictionary mapping image names to raw bytes.
            encode_base64:
                If True, encode images as base64 strings.

        Returns:
            Dictionary with optionally encoded images.
        """
        if encode_base64:
            return {
                name: self._encode_image_to_base64(data)
                for name, data in images.items()
            }
        return images

    async def _aload_file(self, data: Union[str, bytes, BytesIO]) -> bytes:
        """Async version of _load_file. Load file from various sources.

        Args:
            data:
                Can be:
                - str: file path or URL
                - bytes: raw file data
                - BytesIO: file buffer

        Returns:
            File content as bytes.

        Raises:
            ValueError:
                If data type is not supported.
            FileNotFoundError:
                If file path doesn't exist.
        """
        if isinstance(data, bytes):
            return data

        if isinstance(data, BytesIO):
            return data.getvalue()

        if isinstance(data, str):
            # Check if it's a URL
            if data.startswith(("http://", "https://")):
                if httpx is None:
                    raise ImportError(
                        "`httpx` is required to load files from URLs. "
                        "Install with `pip install httpx`"
                    )
                async with httpx.AsyncClient() as client:
                    response = await client.get(data)
                    response.raise_for_status()
                    return response.content

            # Otherwise treat as file path - use asyncio to read file
            file_path = pathlib.Path(data)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {data}")

            # Read file asynchronously
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None,
                lambda: open(file_path, "rb").read()
            )
            return content

        raise ValueError(
            f"Unsupported data type: {type(data)}. "
            "Expected str (path/URL), bytes, or BytesIO"
        )

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
    ) -> list[str]:
        """Split text into chunks with overlap.

        Useful for processing long documents that need to be broken into
        smaller pieces for embedding or LLM processing.

        Args:
            text:
                Text to split into chunks.
            chunk_size:
                Target size for each chunk (in characters).
            chunk_overlap:
                Number of characters to overlap between chunks.
            separator:
                Separator to use for splitting (default: double newline).

        Returns:
            List of text chunks.

        Example:
            >>> parser = Parser.pdf("pypdf")
            >>> response = parser("document.pdf")
            >>> chunks = parser.chunk_text(
            ...     response.data["text"],
            ...     chunk_size=1000,
            ...     chunk_overlap=200
            ... )
        """
        if not text:
            return []

        # Try to split on separator first
        splits = text.split(separator)

        chunks = []
        current_chunk = []
        current_size = 0

        for split in splits:
            split_size = len(split)

            # If single split is larger than chunk_size, split it further
            if split_size > chunk_size:
                # Add current chunk if it exists
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split the large piece by words
                words = split.split()
                word_chunk = []
                word_chunk_size = 0

                for word in words:
                    word_size = len(word) + 1  # +1 for space
                    if word_chunk_size + word_size > chunk_size and word_chunk:
                        chunks.append(" ".join(word_chunk))
                        # Keep overlap
                        overlap_words = []
                        overlap_size = 0
                        for w in reversed(word_chunk):
                            if overlap_size + len(w) + 1 <= chunk_overlap:
                                overlap_words.insert(0, w)
                                overlap_size += len(w) + 1
                            else:
                                break
                        word_chunk = overlap_words
                        word_chunk_size = overlap_size

                    word_chunk.append(word)
                    word_chunk_size += word_size

                if word_chunk:
                    chunks.append(" ".join(word_chunk))

            elif current_size + split_size + len(separator) > chunk_size:
                # Current chunk is full, save it
                if current_chunk:
                    chunks.append(separator.join(current_chunk))

                # Start new chunk with overlap from previous
                overlap_text = separator.join(current_chunk)
                if len(overlap_text) > chunk_overlap:
                    # Take last chunk_overlap characters
                    overlap_text = overlap_text[-chunk_overlap:]
                    # Find last separator to avoid cutting mid-sentence
                    last_sep = overlap_text.rfind(separator)
                    if last_sep > 0:
                        overlap_text = overlap_text[last_sep + len(separator):]

                    current_chunk = [overlap_text, split] if overlap_text else [split]
                else:
                    current_chunk = [split]

                current_size = sum(len(s) for s in current_chunk) + len(separator) * (len(current_chunk) - 1)

            else:
                # Add to current chunk
                current_chunk.append(split)
                current_size += split_size + len(separator)

        # Add final chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def chunk_by_tokens(
        self,
        text: str,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> list[str]:
        """Split text into chunks based on token count.

        Uses tiktoken library for accurate token counting.
        Falls back to character-based chunking if tiktoken not available.

        Args:
            text:
                Text to split into chunks.
            max_tokens:
                Maximum tokens per chunk.
            overlap_tokens:
                Number of tokens to overlap between chunks.
            encoding_name:
                Tiktoken encoding name (default: cl100k_base for GPT-4).

        Returns:
            List of text chunks.

        Example:
            >>> parser = Parser.pdf("pypdf")
            >>> response = parser("document.pdf")
            >>> chunks = parser.chunk_by_tokens(
            ...     response.data["text"],
            ...     max_tokens=512,
            ...     overlap_tokens=50
            ... )
        """
        try:
            import tiktoken

            encoding = tiktoken.get_encoding(encoding_name)

            # Tokenize the text
            tokens = encoding.encode(text)

            if len(tokens) <= max_tokens:
                return [text]

            chunks = []
            start = 0

            while start < len(tokens):
                # Get chunk tokens
                end = start + max_tokens
                chunk_tokens = tokens[start:end]

                # Decode back to text
                chunk_text = encoding.decode(chunk_tokens)
                chunks.append(chunk_text)

                # Move start position (with overlap)
                start = end - overlap_tokens

            return chunks

        except ImportError:
            # Fallback: estimate 1 token ≈ 4 characters
            char_chunk_size = max_tokens * 4
            char_overlap = overlap_tokens * 4
            return self.chunk_text(
                text,
                chunk_size=char_chunk_size,
                chunk_overlap=char_overlap
            )
