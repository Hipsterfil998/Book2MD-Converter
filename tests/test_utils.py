"""Tests for utils.py — pil_to_data_url and sample_indices."""
import base64
from io import BytesIO

import pytest
from PIL import Image

from utils import pil_to_data_url, sample_indices


class TestSampleIndices:
    def test_small_total_returns_all(self):
        assert sample_indices(5, 20) == [0, 1, 2, 3, 4]

    def test_equal_total_returns_all(self):
        assert sample_indices(20, 20) == list(range(20))

    def test_zero_total(self):
        assert sample_indices(0, 20) == []

    def test_large_returns_exactly_n(self):
        assert len(sample_indices(100, 20)) == 20

    def test_output_is_sorted(self):
        result = sample_indices(100, 20)
        assert result == sorted(result)

    def test_indices_in_valid_range(self):
        result = sample_indices(200, 20)
        assert all(0 <= i < 200 for i in result)

    def test_no_duplicates(self):
        result = sample_indices(100, 20)
        assert len(result) == len(set(result))

    def test_always_includes_first_10_pages(self):
        result = sample_indices(100, 20)
        assert all(i in result for i in range(10))

    def test_always_includes_first_10_pages_large(self):
        result = sample_indices(500, 20)
        assert all(i in result for i in range(10))

    def test_first_10_guaranteed_when_n_equals_10(self):
        result = sample_indices(100, 10)
        assert result == list(range(10))

    def test_first_n_guaranteed_when_n_less_than_10(self):
        # If n < 10, guaranteed pages are 0..n-1
        result = sample_indices(50, 5)
        assert result == [0, 1, 2, 3, 4]

    def test_covers_back(self):
        result = sample_indices(100, 20)
        assert any(i >= 80 for i in result)

    def test_n_equals_one(self):
        result = sample_indices(50, 1)
        assert result == [0]

    def test_n_equals_two(self):
        result = sample_indices(50, 2)
        assert result == [0, 1]

    def test_n_equals_three(self):
        result = sample_indices(50, 3)
        assert result == [0, 1, 2]


class TestPilToDataUrl:
    def test_starts_with_data_url_prefix(self):
        img = Image.new("RGB", (10, 10))
        assert pil_to_data_url(img).startswith("data:image/png;base64,")

    def test_base64_decodes_to_valid_png(self):
        img = Image.new("RGB", (10, 10))
        url = pil_to_data_url(img)
        data = base64.b64decode(url.split(",")[1])
        assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes

    def test_roundtrip_preserves_size(self):
        img = Image.new("RGB", (32, 32), color=(128, 64, 0))
        url = pil_to_data_url(img)
        data = base64.b64decode(url.split(",")[1])
        recovered = Image.open(BytesIO(data))
        assert recovered.size == (32, 32)
