"""Tests for calculator and search tools."""

from __future__ import annotations

import pytest

from tools.calculator import calculate
from tools.search import search


class TestCalculator:
    """Test the safe math calculator."""

    def test_basic_addition(self) -> None:
        """Test simple addition."""
        assert calculate("2 + 3") == "5"

    def test_multiplication(self) -> None:
        """Test multiplication."""
        assert calculate("4 * 7") == "28"

    def test_float_division(self) -> None:
        """Test division resulting in float."""
        result = calculate("10 / 3")
        assert result.startswith("3.333")

    def test_whole_number_division(self) -> None:
        """Test division resulting in whole number drops .0."""
        assert calculate("10 / 2") == "5"

    def test_power(self) -> None:
        """Test exponentiation."""
        assert calculate("2 ** 10") == "1024"

    def test_modulo(self) -> None:
        """Test modulo operator."""
        assert calculate("17 % 5") == "2"

    def test_floor_division(self) -> None:
        """Test floor division."""
        assert calculate("17 // 5") == "3"

    def test_negative_numbers(self) -> None:
        """Test unary minus."""
        assert calculate("-5 + 3") == "-2"

    def test_complex_expression(self) -> None:
        """Test combined operations with precedence."""
        assert calculate("(2 + 3) * 4 - 1") == "19"

    def test_division_by_zero(self) -> None:
        """Test division by zero returns error string."""
        result = calculate("1 / 0")
        assert "Error" in result
        assert "zero" in result.lower()

    def test_rejects_function_calls(self) -> None:
        """Test that function calls are rejected."""
        result = calculate("abs(-5)")
        assert "Error" in result

    def test_rejects_import(self) -> None:
        """Test that import statements are rejected."""
        result = calculate("__import__('os')")
        assert "Error" in result

    def test_rejects_variable_names(self) -> None:
        """Test that variable references are rejected."""
        result = calculate("x + 1")
        assert "Error" in result

    def test_rejects_string_literals(self) -> None:
        """Test that strings are rejected."""
        result = calculate("'hello'")
        assert "Error" in result

    def test_empty_expression(self) -> None:
        """Test empty input returns error."""
        result = calculate("")
        assert "Error" in result


class TestSearch:
    """Test the search tool."""

    @pytest.mark.asyncio
    async def test_demo_mode_default(self) -> None:
        """Test search returns demo results with no API key."""
        result = await search("test query")
        assert "Demo Mode" in result or "demo" in result.lower() or "Note:" in result

    @pytest.mark.asyncio
    async def test_demo_mode_gdp_keyword(self) -> None:
        """Test demo mode recognizes GDP keyword."""
        result = await search("France GDP 2024")
        assert "GDP" in result or "Demo Mode" in result

    @pytest.mark.asyncio
    async def test_demo_mode_population_keyword(self) -> None:
        """Test demo mode recognizes population keyword."""
        result = await search("world population")
        assert "population" in result.lower() or "Population" in result

    @pytest.mark.asyncio
    async def test_demo_mode_empty_key(self) -> None:
        """Test empty string API key triggers demo mode."""
        result = await search("anything", api_key="")
        assert "Demo Mode" in result or "Note:" in result

    @pytest.mark.asyncio
    async def test_demo_mode_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that demo mode emits a WARNING so operators notice the degradation."""
        import logging

        with caplog.at_level(logging.WARNING, logger="tools.search"):
            await search("some query")
        assert any("SEARCH_API_KEY" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_real_mode_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that a non-empty API key does NOT trigger the demo mode warning."""
        import logging
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_response = MagicMock()
        mock_response.json.return_value = {"AbstractText": "Result", "RelatedTopics": []}
        mock_response.raise_for_status = lambda: None

        with caplog.at_level(logging.WARNING, logger="tools.search"):
            with patch("httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    return_value=mock_response
                )
                await search("test", api_key="real-key-123")

        assert not any("SEARCH_API_KEY" in r.message for r in caplog.records)
