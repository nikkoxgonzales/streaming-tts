"""Tests for streaming_tts.threadsafe_generators module."""

import pytest
import threading
import time
from streaming_tts.threadsafe_generators import (
    CharIterator,
    AccumulatingThreadSafeGenerator,
)


class TestCharIterator:
    """Tests for CharIterator class."""

    def test_iterate_string(self):
        char_iter = CharIterator()
        char_iter.add("hello")
        result = list(char_iter)
        assert result == ['h', 'e', 'l', 'l', 'o']

    def test_iterate_multiple_strings(self):
        char_iter = CharIterator()
        char_iter.add("hi")
        char_iter.add(" ")
        char_iter.add("there")
        result = list(char_iter)
        assert result == ['h', 'i', ' ', 't', 'h', 'e', 'r', 'e']

    def test_iterate_string_iterator(self):
        def gen():
            yield "ab"
            yield "cd"

        char_iter = CharIterator()
        char_iter.add(gen())
        result = list(char_iter)
        assert result == ['a', 'b', 'c', 'd']

    def test_mixed_items(self):
        def gen():
            yield "bc"

        char_iter = CharIterator()
        char_iter.add("a")
        char_iter.add(gen())
        char_iter.add("d")
        result = list(char_iter)
        assert result == ['a', 'b', 'c', 'd']

    def test_stop_immediately(self):
        char_iter = CharIterator()
        char_iter.add("hello world")
        char_iter.immediate_stop.set()

        result = list(char_iter)
        assert result == []

    def test_stop_midway(self):
        char_iter = CharIterator()
        char_iter.add("hello")

        results = []
        for i, char in enumerate(char_iter):
            results.append(char)
            if i == 2:  # After 'l'
                char_iter.stop()
                break

        assert results == ['h', 'e', 'l']

    def test_iterated_text_accumulates(self):
        char_iter = CharIterator()
        char_iter.add("hello")
        list(char_iter)  # Consume iterator
        assert char_iter.iterated_text == "hello"

    def test_iterated_text_partial(self):
        char_iter = CharIterator()
        char_iter.add("hello")

        # Consume only first 3 characters
        next(char_iter)
        next(char_iter)
        next(char_iter)

        assert char_iter.iterated_text == "hel"

    def test_on_character_callback(self):
        chars_received = []

        def on_char(c):
            chars_received.append(c)

        char_iter = CharIterator(on_character=on_char)
        char_iter.add("abc")
        list(char_iter)

        assert chars_received == ['a', 'b', 'c']

    def test_on_first_text_chunk_callback(self):
        first_chunk_called = []

        def on_first():
            first_chunk_called.append(True)

        char_iter = CharIterator(on_first_text_chunk=on_first)
        char_iter.add("hello")
        list(char_iter)

        assert len(first_chunk_called) == 1

    def test_on_last_text_chunk_callback(self):
        last_chunk_called = []

        def on_last():
            last_chunk_called.append(True)

        char_iter = CharIterator(on_last_text_chunk=on_last)
        char_iter.add("hi")
        list(char_iter)

        assert len(last_chunk_called) == 1

    def test_first_chunk_only_called_once(self):
        first_chunk_calls = []

        def on_first():
            first_chunk_calls.append(True)

        char_iter = CharIterator(on_first_text_chunk=on_first)
        char_iter.add("hello")
        char_iter.add("world")
        list(char_iter)

        assert len(first_chunk_calls) == 1

    def test_empty_items(self):
        char_iter = CharIterator()
        result = list(char_iter)
        assert result == []

    def test_empty_string(self):
        char_iter = CharIterator()
        char_iter.add("")
        result = list(char_iter)
        assert result == []

    def test_iter_returns_self(self):
        char_iter = CharIterator()
        assert iter(char_iter) is char_iter


class TestAccumulatingThreadSafeGenerator:
    """Tests for AccumulatingThreadSafeGenerator class."""

    def test_basic_iteration(self):
        def gen():
            yield "a"
            yield "b"
            yield "c"

        safe_gen = AccumulatingThreadSafeGenerator(gen())
        result = list(safe_gen)
        assert result == ["a", "b", "c"]

    def test_accumulates_tokens(self):
        def gen():
            yield "hello"
            yield " "
            yield "world"

        safe_gen = AccumulatingThreadSafeGenerator(gen())
        list(safe_gen)  # Consume
        assert safe_gen.iterated_text == "hello world"

    def test_accumulated_text_method(self):
        def gen():
            yield "foo"
            yield "bar"

        safe_gen = AccumulatingThreadSafeGenerator(gen())
        list(safe_gen)
        assert safe_gen.accumulated_text() == "foobar"

    def test_is_exhausted_initially_false(self):
        def gen():
            yield "a"

        safe_gen = AccumulatingThreadSafeGenerator(gen())
        assert safe_gen.is_exhausted() is False

    def test_is_exhausted_after_consumption(self):
        def gen():
            yield "a"

        safe_gen = AccumulatingThreadSafeGenerator(gen())
        list(safe_gen)
        assert safe_gen.is_exhausted() is True

    def test_on_first_text_chunk_callback(self):
        first_called = []

        def on_first():
            first_called.append(True)

        def gen():
            yield "a"
            yield "b"

        safe_gen = AccumulatingThreadSafeGenerator(gen(), on_first_text_chunk=on_first)
        list(safe_gen)

        assert len(first_called) == 1

    def test_on_last_text_chunk_callback(self):
        last_called = []

        def on_last():
            last_called.append(True)

        def gen():
            yield "a"
            yield "b"

        safe_gen = AccumulatingThreadSafeGenerator(gen(), on_last_text_chunk=on_last)
        list(safe_gen)

        assert len(last_called) == 1

    def test_thread_safety(self):
        """Test that multiple threads can safely consume from the generator."""
        def gen():
            for i in range(100):
                yield str(i)

        safe_gen = AccumulatingThreadSafeGenerator(gen())
        results = []
        lock = threading.Lock()

        def consumer():
            while True:
                try:
                    token = next(safe_gen)
                    with lock:
                        results.append(token)
                except StopIteration:
                    break

        threads = [threading.Thread(target=consumer) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All tokens should be consumed exactly once
        assert len(results) == 100
        assert set(results) == {str(i) for i in range(100)}

    def test_iter_returns_self(self):
        def gen():
            yield "a"

        safe_gen = AccumulatingThreadSafeGenerator(gen())
        assert iter(safe_gen) is safe_gen

    def test_empty_generator(self):
        def gen():
            return
            yield  # Makes it a generator

        safe_gen = AccumulatingThreadSafeGenerator(gen())
        result = list(safe_gen)
        assert result == []
        assert safe_gen.is_exhausted() is True

    def test_first_chunk_not_called_on_empty(self):
        first_called = []

        def on_first():
            first_called.append(True)

        def gen():
            return
            yield

        safe_gen = AccumulatingThreadSafeGenerator(gen(), on_first_text_chunk=on_first)
        list(safe_gen)

        assert len(first_called) == 0

    def test_last_chunk_not_called_on_empty(self):
        last_called = []

        def on_last():
            last_called.append(True)

        def gen():
            return
            yield

        safe_gen = AccumulatingThreadSafeGenerator(gen(), on_last_text_chunk=on_last)
        list(safe_gen)

        # Last chunk is only called if there was content
        assert len(last_called) == 0

    def test_token_converted_to_string(self):
        def gen():
            yield 123
            yield 456

        safe_gen = AccumulatingThreadSafeGenerator(gen())
        list(safe_gen)
        # Numbers should be converted to strings in accumulated text
        assert "123" in safe_gen.accumulated_text()
        assert "456" in safe_gen.accumulated_text()
