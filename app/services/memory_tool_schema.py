from __future__ import annotations

from google.genai import types

from app.services.memory_repository import MEMORY_TYPES


def build_memory_tools() -> list[types.Tool]:
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="fetch_recent_special_events",
                    description="Fetch recent special events for the current user.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "description": "How many recent special events to return.",
                            }
                        },
                        "additionalProperties": False,
                    },
                    response_json_schema={
                        "type": "object",
                        "properties": {
                            "events": {"type": "array", "items": {"type": "object"}}
                        },
                    },
                ),
                types.FunctionDeclaration(
                    name="fetch_recent_diary_summaries",
                    description="Fetch recent diary summaries for the current user.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                                "description": "How many recent diary summaries to return.",
                            }
                        },
                        "additionalProperties": False,
                    },
                    response_json_schema={
                        "type": "object",
                        "properties": {
                            "summaries": {"type": "array", "items": {"type": "object"}}
                        },
                    },
                ),
                types.FunctionDeclaration(
                    name="search_memory",
                    description="Search stored user memories by keyword.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search keyword to match against stored memories.",
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                                "description": "Maximum number of results to return.",
                            },
                            "types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional list of memory types to filter by.",
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                        "propertyOrdering": ["query", "types", "limit"],
                    },
                    response_json_schema={
                        "type": "object",
                        "properties": {
                            "results": {"type": "array", "items": {"type": "object"}}
                        },
                    },
                ),
                types.FunctionDeclaration(
                    name="search_diary_summaries",
                    description="Search diary summaries by keyword.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search keyword to match diary summaries.",
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                                "description": "Maximum number of summaries to return.",
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                        "propertyOrdering": ["query", "limit"],
                    },
                    response_json_schema={
                        "type": "object",
                        "properties": {
                            "results": {"type": "array", "items": {"type": "object"}}
                        },
                    },
                ),
                types.FunctionDeclaration(
                    name="search_memory_by_period",
                    description=(
                        "Search memories with event_date within a date range. "
                        "Use this for schedule questions such as next week plans."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "start_date": {
                                "type": "string",
                                "description": "Start date in YYYY-MM-DD.",
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date in YYYY-MM-DD.",
                            },
                            "types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Optional memory types. Example: REMINDER, GOAL, SPECIAL_EVENT."
                                ),
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                                "description": "Maximum number of results to return.",
                            },
                        },
                        "required": ["start_date", "end_date"],
                        "additionalProperties": False,
                        "propertyOrdering": [
                            "start_date",
                            "end_date",
                            "types",
                            "limit",
                        ],
                    },
                    response_json_schema={
                        "type": "object",
                        "properties": {
                            "results": {"type": "array", "items": {"type": "object"}}
                        },
                    },
                ),
                types.FunctionDeclaration(
                    name="store_memory",
                    description="Store a new memory about the user.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "memory_type": {
                                "type": "string",
                                "enum": sorted(MEMORY_TYPES),
                                "description": "Category of memory to store.",
                            },
                            "title": {
                                "type": "string",
                                "description": "Short label or title for this memory.",
                            },
                            "content": {
                                "type": "string",
                                "description": "The full memory content to store. Required.",
                            },
                            "event_date": {
                                "type": "string",
                                "description": "YYYY-MM-DD",
                            },
                            "importance": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                                "description": "Importance score from 1 (low) to 5 (high).",
                            },
                            "source": {
                                "type": "string",
                                "enum": ["USER", "AI", "SYSTEM"],
                                "description": "Who provided this memory.",
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional structured metadata for the memory.",
                            },
                        },
                        "required": ["memory_type", "content"],
                        "additionalProperties": False,
                        "propertyOrdering": [
                            "memory_type",
                            "content",
                            "title",
                            "event_date",
                            "importance",
                            "source",
                            "metadata",
                        ],
                    },
                    response_json_schema={
                        "type": "object",
                        "properties": {"stored": {"type": "object"}},
                    },
                ),
            ]
        )
    ]
