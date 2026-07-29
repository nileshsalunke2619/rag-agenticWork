"""
===============================================================================
File: json_normalizer_dynamic.py

# Uses only Python standard library modules:
# ast
# json
# typing

Purpose
-------
Normalize any LLM response, JSON string format.

Supports
--------
- Valid JSON
- Python dictionary strings
- Escaped JSON
- Markdown wrapped JSON
- Dynamic nested step numbering
- Unlimited nesting depth
- Optional steps field

Output Format
-------------

{
    "answer": {
        "description": "",
        "header": "",
        "question_id": "",
        "references": [],
        "steps": {
            "step_1": "...",
            "step_2": {
                "description": "...",
                "sub_steps": {
                    "step_2.1": "...",
                    "step_2.2": "..."
                }
            }
        },
        "subheader": ""
    }
}

===============================================================================
"""

import ast
import json

from typing import Any
from typing import Dict


# =============================================================================
# DEFAULT RESPONSE
# =============================================================================

def get_default_response() -> Dict[str, Any]:

    return {
        "answer": {
            "description": "",
            "header": "",
            "question_id": "",
            "references": [],
            "subheader": ""
        }
    }


# =============================================================================
# EMBEDDED JSON PARSER
# =============================================================================

def parse_embedded_json(
    value: Any
) -> Any:

    if not isinstance(value, str):
        return value

    text = value.strip()

    # Remove markdown wrappers

    if text.startswith("```json"):
        text = text[len("```json"):]

    elif text.startswith("```"):
        text = text[len("```"):]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    # JSON

    try:
        return json.loads(text)

    except Exception:
        pass

    # Escaped JSON

    try:

        unescaped = bytes(
            text,
            "utf-8"
        ).decode(
            "unicode_escape"
        )

        return json.loads(
            unescaped
        )

    except Exception:
        pass

    # Python dict

    try:
        return ast.literal_eval(text)

    except Exception:
        pass

    # Escaped Python dict

    try:

        unescaped = bytes(
            text,
            "utf-8"
        ).decode(
            "unicode_escape"
        )

        return ast.literal_eval(
            unescaped
        )

    except Exception:
        pass

    return value


# =============================================================================
# DYNAMIC STEP NORMALIZER
# =============================================================================

def normalize_steps(
    steps: Any,
    parent_number: str = ""
) -> Dict[str, Any]:
    """
    Convert steps into dynamically numbered hierarchy.

    Example:

    [
        "Step A",

        {
            "Step B": [
                "Step B1",
                "Step B2",

                {
                    "Step B3": [
                        "Step B3.1"
                    ]
                }
            ]
        }
    ]
    """

    if not steps:
        return {}

    result = {}

    # -------------------------------------------------------------------------
    # LIST
    # -------------------------------------------------------------------------

    if isinstance(
        steps,
        list
    ):

        for index, item in enumerate(
            steps,
            start=1
        ):

            step_number = (
                f"{parent_number}.{index}"
                if parent_number
                else str(index)
            )

            step_key = (
                f"step_{step_number}"
            )

            # Simple text step

            if isinstance(
                item,
                str
            ):

                text = item.strip()

                if text:

                    result[
                        step_key
                    ] = text

            # Nested structure

            elif isinstance(
                item,
                dict
            ):

                for title, child_steps in item.items():

                    nested_steps = (
                        normalize_steps(
                            child_steps,
                            step_number
                        )
                    )

                    data = {
                        "description":
                            str(title)
                    }

                    if nested_steps:

                        data[
                            "sub_steps"
                        ] = nested_steps

                    result[
                        step_key
                    ] = data

        return result

    # -------------------------------------------------------------------------
    # DICTIONARY
    # -------------------------------------------------------------------------

    if isinstance(
        steps,
        dict
    ):

        for index, (
            title,
            child_steps
        ) in enumerate(
            steps.items(),
            start=1
        ):

            step_number = (
                f"{parent_number}.{index}"
                if parent_number
                else str(index)
            )

            step_key = (
                f"step_{step_number}"
            )

            nested_steps = (
                normalize_steps(
                    child_steps,
                    step_number
                )
            )

            data = {
                "description":
                    str(title)
            }

            if nested_steps:

                data[
                    "sub_steps"
                ] = nested_steps

            result[
                step_key
            ] = data

        return result

    # -------------------------------------------------------------------------
    # SINGLE STRING
    # -------------------------------------------------------------------------

    text = str(
        steps
    ).strip()

    if text:

        if parent_number:
            key = (
                f"step_{parent_number}"
            )
        else:
            key = "step_1"

        result[key] = text

    return result


# =============================================================================
# ANSWER EXTRACTION
# =============================================================================

def extract_answer_object(
    parsed: Any
):

    if not isinstance(
        parsed,
        dict
    ):
        return None

    if "answer" not in parsed:
        return None

    answer = parsed["answer"]

    # Embedded JSON string

    if isinstance(
        answer,
        str
    ):

        parsed_answer = (
            parse_embedded_json(
                answer
            )
        )

        if (
            isinstance(
                parsed_answer,
                dict
            )
            and "answer"
            in parsed_answer
        ):

            return parsed_answer[
                "answer"
            ]

    # Direct dict

    if isinstance(
        answer,
        dict
    ):
        return answer

    return None


# =============================================================================
# MAIN NORMALIZER
# =============================================================================

def normalize_response(
    input_data: Any
) -> Dict[str, Any]:

    response = get_default_response()

    # -------------------------------------------------------------------------
    # NONE
    # -------------------------------------------------------------------------

    if input_data is None:

        response["answer"][
            "description"
        ] = "Empty response"

        return response

    parsed = None

    # -------------------------------------------------------------------------
    # DICT
    # -------------------------------------------------------------------------

    if isinstance(
        input_data,
        dict
    ):

        parsed = input_data

    # -------------------------------------------------------------------------
    # STRING
    # -------------------------------------------------------------------------

    else:

        text = str(
            input_data
        ).strip()

        try:

            parsed = json.loads(
                text
            )

        except Exception:

            try:

                parsed = ast.literal_eval(
                    text
                )

            except Exception:

                response["answer"][
                    "description"
                ] = (
                    f"{text}"
                )

                return response

    # -------------------------------------------------------------------------
    # EXTRACT ANSWER
    # -------------------------------------------------------------------------

    answer = extract_answer_object(
        parsed
    )

    if answer is None:

        response["answer"][
            "description"
        ] = str(parsed)

        return response

    # -------------------------------------------------------------------------
    # REFERENCES
    # -------------------------------------------------------------------------

    references = answer.get(
        "references",
        []
    )

    if not isinstance(
        references,
        list
    ):

        references = [
            str(references)
        ]

    # -------------------------------------------------------------------------
    # BASE RESPONSE
    # -------------------------------------------------------------------------

    response["answer"] = {

        "description":
            str(
                answer.get(
                    "description",
                    ""
                )
            ),

        "header":
            str(
                answer.get(
                    "header",
                    ""
                )
            ),

        "question_id":
            str(
                answer.get(
                    "question_id",
                    ""
                )
            ),

        "references":
            references
    }

    # -------------------------------------------------------------------------
    # STEPS (OPTIONAL)
    # -------------------------------------------------------------------------

    normalized_steps = (
        normalize_steps(
            answer.get(
                "steps",
                None
            )
        )
    )

    if normalized_steps:

        response["answer"][
            "steps"
        ] = normalized_steps

    # -------------------------------------------------------------------------
    # SUBHEADER LAST
    # -------------------------------------------------------------------------

    response["answer"][
        "subheader"
    ] = str(
        answer.get(
            "subheader",
            ""
        )
    )

    return response


# =============================================================================
# JSON STRING OUTPUT
# =============================================================================

def get_json_string(
    input_data: Any,
    indent: int = 4
) -> str:

    return json.dumps(
        normalize_response(
            input_data
        ),
        indent=indent,
        ensure_ascii=False
    )
