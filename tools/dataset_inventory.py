
import argparse
import bisect
import json
import math
from pathlib import Path
from collections import Counter, defaultdict


FEATURE_COUNT = 17
MIN_SEQUENCE_LENGTH = 15
OPPOSITION_INDEXES = (5, 9, 13, 16)


def source_locations(text):
    """Map JSON value paths to their one-based source lines."""
    decoder = json.JSONDecoder()
    locations = {}
    line_starts = [-1] + [index for index, char in enumerate(text) if char == '\n']

    def whitespace(pos):
        while pos < len(text) and text[pos] in ' \t\r\n':
            pos += 1
        return pos

    def visit(pos, path):
        pos = whitespace(pos)
        locations[path] = bisect.bisect_left(line_starts, pos)
        if text[pos] == '{':
            pos = whitespace(pos + 1)
            while text[pos] != '}':
                key, pos = decoder.raw_decode(text, pos)
                pos = whitespace(pos)
                pos = visit(pos + 1, path + (key,))
                pos = whitespace(pos)
                if text[pos] == ',':
                    pos = whitespace(pos + 1)
                else:
                    break
            return pos + 1
        if text[pos] == '[':
            pos = whitespace(pos + 1)
            index = 0
            while text[pos] != ']':
                pos = visit(pos, path + (index,))
                index += 1
                pos = whitespace(pos)
                if text[pos] == ',':
                    pos = whitespace(pos + 1)
                else:
                    break
            return pos + 1
        return decoder.raw_decode(text, pos)[1]

    visit(0, ())
    return locations


def is_finite_number(value):
    # type() excludes booleans, which Python normally treats as integers.
    return type(value) is int or (
        type(value) is float and math.isfinite(value)
    )


def new_stats():
    """Create an empty set of statistics for one folder."""
    return {
        "files": 0,
        "recordings": 0,
        "sequence_lengths": Counter(),
        "frame_dimensions": Counter(),
        "confidence_counts": Counter({
            "0": 0,
            "1": 0,
            "invalid": 0,
            "missing": 0,
        }),
        "feature_ranges": [
            {"min": None, "max": None}
            for _ in range(FEATURE_COUNT)
        ],
    }


def add_issue(
    issues,
    severity,
    file,
    message,
    recording_index=None,
    frame_index=None,
    feature_index=None,
):
    """Store a validation issue with its location."""
    issue = {
        "severity": severity,
        "file": str(file),
        "recording_index": recording_index,
        "message": message,
    }

    if frame_index is not None:
        issue["frame_index"] = frame_index

    if feature_index is not None:
        issue["feature_index"] = feature_index

    issues.append(issue)


def validate_recording(
    recording,
    file,
    recording_index,
    stats,
    issues,
    opposition_index,
):
    """Validate one recording and update its folder's statistics."""
    stats["recordings"] += 1

    # 1. Recording structure
    if not isinstance(recording, dict):
        stats["confidence_counts"]["missing"] += 1
        add_issue(
            issues, "error", file,
            "Recording is not an object",
            recording_index,
        )
        return

    if not recording:
        add_issue(
            issues, "warning", file,
            "Recording is an empty object",
            recording_index,
        )
        stats["confidence_counts"]["missing"] += 1
        return

    # 2. Required fields
    for field in ("sequenceData", "confidence"):
        if field not in recording:
            add_issue(
                issues, "error", file,
                f"Missing required field: {field}",
                recording_index,
            )

    # 3. Binary confidence label
    if "confidence" not in recording:
        stats["confidence_counts"]["missing"] += 1
    else:
        confidence = recording["confidence"]

        if (
            not is_finite_number(confidence)
            or confidence not in (0, 1)
        ):
            stats["confidence_counts"]["invalid"] += 1
            add_issue(
                issues, "error", file,
                f"confidence must be numeric 0 or 1; got {confidence!r}",
                recording_index,
            )
        else:
            label = str(int(confidence))
            stats["confidence_counts"][label] += 1

    # 4. Sequence structure
    if "sequenceData" not in recording:
        return

    sequence = recording["sequenceData"]

    if not isinstance(sequence, list):
        add_issue(
            issues, "error", file,
            "sequenceData must be a list",
            recording_index,
        )
        return

    sequence_length = len(sequence)
    stats["sequence_lengths"][sequence_length] += 1

    if sequence_length == 0:
        add_issue(
            issues, "error", file,
            "sequenceData is empty",
            recording_index,
        )
        return

    # 5. Short sequences are warnings, not errors
    if sequence_length < MIN_SEQUENCE_LENGTH:
        message = (
            f"Only {sequence_length} frames; "
            "short sequences may not fill a multiclass loader window"
        )

        if sequence_length == 1:
            message += "; one frame cannot support motion-speed calculation"

        add_issue(
            issues, "warning", file,
            message,
            recording_index,
        )

    # 6. Validate individual frames
    for frame_index, frame in enumerate(sequence):

        if not isinstance(frame, list):
            stats["frame_dimensions"]["not a list"] += 1

            add_issue(
                issues, "error", file,
                "Frame is not a list",
                recording_index,
                frame_index,
            )
            continue

        frame_width = len(frame)
        stats["frame_dimensions"][str(frame_width)] += 1

        # 7. Frame dimensions
        if frame_width != FEATURE_COUNT:
            add_issue(
                issues, "error", file,
                f"Expected {FEATURE_COUNT} features, got {frame_width}",
                recording_index,
                frame_index,
            )

        # 8. Validate feature values
        for feature_index, value in enumerate(frame):

            if not is_finite_number(value):
                add_issue(
                    issues, "error", file,
                    f"Feature must be a finite number; got {value!r}",
                    recording_index,
                    frame_index,
                    feature_index,
                )
                continue

            # 9. Track feature minimum and maximum.
            # Only use correctly sized frames.
            if frame_width == FEATURE_COUNT:
                limits = stats["feature_ranges"][feature_index]

                if limits["min"] is None or value < limits["min"]:
                    limits["min"] = value

                if limits["max"] is None or value > limits["max"]:
                    limits["max"] = value

            # 10. Physical plausibility warning
            if feature_index in (OPPOSITION_INDEXES if opposition_index is None else (opposition_index,)) and value < 0:
                add_issue(
                    issues, "warning", file,
                    f"Negative opposition distance: {value}",
                    recording_index,
                    frame_index,
                    feature_index,
                )


def run_self_tests():
    """Small valid and invalid examples for the validation rules."""

    valid_recording = {
        "sequenceData": [[0.0] * 17 for _ in range(15)],
        "confidence": 1,
    }

    def check(recording, expected_errors, expected_warnings=0,
              opposition_index=None):
        found = []

        validate_recording(
            recording,
            "test.json",
            0,
            new_stats(),
            found,
            opposition_index,
        )

        errors = sum(
            issue["severity"] == "error" for issue in found
        )
        warnings = sum(
            issue["severity"] == "warning" for issue in found
        )

        assert errors == expected_errors, found
        assert warnings == expected_warnings, found

    # Valid recording
    check(valid_recording, 0)

    # Incorrect frame width and short sequence
    check(
        {"sequenceData": [[0] * 3], "confidence": 1},
        1, 1,
    )

    # NaN in 15 frames
    check(
        {"sequenceData": [[0] * 16 + [math.nan]] * 15,
         "confidence": 1},
        15,
    )

    # Boolean is not a valid feature number
    check(
        {"sequenceData": [[True] + [0] * 16] * 15,
         "confidence": 1},
        15,
    )

    # Invalid binary label
    check(
        {"sequenceData": [[0] * 17] * 15,
         "confidence": 0.5},
        1,
    )

    # Empty sequence
    check(
        {"sequenceData": [], "confidence": 0},
        1,
    )

    # Structurally valid but unusually short sequence
    check(
        {"sequenceData": [[0] * 17] * 2,
         "confidence": 0},
        0, 1,
    )

    # Negative opposition distance (feature index 1)
    check(
        {"sequenceData": [[0, -1] + [0] * 15] * 15,
         "confidence": 1},
        0, 15,
        opposition_index=1,
    )

    print("All built-in validation tests passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and validate dynamic gesture JSON files."
    )

    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=Path("src/data(Dynamic)"),
        help="Folder containing gesture recordings",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output path for the machine-readable JSON report",
    )

    parser.add_argument(
        "--opposition-index",
        type=int,
        default=None,
        help="0-based index of the opposition-distance feature",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in validation tests",
    )

    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        return

    # Validate arguments
    if not args.data_dir.is_dir():
        parser.error(f"Folder does not exist: {args.data_dir}")

    if args.report is None:
        args.report = Path("dataset_inventory_report.json")
        number = 1
        while args.report.exists():
            args.report = Path(f"dataset_inventory_report_{number}.json")
            number += 1

    if args.report.resolve().is_relative_to(args.data_dir.resolve()):
        parser.error("Report path must be outside the scanned directory")
    if args.report.exists():
        parser.error(f"Report already exists; choose a new --report path: {args.report}")

    if (
        args.opposition_index is not None
        and not 0 <= args.opposition_index < FEATURE_COUNT
    ):
        parser.error(
            f"--opposition-index must be between 0 and {FEATURE_COUNT - 1}"
        )

    print(f"Scanning: {args.data_dir.resolve()}")

    # Exclude the report itself if someone saves it inside data_dir.
    json_files = sorted(
        file for file in args.data_dir.rglob("*.json")
        if file.resolve() != args.report.resolve()
    )

    print(f"Found {len(json_files)} JSON files")

    recording_count = 0
    issues = []

    # defaultdict creates folder statistics when first accessed.
    stats_by_folder = defaultdict(new_stats)

    # Scan every file
    for file_path in json_files:
        short_path = file_path.relative_to(args.data_dir).as_posix()
        folder = file_path.relative_to(args.data_dir).parent.as_posix()

        stats = stats_by_folder[folder]
        stats["files"] += 1

        # JSON -> Python
        try:
            with file_path.open("r", encoding="utf-8") as file:
                text = file.read()
            data = json.loads(text)
            locations = source_locations(text)

        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            add_issue(
                issues, "error", short_path,
                f"Could not read JSON: {error}",
            )
            if isinstance(error, json.JSONDecodeError):
                issues[-1]["line"] = error.lineno
            continue

        # Normalize single objects and lists
        if isinstance(data, dict):
            recordings = [data]

        elif isinstance(data, list):
            recordings = data

        else:
            add_issue(
                issues, "error", short_path,
                "Top-level JSON must be an object or list",
            )
            issues[-1]["line"] = locations[()]
            continue

        recording_count += len(recordings)
        print(f"{short_path}: {len(recordings)} recording(s)")

        if not recordings:
            add_issue(
                issues, "warning", short_path,
                "File contains an empty list of recordings",
            )
            issues[-1]["line"] = locations[()]

        # Validate each recording independently
        for recording_index, recording in enumerate(recordings):
            first_issue = len(issues)
            validate_recording(
                recording,
                short_path,
                recording_index,
                stats,
                issues,
                args.opposition_index,
            )
            base = (recording_index,) if isinstance(data, list) else ()
            for issue in issues[first_issue:]:
                path = base
                if "frame_index" in issue:
                    path += ("sequenceData", issue["frame_index"])
                    if "feature_index" in issue:
                        path += (issue["feature_index"],)
                elif issue["message"].startswith("confidence must"):
                    path += ("confidence",)
                elif issue["message"].startswith(("sequenceData", "Only ")):
                    path += ("sequenceData",)
                issue["line"] = locations.get(path, locations.get(base, 1))

    # Separate errors and warnings
    errors = [
        issue for issue in issues
        if issue["severity"] == "error"
    ]

    warnings = [
        issue for issue in issues
        if issue["severity"] == "warning"
    ]

    invalid_files = sorted({
        issue["file"] for issue in errors
    })

    invalid_recordings = {
        (issue["file"], issue["recording_index"])
        for issue in errors
        if issue["recording_index"] is not None
    }

    # Convert Counters into JSON-compatible dictionaries
    groups = {}

    for folder, stats in sorted(stats_by_folder.items()):
        groups[folder] = {
            "file_count": stats["files"],
            "recording_count": stats["recordings"],
            "sequence_length_distribution": dict(
                sorted(stats["sequence_lengths"].items())
            ),
            "frame_dimension_counts": dict(
                sorted(stats["frame_dimensions"].items())
            ),
            "confidence_label_counts": dict(
                stats["confidence_counts"]
            ),
            "feature_min_max": [
                {
                    "index": index,
                    "min": bounds["min"],
                    "max": bounds["max"],
                }
                for index, bounds in enumerate(stats["feature_ranges"])
            ],
        }

    # Build the machine-readable report
    report = {
        "data_directory": str(args.data_dir.resolve()),
        "opposition_distance_indexes": list(OPPOSITION_INDEXES) if args.opposition_index is None else [args.opposition_index],
        "summary": {
            "file_count": len(json_files),
            "recording_count": recording_count,
            "invalid_file_count": len(invalid_files),
            "invalid_recording_count": len(invalid_recordings),
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
        "invalid_files": invalid_files,
        "groups": groups,
        "issues": issues,
    }

    # Terminal summary
    print("\n==============REPORT_SUMMARY==============")
    print(f"Files in folder: {len(json_files)}")
    print(f"Recording entries: {recording_count}")
    print(f"Invalid files found: {len(invalid_files)}")
    print(f"Invalid recording entries: {len(invalid_recordings)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")


    print("\n==============FOLDER_STATISTICS==============")

    for folder, group in groups.items():
        print(f"\nFolder: {folder}")
        print(f"  Files: {group['file_count']}")
        print(f"  Recordings: {group['recording_count']}")
        print(
            f"  Sequence lengths: "
            f"{group['sequence_length_distribution']}"
        )
        print(
            f"  Frame dimensions: "
            f"{group['frame_dimension_counts']}"
        )
        print(
            f"  Confidence labels: "
            f"{group['confidence_label_counts']}"
        )

        print("  Feature min/max:")
        for feature in group["feature_min_max"]:
            print(
                f"    Feature {feature['index']}: "
                f"{feature['min']} to {feature['max']}"
            )

    # Show only the first five detailed issues
    print("\n==============ISSUES==============")

    for issue in issues[:5]:
        location = issue["file"]

        if "line" in issue:
            location += f", line {issue['line']}"

        if issue["recording_index"] is not None:
            location += f", recording {issue['recording_index']}"

        if "frame_index" in issue:
            location += f", frame {issue['frame_index']}"

        if "feature_index" in issue:
            location += f", feature {issue['feature_index']}"

        print(f"\n[{issue['severity'].upper()}] {location}")
        print(f"Rationale: {issue['message']}")

    if len(issues) > 5:
        print(f"\n{len(issues) - 5} additional issues found")

    if not issues:
        print("No issues found.")

    # Save machine-readable JSON report
    try:
        with args.report.open("x", encoding="utf-8") as file:
            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

    except OSError as error:
        parser.error(f"Could not write report {args.report}: {error}")

    print(f"\nJSON report saved to: {args.report.resolve()}")


if __name__ == "__main__":
    main()
