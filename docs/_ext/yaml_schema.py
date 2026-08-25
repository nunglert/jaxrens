"""Render pydantic config specs as YAML-keyed reference tables.

The ``jaxrens`` YAML schema lives in ``jaxrens.cli.schema.*`` as pydantic
models.  ``autodoc_pydantic`` documents those models faithfully — but it
documents them as *Python classes*: headings read
``pydantic model jaxrens.cli.schema.moves.GMCMoveSpec`` and fields render as
``field n_reflect: int = 5``.  A reader writing a config file is looking for
``type: gmc`` and ``n_reflect:``, and finds neither.

This extension provides the complementary view.  It walks the same models and
emits tables keyed by the YAML names, so the entry-point page can never drift
from the code while still reading like documentation for a config file.  The
exhaustive class-oriented dump stays available on its own page.

Two directives::

    .. yaml-section:: jaxrens.cli.schema.run.RunSpec
       :prefix: run

    .. yaml-variants:: jaxrens.cli.schema.moves.MoveSpec
       :prefix: moves[]
       :own-fields:

``yaml-section`` renders one model as a single table.  ``yaml-variants``
takes a discriminated union and renders one ``tab-item`` per variant, labelled
with the variant's discriminator value (``gmc``, ``npt``, …) rather than its
class name.

Options
-------
``:prefix:``
    YAML key path the fields hang off, prepended to each key in the table.
``:exclude:``
    Comma-separated field names to omit.
``:own-fields:``
    Flag.  Skip fields inherited from a base class, so a variant table shows
    only what is specific to that variant and the shared fields can be
    documented once, above the tab set.
``:no-docstring:``
    Flag.  Suppress the model docstring that is otherwise emitted above the
    table.
"""

from __future__ import annotations

import importlib
import inspect
import re
import types
import typing
from pathlib import Path
from typing import Any, Literal, Union

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

# ---------------------------------------------------------------------------
# Type / default formatting
# ---------------------------------------------------------------------------

_SCALARS = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    Path: "path",
    type(None): "null",
}


def _fmt_type(annotation: Any) -> str:
    """Render a field annotation in YAML terms rather than Python ones."""
    if annotation in _SCALARS:
        return _SCALARS[annotation]

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is Literal:
        return " | ".join(f"``{a}``" for a in args)

    if origin is typing.Annotated:
        return _fmt_type(args[0])

    if origin in (Union, getattr(types, "UnionType", None)):
        # A discriminated union of spec models is *the* common case here
        # (moves, backends, ensembles, …).  Spelling out all ten variants
        # makes the Type column taller than the description it sits next
        # to, and says nothing the section below does not say better — the
        # variants are enumerated there, one tab each.
        if _is_model_union(args):
            return "mapping"
        parts = [_fmt_type(a) for a in args]
        # ``X | None`` reads better as "X or null" than as a bare union.
        if "null" in parts:
            parts = [p for p in parts if p != "null"] + ["null"]
        return " or ".join(dict.fromkeys(parts))

    if origin in (list, set, frozenset):
        if not args:
            return "list"
        inner = _fmt_type(args[0])
        return "list of mappings" if inner == "mapping" else f"list of {inner}"

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return f"list of {_fmt_type(args[0])}"
        return "[" + ", ".join(_fmt_type(a) for a in args) + "]"

    if origin is dict:
        if args:
            value = _fmt_type(args[1])
            key = "name" if args[0] is str else _fmt_type(args[0])
            return f"mapping of {key} to {value}"
        return "mapping"

    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return "mapping"

    return f"``{getattr(annotation, '__name__', annotation)}``"


def _is_model_union(args: tuple[Any, ...]) -> bool:
    """True when every non-``None`` member of a union is a spec model."""
    members = [a for a in args if a is not type(None)]
    return bool(members) and all(
        inspect.isclass(a) and issubclass(a, BaseModel) for a in members
    )


def _discriminator_value(model: type[BaseModel]) -> str | None:
    """Return the ``type:`` literal a variant is selected by, if any."""
    field = model.model_fields.get("type")
    if field is None:
        return None
    if typing.get_origin(field.annotation) is Literal:
        return str(typing.get_args(field.annotation)[0])
    return None


_CONSTRAINT_OPS = {
    "ge": ">=",
    "gt": ">",
    "le": "<=",
    "lt": "<",
    "multiple_of": "multiple of",
    "min_length": "min length",
    "max_length": "max length",
}


def _fmt_constraints(field: Any) -> str:
    """Render a field's ``annotated_types`` bounds, e.g. ``(>= 1)``.

    Only bounds declared on the ``Field`` itself surface here.  Most of this
    schema's validation lives in ``@model_validator`` / ``@field_validator``
    methods instead (the ascending-ladder check, the exactly-one-source rule,
    the soft-core mutex), which no generic renderer can summarise — those are
    described in prose on the field or section they govern.
    """
    parts = []
    for meta in field.metadata:
        for attr, op in _CONSTRAINT_OPS.items():
            value = getattr(meta, attr, None)
            if value is not None:
                parts.append(f"{op} {value}")
    return f" ({', '.join(parts)})" if parts else ""


def _fmt_default(field: Any) -> str:
    """Render a field default as it would be written in YAML."""
    if field.is_required():
        return "*required*"
    if field.default_factory is not None:
        try:
            value = field.default_factory()
        except TypeError:  # factory wants validated data; not our case
            return "—"
        if isinstance(value, BaseModel):
            return "*section defaults*"
        if value == [] or value == {}:
            return "*empty*"
        return f"``{value!r}``"
    default = field.default
    if default is PydanticUndefined:
        return "*required*"
    if default is None:
        return "``null``"
    if isinstance(default, bool):
        return f"``{str(default).lower()}``"
    if isinstance(default, Path):
        return f"``{str(default)}``"
    if isinstance(default, str):
        return f"``{default}``"
    if isinstance(default, (tuple, list)):
        # YAML has no tuple syntax; ``(1, 1, 1)`` is not something a user
        # can type into a config file, ``[1, 1, 1]`` is.
        inner = ", ".join(repr(v) for v in default)
        return f"``[{inner}]``"
    return f"``{default!r}``"


# ---------------------------------------------------------------------------
# reST emission
# ---------------------------------------------------------------------------


def _own_field_names(model: type[BaseModel]) -> set[str]:
    """Field names declared on ``model`` itself, not on any base."""
    inherited: set[str] = set()
    for base in model.__mro__[1:]:
        if inspect.isclass(base) and issubclass(base, BaseModel):
            inherited |= set(base.model_fields)
    return set(model.model_fields) - inherited


def _rows(
    model: type[BaseModel],
    *,
    prefix: str,
    exclude: set[str],
    own_only: bool,
) -> list[tuple[str, str, str, str]]:
    keep = _own_field_names(model) if own_only else set(model.model_fields)
    rows = []
    for name, field in model.model_fields.items():
        if name in exclude or name not in keep:
            continue
        key = f"{prefix}.{name}" if prefix else name
        rows.append(
            (
                f"``{key}``",
                _fmt_type(field.annotation) + _fmt_constraints(field),
                _fmt_default(field),
                (field.description or "").strip(),
            )
        )
    return rows


def _table_lines(rows: list[tuple[str, str, str, str]]) -> list[str]:
    """Emit a ``list-table`` — robust against long, marked-up cell text."""
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 22 16 16 46",
        "   :class: yaml-schema-table",
        "",
        "   * - Key",
        "     - Type",
        "     - Default",
        "     - Description",
    ]
    for key, type_, default, desc in rows:
        lines += [
            f"   * - {key}",
            f"     - {type_}",
            f"     - {default}",
            f"     - {desc or '—'}",
        ]
    lines.append("")
    return lines


_YAML_START = re.compile(r"^\s*(-\s*\{|[A-Za-z_][\w.]*\s*:)")


def _promote_yaml_blocks(lines: list[str]) -> list[str]:
    """Turn ``::`` literal blocks holding YAML into highlighted code blocks.

    The spec docstrings introduce their config examples with plain reST
    literal blocks, which is the right thing for a docstring — ``help()``
    shows readable text, not directive syntax.  Rendered, though, those
    blocks fall to the ``default`` lexer and come out unhighlighted, which
    on a page that is entirely about YAML is a wasted cue.  Rewriting the
    marker here keeps the docstrings idiomatic and the page highlighted.

    Only blocks whose first content line looks like YAML are promoted, so a
    literal block holding anything else is left alone.
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.rstrip().endswith("::"):
            out.append(line)
            i += 1
            continue

        # Find the block this marker opens: the following indented run,
        # after any blank lines.
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        indent = (
            len(lines[j]) - len(lines[j].lstrip()) if j < len(lines) else 0
        )
        if j >= len(lines) or indent == 0 or not _YAML_START.match(lines[j]):
            out.append(line)
            i += 1
            continue

        stripped = line.rstrip()
        if stripped == "::":
            marker_indent = len(line) - len(line.lstrip())
        else:
            # ``Some prose::`` renders as ``Some prose:`` plus a block.
            out.append(stripped[:-1])
            out.append("")
            marker_indent = len(line) - len(line.lstrip())
        pad = " " * marker_indent
        out.append(f"{pad}.. code-block:: yaml")
        out.append("")
        i += 1
    return out


def _split_docstring(model: type[BaseModel]) -> tuple[list[str], list[str]]:
    """Split a docstring into its summary line and the rest.

    A variant panel reads best as *summary, then parameter table, then the
    detail* — the reader wants to know what the move is and which keys it
    takes before wading into caveats.  Splitting on the PEP 257 summary
    paragraph gets that ordering out of an ordinary docstring.

    The remainder is kept intact rather than reordered further: several
    specs introduce their YAML example with a trailing ``::`` on the
    preceding sentence, so lifting the example out from under it would
    leave a literal-block marker with nothing after it.
    """
    doc = inspect.getdoc(model)
    if not doc:
        return [], []
    lines = doc.splitlines()
    try:
        split = lines.index("")
    except ValueError:
        return lines + [""], []
    summary = lines[:split] + [""]
    body = _promote_yaml_blocks(lines[split + 1 :])
    while body and not body[0].strip():
        body.pop(0)
    return summary, (body + [""] if body else [])


def _resolve(path: str) -> Any:
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr), module


class _Base(Directive):
    required_arguments = 1
    has_content = False
    option_spec = {
        "prefix": directives.unchanged,
        "exclude": directives.unchanged,
        "own-fields": directives.flag,
        "no-docstring": directives.flag,
    }

    def _parse(self, lines: list[str]):
        container = nodes.container()
        self.state.nested_parse(
            StringList(lines, source=""), self.content_offset, container
        )
        return container.children

    def _note_dependency(self, module) -> None:
        """Rebuild this page when the schema module itself changes.

        Sphinx tracks .rst mtimes; without this a description edited in
        ``cli/schema/*.py`` would not invalidate the generated page.
        """
        source = getattr(module, "__file__", None)
        if source:
            self.state.document.settings.env.note_dependency(source)

    @property
    def _exclude(self) -> set[str]:
        raw = self.options.get("exclude", "")
        return {x.strip() for x in raw.split(",") if x.strip()}


class YamlSectionDirective(_Base):
    """Render one pydantic model as a YAML-keyed table."""

    def run(self):
        model, module = _resolve(self.arguments[0])
        self._note_dependency(module)
        summary, body = (
            ([], [])
            if "no-docstring" in self.options
            else _split_docstring(model)
        )
        lines: list[str] = list(summary)
        lines += _table_lines(
            _rows(
                model,
                prefix=self.options.get("prefix", ""),
                exclude=self._exclude,
                own_only="own-fields" in self.options,
            )
        )
        lines += body
        return self._parse(lines)


class YamlVariantsDirective(_Base):
    """Render a discriminated union as one tab per ``type:`` value."""

    def run(self):
        union, module = _resolve(self.arguments[0])
        self._note_dependency(module)

        args = typing.get_args(union)
        variants = [v for v in typing.get_args(args[0]) if inspect.isclass(v)]
        if not variants:  # a single-member Union collapses to the class
            variants = [args[0]]

        prefix = self.options.get("prefix", "")
        # The class is what docs/_static/yaml-schema.css hooks onto to draw a
        # visible panel around the selected variant.  Without it a long
        # variant (``gmc``) reads as if it continued into the next section:
        # sphinx-design's default panel has no border of its own.
        lines: list[str] = [
            ".. tab-set::",
            "   :class: yaml-variant-tabs",
            "",
        ]
        for variant in variants:
            label = _discriminator_value(variant) or variant.__name__
            lines += [f"   .. tab-item:: {label}", ""]
            summary, detail = (
                ([], [])
                if "no-docstring" in self.options
                else _split_docstring(variant)
            )
            body: list[str] = list(summary)
            rows = _rows(
                variant,
                prefix=prefix,
                exclude=self._exclude | {"type"},
                own_only="own-fields" in self.options,
            )
            if rows:
                body += _table_lines(rows)
            else:
                body += [
                    f"Takes no keys of its own — ``type: {label}`` plus the "
                    "shared keys above.",
                    "",
                ]
            body += detail
            lines += [f"      {line}" if line else "" for line in body]
        return self._parse(lines)


def setup(app):
    app.add_directive("yaml-section", YamlSectionDirective)
    app.add_directive("yaml-variants", YamlVariantsDirective)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
