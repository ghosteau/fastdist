"""
Hold the public docstrings to the behavior they describe.

Two problems this catches, both of which were live when the file was written:

  * A worked example that states the wrong answer. Fourteen of the twenty-five
    examples in `normal.py` and `uniform.py` were wrong, including a CGF value
    wrong in its fourth significant digit, because the section was headed
    `Example` rather than `Examples` and so no tool ever ran them. Four more in
    `exponential.py` were collectable but had never been run.

  * A missing docstring on a public entry point, which `pydoc` and Sphinx both
    render as a blank.

The first is checked by executing every example; the second by walking the
classes. Neither needs updating when a distribution is added, so a new class
arrives already covered.
"""

import ast
import doctest
import importlib
import inspect
import pkgutil

import pytest

import fastdist
import fastdist.distributions as distributions

# Every module under fastdist.distributions, discovered rather than listed, so
# a new distribution is covered the moment it is importable.
MODULE_NAMES = sorted(
    info.name for info in pkgutil.iter_modules(distributions.__path__)
)

# Classes exported from the package root.
EXPORTED = sorted(
    name for name in fastdist.__all__
    if name[:1].isupper() and inspect.isclass(getattr(fastdist, name))
)


def _public_methods(cls):
    """Methods a user can reach: no dunders, no property accessors."""
    for name, member in vars(cls).items():
        if name.startswith("__"):
            continue
        if isinstance(member, property):
            continue
        fn = member.__func__ if isinstance(member, (classmethod, staticmethod)) else member
        if inspect.isfunction(fn):
            yield name, fn


# ---------------------------------------------------------------------------
# Worked examples must produce what they claim
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_docstring_examples_are_accurate(module_name):
    """
    Run every doctest in the module and fail on any mismatch.

    This is the check that was missing: an example stating the wrong number is
    worse than no example, because a reader has no reason to doubt it.
    """
    module = importlib.import_module(f"fastdist.distributions.{module_name}")
    results = doctest.testmod(module, verbose=False, report=False,
                              optionflags=doctest.ELLIPSIS)
    assert results.failed == 0, (
        f"{results.failed} of {results.attempted} docstring examples in "
        f"{module_name}.py do not produce what they claim"
    )


def test_some_examples_are_actually_present():
    """
    Guard the guard: if the examples stopped being collected, the test above
    would pass by vacuously running nothing.

    This is exactly how the earlier errors stayed hidden -- the section was
    headed `Example` rather than `Examples`, so there was nothing to run.
    """
    attempted = 0
    for module_name in MODULE_NAMES:
        module = importlib.import_module(f"fastdist.distributions.{module_name}")
        attempted += doctest.testmod(module, verbose=False, report=False,
                                     optionflags=doctest.ELLIPSIS).attempted
    assert attempted >= 20, f"only {attempted} examples collected; are the sections headed 'Examples'?"


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_example_sections_are_plural(module_name):
    """
    NumPy style spells the section `Examples`. The singular form renders as a
    plain heading and is silently skipped by every doctest runner.
    """
    module = importlib.import_module(f"fastdist.distributions.{module_name}")
    source = inspect.getsource(module)
    offenders = [
        i + 1 for i, line in enumerate(source.split("\n"))
        if line.strip() == "Example"
    ]
    assert not offenders, (
        f"{module_name}.py uses the singular 'Example' heading at line(s) "
        f"{offenders}; NumPy style is 'Examples', and the singular form is "
        f"never executed"
    )


# ---------------------------------------------------------------------------
# Public entry points must be documented
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("class_name", EXPORTED)
def test_exported_classes_have_docstrings(class_name):
    """Every class reachable as fastdist.X carries a class-level docstring."""
    cls = getattr(fastdist, class_name)
    assert cls.__doc__ and cls.__doc__.strip(), f"{class_name} has no class docstring"


@pytest.mark.parametrize("class_name", EXPORTED)
def test_public_methods_have_docstrings(class_name):
    """
    Every method on an exported class is documented, including the underscore
    classmethods: they are part of the published surface and appear in the
    generated reference.
    """
    cls = getattr(fastdist, class_name)
    undocumented = [
        name for name, fn in _public_methods(cls)
        if not (fn.__doc__ and fn.__doc__.strip())
    ]
    assert not undocumented, f"{class_name} has undocumented methods: {sorted(undocumented)}"


@pytest.mark.parametrize("class_name", EXPORTED)
def test_docstring_sections_are_well_formed(class_name):
    """
    Each NumPy section heading must be followed by a dashed underline of the
    same length, or numpydoc renders the section as body text.
    """
    sections = {"Parameters", "Returns", "Raises", "Notes", "Attributes",
                "Examples", "See Also", "Yields", "Warns", "References"}
    cls = getattr(fastdist, class_name)

    problems = []
    docs = [("<class>", cls.__doc__)] + [(n, f.__doc__) for n, f in _public_methods(cls)]
    for owner, doc in docs:
        if not doc:
            continue
        lines = doc.split("\n")
        for i, line in enumerate(lines):
            heading = line.strip()
            if heading not in sections:
                continue
            underline = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if set(underline) != {"-"} or len(underline) != len(heading):
                problems.append(f"{owner}: '{heading}' underlined with {underline!r}")

    assert not problems, f"{class_name}: " + "; ".join(problems)


# ---------------------------------------------------------------------------
# The C++ core's own bindings
# ---------------------------------------------------------------------------

def test_every_core_binding_has_a_docstring():
    """
    The pybind11 bindings carry their own docstrings, which is what `help()`
    shows for fastdist._fastdist. A binding added without one shows a bare
    signature.
    """
    core = fastdist._fastdist
    undocumented = []
    for name in dir(core):
        if name.startswith("_"):
            continue
        member = getattr(core, name)
        if not callable(member):
            continue
        doc = (member.__doc__ or "").strip()
        # pybind11 prepends the signature; a documented binding says more.
        body = "\n".join(line for line in doc.split("\n")
                         if not line.startswith(name + "("))
        if not body.strip():
            undocumented.append(name)
    assert not undocumented, f"core bindings without a docstring: {sorted(undocumented)}"
