# Security Policy

## Supported versions

fastdist is pre-1.0 and under active development. Only the latest release receives fixes; there are no
maintained back-branches.

| Version | Supported |
|---|---|
| 0.1.x | yes |
| < 0.1 | no |

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Use GitHub's private vulnerability reporting: go to the
[Security tab](https://github.com/ghosteau/fastdist/security/advisories/new) and open a draft advisory.
That keeps the report private to the maintainers until a fix is available.

Include what you have:

- what the issue is and roughly how severe you think it is
- steps to reproduce, ideally a minimal snippet
- affected version, OS, Python version, and whether the build had CUDA enabled
- any crash output or stack trace

Expect an acknowledgement within about a week. Since this is a small volunteer project, please allow
reasonable time for a fix before disclosing publicly.

## Scope

fastdist is a numerical library. It does not handle authentication, network traffic, or untrusted input in
the usual sense, so the realistic security surface is memory safety in the C++ and CUDA layers rather than
anything protocol-level. Reports worth filing privately include:

- out-of-bounds reads or writes reachable from the Python API, including via array inputs, `step_size`, or
  size and shape arguments
- buffer overflows in the CUDA kernels or the batch paths
- use-after-free, double-free, or other memory corruption
- anything that lets crafted input to a public function execute arbitrary code

Out of scope, and better as ordinary public issues:

- numerical inaccuracy, overflow to `inf`/`NaN`, or precision loss that does not corrupt memory — for
  example the known `std::tgamma` overflow in `negative_binomial_pmf_scalar`
- resource exhaustion from deliberately enormous inputs
- crashes only reachable by calling private (`_`-prefixed) functions with invalid arguments
- issues in dependencies; report those upstream

## Disclosure

Once a fix is ready we will publish a GitHub Security Advisory, release a patched version, and note it in
[CHANGELOG.md](CHANGELOG.md). Reporters are credited unless they ask otherwise.
